# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import os

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, Imu

from .aliengo_env_cfg import AliengoFlatEnvCfg
from .modules.mcp import MCP, MCPDims, MCPDataset
from .modules.spawner import QuadrupedSpawner

class FallRecoveryEnv(DirectRLEnv):
    cfg: AliengoFlatEnvCfg

    def __init__(self, cfg: AliengoFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "r_orient_base_gxy",
                "r_orient_upright",
                "r_contact_body",
                "r_contact_feet",
                "r_motion_torque",
                "r_motion_action_smooth",
            ]
        }
        # Allowlist for logging: None means log all computed keys
        try:
            cfg_keys = getattr(self.cfg, "log_metric_keys", None)
            self._log_metric_keys = set(cfg_keys) if cfg_keys is not None else None
        except Exception:
            self._log_metric_keys = None

        # Track total return per episode for logging
        self._episode_return = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._calf_ids, _  = self._contact_sensor.find_bodies(".*calf|.*shank")
        self._undesired_contact_body_ids = self._base_id + self._hip_ids + self._thigh_ids
        
        self._feet_ids_robot, _ = self._robot.find_bodies(".*foot")

        # Precompute laying joint pose tensor for action freezing in final phase
        laying_np = torch.tensor(self.cfg.laying_joint_pos, device=self.device)
        self._laying_joint_pos_tensor = laying_np.unsqueeze(0).repeat(self.num_envs, 1)
        self._phase4_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


        # MCP buffers & net
        self._mcp_dims = MCPDims(o_dim=self.cfg.single_o_dim, m_dim=4, c_dim=13, z_dim=self.cfg.mcp_latent_dim, H=self.cfg.mcp_history)
        if self.cfg.use_mcp:
            self._mcp = MCP(self._mcp_dims).to(self.device)
            self._mcp.dataset = MCPDataset(max_size=400_000, device=self.device)
            self._mcp_opt = torch.optim.Adam(self._mcp.parameters(), lr=self.cfg.mcp_lr)
            # [N, H, 42]
            self._o_hist = torch.zeros(self.num_envs, self._mcp_dims.H, self._mcp_dims.o_dim, device=self.device)

        self.quad_spawner = QuadrupedSpawner(device=self.device, 
                                             env_origins=self._terrain.env_origins, 
                                             default_root_state=self._robot.data.default_root_state, 
                                             default_joint_pos=self._robot.data.default_joint_pos, 
                                             default_joint_vel=self._robot.data.default_joint_vel)

        load_path = getattr(self.cfg, "mcp_load_path", "")
        if load_path:
            ckpt = torch.load(load_path, map_location=self.device)
            self._mcp.load_state_dict(ckpt["state_dict"])
            if "opt_state" in ckpt:
                self._mcp_opt.load_state_dict(ckpt["opt_state"])


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos


    def _apply_action(self):
        if self.cfg.no_action:
            laying_joint_pos = torch.tensor(self.cfg.laying_joint_pos, device=self.device)
            self._robot.set_joint_position_target(laying_joint_pos)
        else:
            targets = self._processed_actions
            if hasattr(self, "_phase4_mask") and self._phase4_mask is not None:
                mask = self._phase4_mask.unsqueeze(1)
                targets = torch.where(mask, self._laying_joint_pos_tensor, targets)
            self._robot.set_joint_position_target(targets)

    def _get_observations(self) -> dict:
        
        # 1) build current proprio o_t (your existing helper)
        o_t = self._build_o_t()  # (N, o_dim)
    
        # Assemble actor obs
        obs_policy = o_t
        observations = {"policy": obs_policy}

        # Critic obs (privileged) if asymmetric PPO
        observations["critic"] = self._get_privileged_observation()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Rewards exactly as in the paper's table:

        Orientation/Posture:
        - Base Orientation:      g_xy^2
        - Upright Orientation:   exp(- (g_z+1)^2 / (2 eps^2))
        - Target Posture:        exp(-||q - q_stand||^2)   only if |g_z+1| < eps

        Contact Management:
        - Feet Contact:          sum_i I_contact_i  (i=1..4)
        - Body Contact:          I_contact (base/hip/thigh)

        Stability Control:
        - Safety Force:          sum_i ||f_i^{xy}||_2      (feet)
        - Body-bias:             clip( ||p_xy - p_xy_init||_2 , 0, 4 )

        Motion Constraints:
        - Position Limits:       sum_i 1[q_i > q_max or q_i < q_min]
        - Angular Vel Limit:     sum_i max(|qd_i| - 0.8, 0)
        - Joint Acceleration:    ||qdd||_2^2
        - Joint Velocity:        ||qd||_2^2
        - Action Smoothing:      ||a_t - a_{t-1}||_2^2
        - Joint Torques:         ||tau||_2^2
        """
        dt = self.step_dt

        if not hasattr(self, "_init_root_xy"):
            self._init_root_xy = self._robot.data.root_state_w[:, :2].clone()
        g_b  = self._robot.data.projected_gravity_b             # (N,3)
        q    = self._robot.data.joint_pos                       # (N,J)
        qd   = self._robot.data.joint_vel                       # (N,J)
        qdd  = self._robot.data.joint_acc                       # (N,J)
        tau  = self._robot.data.applied_torque                  # (N,J)
        a_t   = self._actions
        a_tm1 = self._previous_actions

        # current contact forces
        F_w   = self._contact_sensor.data.net_forces_w          # (N, bodies, 3)
        Fmag  = torch.linalg.norm(F_w, dim=-1)                  # (N, bodies)

        # feet contact (binary) and horizontal forces for safety-force
        feet_ids = self._feet_ids
        foot_F   = F_w[:, feet_ids, :2]                         # (N,4,2)
        safety_force = torch.norm(foot_F, dim=-1).sum(dim=1)    # Σ ||f_i^xy||, (N,)
        feet_contact = (Fmag[:, feet_ids].max(dim=1).values > 1.0).float()  # (N,)

        # body contact (base + hips + thighs)
        undesired_ids = self._undesired_contact_body_ids
        body_contact = (Fmag[:, undesired_ids].max(dim=1).values > 1.0).float()  # (N,)

        # body-bias distance in XY (clipped to 4)
        p_xy   = self._robot.data.root_state_w[:, :2]
        disp   = torch.norm(p_xy - self._init_root_xy, dim=1)                   # (N,)
        body_bias = torch.clamp(disp, 0.0, 4.0)

        # position limits (count of violations)
        pos_limit_term = torch.zeros(self.num_envs, device=self.device)
        limits = getattr(self._robot.data, "joint_pos_limits", None)
        if limits is not None:
            limits = torch.as_tensor(limits, device=self.device, dtype=q.dtype)
            if limits.dim() == 2:  # (J,2) -> (N,J,2)
                limits = limits.unsqueeze(0).expand(self.num_envs, -1, -1)
            lower, upper = limits[..., 0], limits[..., 1]
            pos_limit_term = ((q < lower) | (q > upper)).float().sum(dim=1)

        # angular velocity limit excess over 0.8 rad/s
        qd_excess = torch.clamp(qd.abs() - 0.8, min=0.0).sum(dim=1)

        # quadratic costs
        qdd_sq = (qdd ** 2).sum(dim=1) # joint acceleration squared
        qd_sq  = (qd  ** 2).sum(dim=1) # joint velocity squared
        act_smooth = ((a_t - a_tm1) ** 2).sum(dim=1) # action smoothing squared
        tau_sq = (tau ** 2).sum(dim=1) # joint torque squared

        gxy_sq = (g_b[:, :2] ** 2).sum(dim=1)
        gz = g_b[:, 2]
        eps_upright = float(getattr(self.cfg, "upright_eps", 0.10))
        upright = torch.exp(-((gz + 1.0) ** 2) / (2.0 * eps_upright * eps_upright))
        near_upright = (gz + 1.0).abs() < eps_upright
        # Stance phase definition - simple upright hold
        if not hasattr(self, "_upright_hold_counter") or self._upright_hold_counter.shape[0] != self.num_envs:
            self._upright_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Stance phase: robot holds upright for a short period
        hold_upright_steps = max(1, int(0.3 / float(self.step_dt)))  # ~0.3s hold requirement
        self._upright_hold_counter = torch.where(near_upright, self._upright_hold_counter + 1, torch.zeros_like(self._upright_hold_counter))
        stance_phase = self._upright_hold_counter >= hold_upright_steps

        q_laying = torch.tensor(self.cfg.laying_joint_pos, device=self.device)
        posture_err_sq = ((q - q_laying) ** 2).sum(dim=1)
        
        # Phased approach: 1) Calves first, 2) Upright, 3) Full pose
        # Phase 1: Focus on calves (KFE joints) - indices 8-11
        calf_indices = [8, 9, 10, 11]
        calf_err = q[:, calf_indices] - q_laying[calf_indices]
        calf_err_sq = calf_err ** 2
        calf_mean = calf_err_sq.mean(dim=1)
        calf_max = calf_err_sq.max(dim=1).values
        front_calf_err = calf_err_sq[:, :2].max(dim=1).values  # FL, FR
        rear_calf_err = calf_err_sq[:, 2:].max(dim=1).values   # RL, RR
        fl_calf_err = calf_err_sq[:, 0]
        fr_calf_err = calf_err_sq[:, 1]
        # Front-right full leg pose error (hip/thigh/calf)
        fr_leg_err = (
            (q[:, 1] - q_laying[1]) ** 2 +
            (q[:, 5] - q_laying[5]) ** 2 +
            (q[:, 9] - q_laying[9]) ** 2
        )
        fr_leg_good = fr_leg_err < 0.02
        calves_close = calf_max < 0.01  # ensure every calf joint is close (~0.1 rad)
        phase1_active = ~calves_close

        # Phase 2: Once calves are good, focus on upright
        phase2_active = calves_close & ~near_upright  # Phase 2: Calves good, but not upright
        phase3_active = calves_close & near_upright & fr_leg_good  # Phase 3: Calves good, upright, FR leg aligned

        # Phase 1: Reward for contracting calves
        calf_reward = torch.exp(-calf_mean / 0.02) * phase1_active.float()
        calf_penalty = calf_max  # persistent penalty if any calf drifts away from target
        front_calf_penalty = front_calf_err + 0.5 * (fl_calf_err + fr_calf_err)
        rear_calf_penalty = rear_calf_err
        fl_calf_reward = torch.exp(-fl_calf_err / 0.01) * phase1_active.float()
        fr_calf_reward = torch.exp(-(fr_calf_err + fr_leg_err) / 0.008) * phase1_active.float()
        fr_leg_reward = torch.exp(-fr_leg_err / 0.015) * calves_close.float()
        fr_leg_penalty = fr_leg_err
        
        # Phase 2: Reward for getting upright (calves already good)
        upright_phase2 = torch.exp(-((gz + 1.0) ** 2) / (2.0 * eps_upright * eps_upright)) * phase2_active.float()
        
        # Phase 3: Reward for full laying down pose (calves good AND upright)
        target_posture = torch.exp(-posture_err_sq) * phase3_active.float()   
        

        # Height reward removed — final pose driven by joint alignment and stillness
        current_height = self._robot.data.root_state_w[:, 2]
        target_height_low = getattr(self.cfg, "target_height_low", 0.15)

        # Phase 4: final stillness (no motion once pose achieved)
        if not hasattr(self, "_final_hold_counter") or self._final_hold_counter.shape[0] != self.num_envs:
            self._final_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        motion_level = qd.abs().mean(dim=1) + self._imu.data.ang_vel_b.abs().mean(dim=1)
        motion_quiet = motion_level < 0.02
        pose_aligned = phase3_active & (front_calf_err < 0.01) & (rear_calf_err < 0.01) & motion_quiet
        self._final_hold_counter = torch.where(pose_aligned, self._final_hold_counter + 1, torch.zeros_like(self._final_hold_counter))
        hold_final_steps = max(1, int(0.6 / float(self.step_dt)))
        phase4_active = self._final_hold_counter >= hold_final_steps
        self._phase4_mask = phase4_active.detach()

        # Stillness penalty in final pose (reduce shaking)
        final_stillness_penalty = motion_level * phase4_active.float()
        final_stillness_reward = torch.exp(-motion_level / 0.005) * phase4_active.float()

        # Penalty for being upside down (g_z > 0.5)
        upside_down_penalty = torch.clamp(g_b[:, 2] - 0.5, min=0.0)  # Penalty when g_z > 0.5
        
        # Recovery progress reward - encourages getting closer to upright
        recovery_progress = torch.clamp(-g_b[:, 2], min=0.0, max=1.0)  # 0 when upside down, 1 when upright
        
        # Strong orientation reward - heavily penalize any deviation from upright
        orientation_penalty = torch.abs(g_b[:, 2] + 1.0)  # 0 when perfectly upright (g_z = -1), increases with deviation

        if not hasattr(self, "_ever_reached_target") or self._ever_reached_target.shape[0] != self.num_envs:
            self._ever_reached_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # penalize belly/base contact only when near upright
        # belly penalty only during stance phase
        belly_contact_upright = body_contact * stance_phase.float()

        # stance conditions: upright, near laying down joints, low height (belly on ground)
        posture_close = posture_err_sq < 0.5  # More forgiving for "almost perfect" laying down pose
        height_low = current_height < (target_height_low + 0.05)  # Belly close to ground
        stance_ok = posture_close & height_low & stance_phase
        # sustained stance counter to avoid one-frame spikes
        if not hasattr(self, "_stance_counter") or self._stance_counter.shape[0] != self.num_envs:
            self._stance_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        hold_steps = max(1, int(0.4 / float(self.step_dt)))  # ~0.4s hold
        self._stance_counter = torch.where(stance_ok, self._stance_counter + 1, torch.zeros_like(self._stance_counter))
        stance_held = self._stance_counter >= hold_steps
        newly_stanced = stance_held & (~self._ever_reached_target)

        # Refinement phase: after robot has been stably standing for a while
        if not hasattr(self, "_refinement_counter") or self._refinement_counter.shape[0] != self.num_envs:
            self._refinement_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        refinement_delay = max(1, int(2.0 / float(self.step_dt)))  # 2 seconds after stance phase starts
        self._refinement_counter = torch.where(stance_phase, self._refinement_counter + 1, torch.zeros_like(self._refinement_counter))
        refinement_phase = self._refinement_counter >= refinement_delay
        
        # Gentle refinement reward - only when in refinement phase
        refinement_posture = torch.exp(-posture_err_sq * 0.5) * refinement_phase.float()  # Gentle exponential

        # penalize wide leg spread (hip abduction) when near upright
        hip_abduction = q[:, 0:4].abs().mean(dim=1)
        hip_spread_penalty = hip_abduction * stance_phase.float()
        
        # Penalty for feet off the ground - encourage all feet down (when in final pose)
        feet_heights = self._robot.data.body_pos_w[:, self._feet_ids_robot, 2]  # Z positions of all feet
        feet_height_mean = feet_heights.mean(dim=1)  # Average distance from ground
        feet_off_ground_penalty = torch.clamp(feet_height_mean, min=0.0) * stance_phase.float()  # Penalize when feet are above ground

        # Torso/limb contacts (body awareness)
        contact_13 = self._get_true_contact_13()
        hip_contact = contact_13[:, [1, 4, 7, 10]].max(dim=1).values
        thigh_contact = contact_13[:, [2, 5, 8, 11]].max(dim=1).values
        torso_obstruction = (hip_contact + thigh_contact) * (phase1_active.float() + phase2_active.float())

        w = dict(
            base_orient=-0.5,
            upright=15.0,  # Much stronger - this is the main goal
            calf_reward=12.0,  # Strong reward for contracting calves first (Phase 1)
            calf_penalty=-15.0,  # Persistent penalty if calves drift from target
            front_calf_penalty=-25.0,  # Keep front calves tucked tightly
            rear_calf_penalty=-10.0,   # Ensure rear calves stay tucked
            fl_calf_reward=6.0,
            fr_calf_reward=6.0,
            fr_leg_reward=4.0,
            fr_leg_penalty=-18.0,
            final_stillness=-12.0,
            final_stillness_reward=6.0,
            upright_phase2=12.0,  # Strong reward for getting upright after calves are good (Phase 2)
            target_posture=4.0,  # Stronger pull into laying down pose when upright (Phase 3)
            upside_down_penalty=-10.0,  # Very strong penalty for upside down
            recovery_progress=5.0,  # Strong reward for any progress toward upright
            orientation_penalty=-8.0,  # Strong penalty for any deviation from upright
            contact_belly_upright=0.0,  # No penalty - we want belly on ground for laying down
            torso_obstruction=-6.0,  # Penalize hips/thighs colliding before final pose
            stance_bonus=25.0,  # bigger one-shot bonus on sustained stance
            hip_spread_penalty=-10.0,  # Strongly discourage wide leg spread when upright (increased)
            feet_off_ground_penalty=-8.0,  # Strongly discourage feet off ground in stance
            refinement_posture=2.0,  # gentle refinement reward for perfect stance
            feet_contact=0.1,  # Minimal - not important during recovery
            body_contact=-0.05,  # Minimal penalty - some contact needed for flipping
            safety_force=-1.0e-3,  # Minimal - allow aggressive movements
            body_bias=-0.02,  # Minimal - robot needs to move around
            pos_limits=-0.2,  # Minimal - allow joint movement
            ang_vel_limit=-0.02,  # Minimal - allow fast movement
            qdd_sq=-5.0e-7,  # Minimal - allow aggressive accelerations
            qd_sq=-2.0e-3,  # Minimal - allow fast joint movement
            act_smooth=-0.002,  # Minimal - allow dynamic actions
            tau_sq=-1.0e-4,  # Minimal - allow high torques
        )

        rewards = {
            "r_orient_base_gxy":       w["base_orient"]   * gxy_sq * dt,
            "r_orient_upright":        w["upright"]       * upright * dt,
            "r_calf_reward":           w["calf_reward"]   * calf_reward * dt,
            "r_calf_fl_reward":       w["fl_calf_reward"] * fl_calf_reward * dt,
            "r_calf_fr_reward":       w["fr_calf_reward"] * fr_calf_reward * dt,
            "r_fr_leg_reward":        w["fr_leg_reward"] * fr_leg_reward * dt,
            "r_calf_penalty":          w["calf_penalty"]  * calf_penalty * dt,
            "r_front_calf_penalty":    w["front_calf_penalty"] * front_calf_penalty * dt,
            "r_rear_calf_penalty":     w["rear_calf_penalty"] * rear_calf_penalty * dt,
            "r_fr_leg_penalty":        w["fr_leg_penalty"] * fr_leg_penalty * dt,
            "r_final_stillness":       w["final_stillness"] * final_stillness_penalty * dt,
            "r_final_stillness_reward": w["final_stillness_reward"] * final_stillness_reward * dt,
            "r_upright_phase2":        w["upright_phase2"] * upright_phase2 * dt,
            "r_orient_target_posture": w["target_posture"]* target_posture * dt,
            "r_upside_down_penalty":   w["upside_down_penalty"] * upside_down_penalty * dt,
            "r_orientation_penalty":   w["orientation_penalty"] * orientation_penalty * dt,
            "r_contact_belly_upright": w["contact_belly_upright"] * belly_contact_upright * dt,
            "r_stance_bonus":          w["stance_bonus"] * newly_stanced.float(),
            "r_hip_spread_penalty":    w["hip_spread_penalty"] * hip_spread_penalty * dt,
            "r_feet_off_ground_penalty": w["feet_off_ground_penalty"] * feet_off_ground_penalty * dt,
            "r_refinement_posture":    w["refinement_posture"] * refinement_posture * dt,
            "r_recovery_progress":     w["recovery_progress"] * recovery_progress * dt,
            "r_contact_feet":          w["feet_contact"]  * feet_contact * dt,
            "r_contact_body":          w["body_contact"]  * body_contact * dt,
            "r_stability_safety_force":w["safety_force"]  * safety_force * dt,
            "r_stability_body_bias":   w["body_bias"]     * body_bias * dt,
            "r_motion_pos_limits":     w["pos_limits"]    * pos_limit_term * dt,
            "r_motion_ang_vel_limit":  w["ang_vel_limit"] * qd_excess * dt,
            "r_motion_joint_acc":      w["qdd_sq"]        * qdd_sq * dt,
            "r_motion_joint_vel":      w["qd_sq"]         * qd_sq * dt,
            "r_motion_action_smooth":  w["act_smooth"]    * act_smooth * dt,
            "r_motion_torque":         w["tau_sq"]        * tau_sq * dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        if not hasattr(self, "_episode_sums"):
            self._episode_sums = {}
        for k, v in rewards.items():
            if (self._log_metric_keys is None) or (k in self._log_metric_keys):
                buf = self._episode_sums.get(k)
            if buf is None or buf.shape[0] != self.num_envs or buf.device != self.device:
                self._episode_sums[k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[k] += v
        if not hasattr(self, "_episode_return") or self._episode_return.shape[0] != self.num_envs:
            self._episode_return = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_return += reward

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple time-based termination only."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if env_ids is not None and len(env_ids) == self.num_envs:
            # Spread resets
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Clear action memory
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Spawn state (pose + joints)
        use_good_pose = bool(getattr(self.cfg, "use_good_pose_reset", False))
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        if use_good_pose:
            root, jp, jv = self.quad_spawner.spawn_layingdown(env_ids)
        else:
            root, jp, jv = self.quad_spawner.spawn_bad_pose(env_ids)

        default_root_state[:, :7] = root[:, :7]
        default_root_state[:, 7:] = 0.0

        # Write to sim
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(jp, jv, None, env_ids)

        # Reset MCP caches (only when MCP enabled)
        if self.cfg.use_mcp:
            self._o_hist[env_ids] = 0.0

        # Reset reference point for body-bias term (used in rewards)
        if hasattr(self, "_init_root_xy"):
            self._init_root_xy[env_ids] = default_root_state[:, :2]

        # Reset target tracking flags
        if hasattr(self, "_ever_reached_target"):
            self._ever_reached_target[env_ids] = False
        if hasattr(self, "_final_hold_counter"):
            self._final_hold_counter[env_ids] = 0
        if hasattr(self, "_phase4_mask"):
            self._phase4_mask[env_ids] = False

        # Logging (filtered)
        extras = dict()
        keys_to_log = [k for k in self._episode_sums.keys() if (self._log_metric_keys is None) or (k in self._log_metric_keys)]
        for key in keys_to_log:
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        if hasattr(self, "_episode_return"):
            episodic_return_avg = torch.mean(self._episode_return[env_ids])
            extras["Episode/Return"] = episodic_return_avg / self.max_episode_length_s
            self._episode_return[env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        if (self._terrain.cfg.terrain_generator is not None 
            and self._terrain.cfg.terrain_generator.curriculum is True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())

        self.extras["log"].update(extras)


    def _get_mcp_features(self, clock_data=None):
        """
        Online MCP (mass/contact predictor) like your concurrent state estimator:
        - Uses history of proprio (H * o_dim)
        - Predicts (m_hat, c_hat, z_hat)
        - Adds supervised sample to dataset with privileged labels
        - Trains periodically inside this call (grad-safe)
        Returns:
            m_hat: (N, 4)    mass-group distribution
            c_hat: (N, 13)   contact indicator vector
            z_hat: (N, zdim) latent
        """
        if not getattr(self.cfg, "use_mcp", False):
            N = self.num_envs
            zdim = int(getattr(self.cfg, "mcp_latent_dim", 16))
            return (torch.zeros(N, 4, device=self.device),
                    torch.zeros(N, 13, device=self.device),
                    torch.zeros(N, zdim, device=self.device))

        # -------------- Hyperparams (cfg with sensible defaults) --------------
        H        = int(getattr(self.cfg, "mcp_history", 5))
        o_dim    = int(getattr(self.cfg, "single_o_dim", 58))
        z_dim    = int(getattr(self.cfg, "mcp_latent_dim", 16))
        # training cadence in *steps* (not episodes)
        train_every_steps = int(getattr(self.cfg, "mcp_train_every_steps", 240))  # ~10 Hz if dt=0.04
        min_buffer        = int(getattr(self.cfg, "mcp_min_buffer", 4096))
        batch_size        = int(getattr(self.cfg, "mcp_batch_size", 2048))
        lr                = float(getattr(self.cfg, "mcp_lr", 1e-3))
        warmup_steps      = int(getattr(self.cfg, "mcp_warmup_steps", 1000))
        # loss weights
        lam_m   = float(getattr(self.cfg, "mcp_lambda_m",   1.0))
        lam_c   = float(getattr(self.cfg, "mcp_lambda_c",   1.0))
        lam_rec = float(getattr(self.cfg, "mcp_lambda_rec", 1.0))
        lam_kl  = float(getattr(self.cfg, "mcp_lambda_kl",  1e-3))

        # -------------- Lazy init --------------
        if not hasattr(self, "_mcp"):
            from basic_locomotion_dls_isaaclab.tasks.fall_recovery.modules.mcp import MCP, MCPDataset, MCPDims
            dims = MCPDims(o_dim=o_dim, m_dim=4, c_dim=13, z_dim=z_dim, H=H)
            self._mcp = MCP(dims).to(self.device)
            self._mcp.dataset = MCPDataset(max_size=int(getattr(self.cfg, "mcp_buffer_size", 200_000)),
                                        device=self.device)
            self._mcp_opt = torch.optim.Adam(self._mcp.parameters(), lr=lr)

        if not hasattr(self, "_o_hist"):
            self._o_hist = torch.zeros(self.num_envs, H, o_dim, device=self.device)
        if not hasattr(self, "_last_o_hist_flat"):
            self._last_o_hist_flat = torch.zeros(self.num_envs, H * o_dim, device=self.device)
        if not hasattr(self, "_last_m_true"):
            self._last_m_true = torch.zeros(self.num_envs, 4, device=self.device)
        if not hasattr(self, "_last_c_true"):
            self._last_c_true = torch.zeros(self.num_envs, 13, device=self.device)

        # -------------- Build current proprio and update history --------------
        # Use your existing builder (same vector you pass to policy’s o_t part)
        o_t = self._build_o_t()  # (N, o_dim)

        # shift left, append newest
        self._o_hist = torch.cat([self._o_hist[:, 1:, :], o_t.unsqueeze(1)], dim=1)  # (N,H,o)
        o_hist_flat  = self._o_hist.reshape(self.num_envs, -1)                        # (N,H*o)

        # -------------- Dataset add: (o_hist_{t-1} -> o_t) with privileged labels --------------
        if self.common_step_counter > 0 and self._mcp.dataset is not None:
            self._mcp.dataset.add(
                o_hist=self._last_o_hist_flat,
                o_next=o_t,
                m_true=self._last_m_true,
                c_true=self._last_c_true,
            )

        # cache labels for next step
        self._last_o_hist_flat = o_hist_flat.detach()
        self._last_m_true = self._get_true_mass_distribution().detach()  # (N,4)
        self._last_c_true = self._get_true_contact_13().detach()         # (N,13)

        # -------------- Prediction (no grad during rollout) --------------
        if self.common_step_counter >= warmup_steps:
            with torch.no_grad():
                m_hat, c_hat, z_hat, _ = self._mcp(o_hist_flat)  # (N,4),(N,13),(N,z)
        else:
            N = self.num_envs
            m_hat = torch.zeros(N, 4, device=self.device)
            c_hat = torch.zeros(N, 13, device=self.device)
            z_hat = torch.zeros(N, z_dim, device=self.device)

        # -------------- Periodic training (inside this function) --------------
        ds = self._mcp.dataset
        train_enabled = bool(getattr(self.cfg, "mcp_train_enabled", True))
        if (train_enabled
            and ds is not None
            and isinstance(len(ds), int)  # quiet type checkers
            and len(ds) >= min_buffer
            and (int(self.common_step_counter) % train_every_steps == 0)):
            # Escape any outer inference/no_grad so autograd works
            import torch as _torch
            with _torch.inference_mode(False):
                with _torch.enable_grad():
                    batch = self._mcp.dataset.sample(batch_size)
                    weights = {"lambda_m": lam_m, "lambda_c": lam_c, "lambda_rec": lam_rec, "lambda_kl": lam_kl}
                    logs = self._mcp.train_step(batch, weights, self._mcp_opt)
                    # lightweight logging
                    self.extras.setdefault("log", {}).update({f"MCP/{k}": v.item() for k, v in logs.items()})
                    # Save best model by total loss
                    if train_enabled:
                        try:
                            curr_loss = float(logs.get("loss", 1e9))
                            best = getattr(self, "_mcp_best_loss", None)
                            if (best is None) or (curr_loss < best):
                                setattr(self, "_mcp_best_loss", curr_loss)
                                save_root_best = getattr(self.cfg, "log_dir", None)
                                best_dir = os.path.join(save_root_best, "mcp") if isinstance(save_root_best, str) and len(save_root_best) > 0 else "logs/mcp"
                                os.makedirs(best_dir, exist_ok=True)
                                best_path = os.path.join(best_dir, "mcp_best.pth")
                                torch.save({
                                    "state_dict": self._mcp.state_dict(),
                                    "opt_state": self._mcp_opt.state_dict(),
                                    "best_loss": curr_loss,
                                    "dims": {"H": H, "o_dim": o_dim, "z_dim": z_dim}
                                }, best_path)
                                print(f"[INFO] MCP: saving best checkpoint: {best_path} (loss={curr_loss:.5f})")
                                self.extras.setdefault("log", {})["MCP/best_ckpt"] = best_path
                        except Exception:
                            pass


        # -------------- Optional: periodic checkpoint --------------
        save_every = int(getattr(self.cfg, "mcp_save_every_steps", 10000))
        # Prefer task log_dir if provided (e.g., set by runner), else fallback
        save_root = getattr(self.cfg, "log_dir", None)
        default_dir = "logs/mcp"
        save_dir   = os.path.join(save_root, "mcp") if isinstance(save_root, str) and len(save_root) > 0 else default_dir
        if train_enabled and save_every > 0 and (int(self.common_step_counter) % save_every == 0):
            print("saving MCP model at step", int(self.common_step_counter))
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"mcp_step_{int(self.common_step_counter):08d}.pth")
            torch.save({"state_dict": self._mcp.state_dict(),
                        "opt_state": self._mcp_opt.state_dict(),
                        "dims": {"H": H, "o_dim": o_dim, "z_dim": z_dim}}, ckpt_path)
            # lightweight breadcrumb in extras
            self.extras.setdefault("log", {})["MCP/last_ckpt"] = ckpt_path
        # Throttled MCP stats to log structure (no prints per step)
        if int(self.common_step_counter) % (10 * train_every_steps) == 0:
            self.extras.setdefault("log", {}).update({
                "MCP/m_mean": m_hat.mean().item(),
                "MCP/c_mean": c_hat.mean().item(),
                "MCP/z_mean": z_hat.mean().item(),
                "MCP/buffer_size": len(self._mcp.dataset) if self._mcp.dataset is not None else 0,
            })
        return m_hat, c_hat, z_hat


    def _build_o_t(self):
        ang_vel_b = self._imu.data.ang_vel_b
        proj_g_b = self._imu.data.projected_gravity_b
        q = self._robot.data.joint_pos
        qd = self._robot.data.joint_vel
        a_prev = self._previous_actions

        root_state = self._robot.data.root_state_w
        base_height = root_state[:, 2:3]
        base_roll, base_pitch, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        base_roll = base_roll.unsqueeze(-1)
        base_pitch = base_pitch.unsqueeze(-1)

        contact_13 = self._get_true_contact_13()

        return torch.cat([ang_vel_b, proj_g_b, q, qd, a_prev,
                          base_height, base_roll, base_pitch, contact_13], dim=-1)

    def _get_privileged_observation(self):
        o_t = self._build_o_t()
        pcom = self._get_com_xy()
        c_t = self._get_true_contact_13()
        c_f = self._get_foot_contact_forces_4()
        return torch.cat([o_t, pcom, c_t, c_f], dim=-1)


    def _get_com_xy(self):
        try:
            com_w = self._robot.data.com_pos_w
        except:
            com_w = self._robot.data.root_state_w[:, :3]
        return com_w[:, :2]

    def _get_foot_contact_forces_4(self):
        feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        F = self._contact_sensor.data.net_forces_w[:, feet_ids, 2]
        return torch.abs(F)
    
    def _get_true_contact_13(self):
        netF = self._contact_sensor.data.net_forces_w
        force_mag = torch.linalg.norm(netF, dim=-1)
        contact = (force_mag > 1.0).float()

        names = self._robot.body_names
        def mask(substr): return torch.tensor([(substr in n) for n in names], device=self.device)
        masks = [
            mask("base"),
            mask("hip") & mask("LF"), mask("thigh") & mask("LF"), mask("calf") & mask("LF"),
            mask("hip") & mask("RF"), mask("thigh") & mask("RF"), mask("calf") & mask("RF"),
            mask("hip") & mask("LH"), mask("thigh") & mask("LH"), mask("calf") & mask("LH"),
            mask("hip") & mask("RH"), mask("thigh") & mask("RH"), mask("calf") & mask("RH"),
        ]
        outs = [(contact * m).max(dim=1, keepdim=True)[0] for m in masks]
        return torch.cat(outs, dim=-1)
    
    def _init_mass_group_indices(self):
        """Cache body index tensors (on self.device) for [base, hip, thigh, calf]."""
        if hasattr(self, "_mass_group_idx"):
            return
        # Flexible patterns to cover different robots
        base_ids,  _ = self._robot.find_bodies("base|trunk|chassis|torso")
        hip_ids,   _ = self._robot.find_bodies("(?i).*hip")
        thigh_ids, _ = self._robot.find_bodies("(?i).*thigh")
        calf_ids,  _ = self._robot.find_bodies("(?i).*(calf|knee|shank)")

        def to_idx(ids):
            return (torch.tensor(ids, dtype=torch.long, device=self.device)
                    if len(ids) > 0 else torch.empty(0, dtype=torch.long, device=self.device))

        self._mass_group_idx = {
            "base":  to_idx(base_ids),
            "hip":   to_idx(hip_ids),
            "thigh": to_idx(thigh_ids),
            "calf":  to_idx(calf_ids),
        }

    def _get_true_mass_distribution(self):
        """
        Returns (N, 4): total mass grouped as [base, hip*, thigh*, calf*].
        - Uses PhysX view to get per-body masses.
        - Device-safe (moves masses to self.device).
        - Works whether get_masses() gives (B,) or (N, B).
        """
        self._init_mass_group_indices()

        # 1) Fetch masses from PhysX and move to the same device as the rest of the sim tensors
        try:
            masses = self._robot.root_physx_view.get_masses()  # usually CPU
        except AttributeError:
            masses = self._robot._articulation_view.get_masses()

        masses = torch.as_tensor(masses, dtype=torch.float32, device=self.device)

        # 2) Ensure shape is (N, B): replicate across envs if needed
        if masses.dim() == 1:  # (B,)
            masses = masses.unsqueeze(0).expand(self.num_envs, -1)  # (N, B)
        elif masses.shape[0] != self.num_envs and masses.shape[1] == self.num_envs:
            # (B, N) -> (N, B) if some builds flip axes
            masses = masses.transpose(0, 1).contiguous()

        # 3) Safe group-sum helper (index tensor already on self.device)
        def sum_group(idx: torch.Tensor):
            if idx.numel() == 0:
                return torch.zeros(self.num_envs, 1, device=self.device)
            return masses.index_select(1, idx).sum(dim=1, keepdim=True)

        base  = sum_group(self._mass_group_idx["base"])
        hip   = sum_group(self._mass_group_idx["hip"])
        thigh = sum_group(self._mass_group_idx["thigh"])
        calf  = sum_group(self._mass_group_idx["calf"])

        return torch.cat([base, hip, thigh, calf], dim=-1)
