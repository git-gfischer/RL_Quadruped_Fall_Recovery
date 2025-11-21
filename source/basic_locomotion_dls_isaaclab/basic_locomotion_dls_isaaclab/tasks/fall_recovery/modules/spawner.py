from __future__ import annotations

import torch
from typing import Tuple

from isaaclab.utils import math as math_utils
from dataclasses import dataclass


@dataclass
class StandingParams:
    height: float = 0.35
    yaw_range: float = torch.pi
    joint_noise: float = 0.05
    pos_jitter: float = 0.0 


@dataclass
class LayingDownParams:
    belly_height: float = 0.15
    roll_noise: float = 0.0
    pitch_noise: float = 0.0
    yaw_range: float = torch.pi
    pos_jitter: float = 0.0
    joint_noise: float = 0.05
    # Laying down joint configuration (tucked legs)
    haa_angles: tuple = (0.0, 0.0, 0.0, 0.0)  # HAA: slight outward splay
    hfe_angles: tuple = (0.0, 0.0, 0.0, 0.0)    # HFE: thighs folded under body
    kfe_angles: tuple = (0.0, 0.0, 0.0, 0.0)    # KFE: shanks tucked in


@dataclass
class UpsideDownParams:
    height: float = 0.25
    roll_noise: float = 0.15
    pitch_noise: float = 0.15
    yaw_range: float = torch.pi
    pos_jitter: float = 0.0
    joint_noise: float = 0.05


@dataclass
class BadPoseParams:
    height: float = 0.15
    roll_noise: float = 0.15
    pitch_noise: float = 0.15
    yaw_range: float = torch.pi
    pos_jitter: float = 0.0
    joint_noise: float = 0.05


class QuadrupedSpawner:
    """
    Utility to generate spawn poses for a quadruped in batched vectorized envs.

    Returns tensors suitable for write_*_to_sim calls:
      - root_state: (n, 13) [pos(3), quat(x,y,z,w)(4), lin vel(3), ang vel(3)]
      - joint_pos:  (n, J)
      - joint_vel:  (n, J)
    """

    def __init__(
        self,
        device: torch.device,
        env_origins: torch.Tensor,           # (N, 3)
        default_root_state: torch.Tensor,    # (N, 13)
        default_joint_pos: torch.Tensor,     # (N, J)
        default_joint_vel: torch.Tensor,     # (N, J)
        standing: StandingParams | None = None,
        laying: LayingDownParams | None = None,
        upside: UpsideDownParams | None = None,
        bad_pose: BadPoseParams | None = None,
    ) -> None:
        self.device = device
        self.env_origins = env_origins
        self.default_root_state = default_root_state
        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        
        # Pose parameters
        self.standing = standing or StandingParams()
        self.laying = laying or LayingDownParams()
        self.upside = upside or UpsideDownParams()
        self.bad_pose = bad_pose or BadPoseParams()
    # ---------------------------- helpers ----------------------------
    def _base_from_origin(self, env_ids: torch.Tensor, height: float, pos_jitter: float = 0.0) -> torch.Tensor:
        n = len(env_ids)
        root = self.default_root_state[env_ids].clone()
        root[:, :3] = self.env_origins[env_ids]
        pj = pos_jitter
        if pj and pj > 0.0:
            root[:, 0] += pj * (torch.rand(n, device=self.device) - 0.5)
            root[:, 1] += pj * (torch.rand(n, device=self.device) - 0.5)
        root[:, 2] = height
        root[:, 7:] = 0.0  # zero velocities
        return root
    #--------------------------------------------------------
    def _defaults(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.default_root_state[env_ids].clone(), 
            self.default_joint_pos[env_ids].clone(),
            torch.zeros_like(self.default_joint_vel[env_ids]),
        )
    # ---------------------------- public APIs ----------------------------
    def spawn_standing(self, env_ids: torch.Tensor, params: StandingParams | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root, jp, jv = self._defaults(env_ids)
        n = len(env_ids)
        # place at origin height with small yaw
        root[:, :3] = self.env_origins[env_ids]
        p = params or self.standing
        root[:, 2] = p.height
        yr = p.yaw_range
        yaw = 2 * yr * torch.rand(n, device=self.device) - yr
        quat = math_utils.quat_from_euler_xyz(torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), yaw)
        root[:, 3:7] = quat
        jp += p.joint_noise * torch.randn_like(jp)
        return root, jp, jv
    # --------------------------------------------------------
    def spawn_layingdown(self, env_ids: torch.Tensor, params: LayingDownParams | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(env_ids)
        p = params or self.laying
        root = self._base_from_origin(env_ids, p.belly_height, p.pos_jitter)
        # belly down: pitch ~ 0 (belly touching floor), configurable noise
        roll = p.roll_noise * torch.randn(n, device=self.device)
        pitch = p.pitch_noise * torch.randn(n, device=self.device)  # around 0, not pi
        yaw = 2 * p.yaw_range * torch.rand(n, device=self.device) - p.yaw_range
        root[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        # joints: proper laying down configuration (tucked legs)
        jp = self.default_joint_pos[env_ids].clone()
        # Laying down joint configuration from parameters
        jp[:, 0:4] = torch.tensor(p.haa_angles, device=self.device)  # HAA: slight outward splay
        jp[:, 4:8] = torch.tensor(p.hfe_angles, device=self.device)  # HFE: thighs folded under body
        jp[:, 8:12] = torch.tensor(p.kfe_angles, device=self.device)  # KFE: shanks tucked in
        jv = torch.zeros_like(self.default_joint_vel[env_ids])
        jp += p.joint_noise * torch.randn_like(jp)
        return root, jp, jv

    def spawn_upsidedown(self, env_ids: torch.Tensor, params: UpsideDownParams | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(env_ids)
        p = params or self.upside
        root = self._base_from_origin(env_ids, p.height, p.pos_jitter)
        roll = torch.pi + p.roll_noise * torch.randn(n, device=self.device)
        pitch = p.pitch_noise * torch.randn(n, device=self.device)
        yaw = 2 * p.yaw_range * torch.rand(n, device=self.device) - p.yaw_range
        root[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        jp = self.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.default_joint_vel[env_ids])
        jp += p.joint_noise * torch.randn_like(jp)
        return root, jp, jv

    def spawn_bad_pose(self, env_ids: torch.Tensor, params: BadPoseParams | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        n = len(env_ids)
        p = params or self.bad_pose
        root = self._base_from_origin(env_ids, p.height, p.pos_jitter)
        jp = self.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.default_joint_vel[env_ids])

        probs = torch.rand(n, device=self.device)
        roll  = torch.zeros(n, device=self.device)
        pitch = torch.zeros(n, device=self.device)
        yaw   = 2 * torch.pi * torch.rand(n, device=self.device) - torch.pi  # free yaw

        # 40% upside-down around pi roll (belly up)
        m1 = probs < 0.40
        if m1.any():
            roll[m1]  = torch.pi + 0.30 * torch.randn(m1.sum(), device=self.device).clamp(-0.5, 0.5)
            pitch[m1] = 0.25 * torch.randn(m1.sum(), device=self.device).clamp(-0.5, 0.5)

        # 25% side-lying around ±pi/2 roll (arms/legs splayed)
        m2 = (~m1) & (probs < 0.65)
        if m2.any():
            sign = torch.sign(torch.rand(m2.sum(), device=self.device) - 0.5)
            roll[m2]  = (torch.pi / 2) * sign + 0.25 * torch.randn(m2.sum(), device=self.device).clamp(-0.5, 0.5)
            pitch[m2] = 0.25 * torch.randn(m2.sum(), device=self.device).clamp(-0.5, 0.5)

        # 20% nose-diving / back-diving (large pitch)
        m3 = (~(m1 | m2)) & (probs < 0.85)
        if m3.any():
            pitch_sign = torch.sign(torch.rand(m3.sum(), device=self.device) - 0.5)
            pitch[m3] = pitch_sign * (0.75 * torch.pi) + 0.25 * torch.randn(m3.sum(), device=self.device).clamp(-0.5, 0.5)
            roll[m3]  = 0.3 * torch.randn(m3.sum(), device=self.device)

        # 15% fully random tumble with high angular offsets
        m4 = ~(m1 | m2 | m3)
        if m4.any():
            roll[m4]  = (torch.rand(m4.sum(), device=self.device) * 2 - 1) * torch.pi
            pitch[m4] = (torch.rand(m4.sum(), device=self.device) * 2 - 1) * torch.pi

        bad_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)  # (n,4) (x,y,z,w)
        root[:, 3:7] = bad_quat

        # --- Sample "bad" joint poses: sprawled, tucked, or twisted ---
        mode = torch.randint(0, 3, (n,), device=self.device)

        # Sprawl: spread hips out, legs extended
        sprawl = mode == 0
        if sprawl.any():
            jp[sprawl, 0:4]  += 0.7 * torch.tensor([ 1, -1,  1, -1], device=self.device)
            jp[sprawl, 4:8]  += 0.35
            jp[sprawl, 8:12] += 0.25

        # Curl: tuck under body deeply
        curl = mode == 1
        if curl.any():
            jp[curl, 4:8]  += -0.8
            jp[curl, 8:12] +=  1.0
            jp[curl, 0:4]  += 0.3 * torch.tensor([-1, 1, -1, 1], device=self.device)

        # Twisted: mix front/back leg configurations
        twist = mode == 2
        if twist.any():
            jp[twist, 0:2] += 0.6 * torch.tensor([1, -1], device=self.device)
            jp[twist, 2:4] += -0.6 * torch.tensor([1, -1], device=self.device)
            jp[twist, 4:6] += -0.5
            jp[twist, 6:8] += 0.5
            jp[twist, 8:10] += 0.8
            jp[twist, 10:12] += -0.3

        # Add stronger joint noise and random joint velocities
        jp += (p.joint_noise * 1.5) * torch.randn_like(jp)
        jv = 1.5 * torch.randn_like(jv)

        # Occasionally drop from a higher height to create chaos
        high_drop_mask = torch.rand(n, device=self.device) < 0.25
        if high_drop_mask.any():
            root[high_drop_mask, 2] += torch.rand(high_drop_mask.sum(), device=self.device) * 0.3  # up to +0.3m

        return root, jp, jv
    # --------------------------------------------------------
    def spawn_custom(
        self,
        env_ids: torch.Tensor,
        pos: torch.Tensor | None = None,
        euler_rpy: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        height: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root, jp, jv = self._defaults(env_ids)
        if height is not None:
            root[:, 2] = height
        if pos is not None:
            root[:, :3] = pos.to(device=self.device, dtype=root.dtype)
        if euler_rpy is not None:
            r, p, y = euler_rpy[:, 0], euler_rpy[:, 1], euler_rpy[:, 2]
            root[:, 3:7] = math_utils.quat_from_euler_xyz(r.to(self.device), p.to(self.device), y.to(self.device))
        if joint_pos is not None:
            jp = joint_pos.to(device=self.device, dtype=jp.dtype)
        return root, jp, jv