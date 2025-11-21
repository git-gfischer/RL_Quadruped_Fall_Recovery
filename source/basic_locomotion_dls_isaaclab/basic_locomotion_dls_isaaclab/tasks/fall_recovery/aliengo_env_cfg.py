import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from basic_locomotion_dls_isaaclab.assets.aliengo_asset import ALIENGO_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import basic_locomotion_dls_isaaclab.tasks.custom_events as custom_events
import basic_locomotion_dls_isaaclab.tasks.custom_curriculums as custom_curriculums

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )

    
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    

    scale_all_joint_friction_model = EventTerm(
        func=custom_events.randomize_joint_friction_model,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
                "friction_distribution_params": (0.2, 2.0),
                "operation": "scale"},
    )


    scale_all_joint_armature_model = EventTerm(
        func=custom_events.randomize_joint_friction_model,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
                "armature_distribution_params": (0.0, 1.0),
                "operation": "scale"},
    )
    


    actuator_gains = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (-5.0, 5.0),
        "damping_distribution_params": (-1.0, 1.0),
        "operation": "add",
        "distribution": "uniform",
    },
    )
    
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                                   "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
    )

    # zero command velocity
    """zero_command_velocity = EventTerm(
        func=custom_events.zero_command_velocity,
        mode="interval",
        interval_range_s=(19.0, 19.0),
    )"""

    """# reset command velocity
    resample_command_velocity = EventTerm(
        func=custom_events.resample_command_velocity,
        mode="interval",
        interval_range_s=(11.0, 11.0),
    )"""




@configclass
class AliengoFlatEnvCfg(DirectRLEnvCfg):

    # Viewer
    #viewer = ViewerCfg(eye=(1.5, 1.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 2.0  # Increased from 1.0 for more powerful actions
    action_space = 12
    observation_space = 42
    state_space = 0
    
    # Height reward parameters
    target_height = 0.35  # Desired robot height for height reward
    target_height_low = 0.15  # Desired low height for laying down (belly close to ground)

    # MCP parameters
    use_mcp = False
    mcp_history = 5          # H = 5
    single_o_dim = 58        # proprio + body-awareness features (ang vel, gravity, q, qd, a_prev, height, roll, pitch, contacts)
    mcp_latent_dim = 16
    mcp_lambda_m = 1.0
    mcp_lambda_c = 1.0
    mcp_lambda_rec = 1.0
    mcp_lambda_kl = 1e-3
    mcp_batch_size = 2048
    mcp_train_every_ep = 4   # train interval (episodes-equivalent steps)
    mcp_lr = 1e-3

    # critic height scan size (paper uses 187 points)
    critic_height_dim = 187
    
    

    use_clock_signal = False
    if(use_clock_signal):
        observation_space += 4

    # observation history
    use_observation_history = False
    history_length = 5
    if(use_observation_history):
        single_observation_space = observation_space # Placeholder. Later we may add map, but only from the latest obs
        observation_space *= history_length

    #use_imu = False

    use_cuncurrent_state_est = False
    if(use_cuncurrent_state_est):
        cuncurrent_state_est_output_space = 3 #lin_vel_b
        single_cuncurrent_state_est_observation_space = single_observation_space
        cuncurrent_state_est_observation_space = observation_space
        cuncurrent_state_est_batch_size = 32
        cuncurrent_state_est_train_epochs = 500
        cuncurrent_state_est_lr = 1e-3
        cuncurrent_state_est_ep_saving_interval = 1000


    use_filter_actions = False

    # asymmetric ppo
    use_asymmetric_ppo = True
    # if(use_asymmetric_ppo):
    #     state_space = observation_space
    #     state_space += 12 # P gain
    #     state_space += 12 # D gain
    #     #state_space += 1*17 # mass*num_bodies
    #     #state_space += 1*17 # inertia*num_bodies
    #     #state_space += 1 # wrench
    #     state_space += 12 # friction static
    #     state_space += 12 # friction dynamic
    #     state_space += 12 # armature
    #     #state_space += 1 # restitution

    # use_amp = False


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        #physx=PhysxCfg(
        #    gpu_max_rigid_contact_count=2**20,
        #    gpu_max_rigid_patch_count=2**24,
        #),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        #attach_yaw_only=True,
        ray_alignment='yaw',
        #pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.4, 1.0]),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # an imu sensor in case we don't want any state estimator
    imu = ImuCfg(prim_path="/World/envs/env_.*/Robot/base", debug_vis=True)


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()


    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # robot
    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Final Stand position (joint positions)
    # For aliengo robot
    #[hip_FL, hip_FR, hip_RL, hip RR, thigh_FL, thigh_FR, thigh_RR, thigh RL, calf_FL, calf_FR, calf_RR, calf RL]
    stand_joint_pos =  [ 0.00,  0.00,  0.00,  0.00,
                         0.90,  0.90,  0.90,  0.90,
                        -1.80, -1.80, -1.80, -1.80 ]  

    laying_joint_pos =  [0.4, -0.4, 0.4, -0.4, 
                         2.00, 2.00, 2.00, 2.00, 
                         -2.50, -2.50, -2.50, -2.50]
    
    use_good_pose_reset = False   # Use lying-down poses
                                 # If False: Use bad/tumbled poses (default)
    
    # Debugging: No action during recovery
    no_action = False
   

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
@configclass
class AliengoRoughBlindEnvCfg(AliengoFlatEnvCfg):

    ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        curriculum=False,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.2
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.10), platform_width=2.0,
            ),
            "star": terrain_gen.MeshStarTerrainCfg(
                proportion=0.1, num_bars=10, bar_width_range=(0.15, 0.20), bar_height_range=(0.05, 0.15), platform_width=2.0,
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.05, 0.18), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
        },
    )

    """Rough terrains configuration."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=10,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )





@configclass
class AliengoRoughVisionEnvCfg(AliengoRoughBlindEnvCfg):
    # env
    #observation_space = 276
    observation_space = 429

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        #ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 1.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )