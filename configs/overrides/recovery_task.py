# Ege
import glob
import numpy as np
from typing import Tuple
from dataclasses import dataclass, field
from configs.definitions import (EnvConfig, ObservationConfig, AlgorithmConfig, RunnerConfig, DomainRandConfig,
                                 NoiseConfig, ControlConfig, InitStateConfig, TerrainConfig,
                                 RewardsConfig, AssetConfig, CommandsConfig, TaskConfig, TrainConfig)
from configs.overrides.terrain import TrimeshTerrainConfig

#########
# Task
#########

@dataclass
class RecoveryEnvConfig(EnvConfig):
    episode_length_s: float = 10.

    #Ege
    num_inactive_steps: int = 30 # number of steps in the beginning of each episode where the robot is kept inactive
    test_envs_per_tile: int = 10 # number of robots to be placed to each tile in test mode
    test_num_episodes: int = 30

# Ege
@dataclass
class RecoveryObservationConfig(ObservationConfig):
    base_vel_in_obs: bool = True  # whether the base (linear/angular) velocity should be in policy observation

@dataclass 
class RecoveryDomainRandConfig(DomainRandConfig):
        # Ege - adding randomized limb angles and base orientation
        randomize_base_orientation: bool = True
        randomize_dof_orientation: bool = True

# Ege Class
@dataclass
class GoalStateConfig:
    min_height_to_goal_ratio: float = 0.7
    max_height_to_goal_ratio: float = 1.2
    max_speed: float = 0.25
    max_angular_vel: float = 0.2
    max_z_deviation: float = 0.9
    max_xy_distance: float = 2
    required_goal_time_s: float = 1
    reset_on_goal: bool = True

# Ege - created class for rules on how the curriculum is structured
@dataclass
class CurriculumConfig:
    success_rate_mean_window: int = 10
    promotion_success_rate: float = 0.9
    demotion_success_rate: float = 0.6

@dataclass
class RecoveryTerrainConfig(TrimeshTerrainConfig):
    terrain_type: str = "semivalley"
    static_friction: float = 1.2 # Ege - used to be 1.0
    dynamic_friction: float = 1.2 # Ege - used to be 1.0
    restitution: float = 0.
    slope_threshold: float = 0.99 # slopes above this threshold will be corrected to vertical surfaces # Ege - was 0.75
    max_init_terrain_level: int = 0
    # Ege
    equal_distribution: dict = field(default_factory=lambda: dict(
        enabled=True,
        row_range=(0,0), #end-inclusive
        col_range=(0,9)  #end-inclusive
    ))
    record_roughness: bool = False
    record_heightmaps: bool = True

@dataclass
class RecoveryRewardsConfig(RewardsConfig):
    only_positive_rewards: bool = False 
    base_height_target: float = 0.25
    soft_dof_pos_limit: float = 0.9
    #dof_pos_limits: float = -10.0 # Ege - adding penalty for out-of-limit joint positions

    @dataclass
    class RecoveryRewardsScalesConfig(RewardsConfig.RewardScalesConfig):
        z_axis_orientation: float = 2.5
        torques: float = 0.001 # Ege - used to be -0.00001
        base_height: float = 40.0
        collision: float = 1
        action_change: float = 0.01
        xy_drift: float = 0.1 # Ege - adding penalty for drifting from start position
        feet_contact: float = 8.0
        lin_vel_z: float = 0.1
        ang_vel_xy: float = 0.005
    scales: RecoveryRewardsScalesConfig = RecoveryRewardsScalesConfig()

@dataclass
class RecoveryTaskConfig(TaskConfig):
    _target_: str = "legged_gym.envs.a1_recovery.A1Recovery"
    env: RecoveryEnvConfig = RecoveryEnvConfig()
    observation: RecoveryObservationConfig = RecoveryObservationConfig()
    terrain: RecoveryTerrainConfig = RecoveryTerrainConfig()
    commands: CommandsConfig = CommandsConfig(
        heading_command=False,
        ranges=CommandsConfig.CommandRangesConfig(
            lin_vel_x=(0., 0.),
            lin_vel_y=(0., 0.),
            ang_vel_yaw=(0., 0.),
            heading=(0., 0.)
        )
    )
    init_state: InitStateConfig = InitStateConfig(
        pos = (0.0, 0.0, 0.5), # x,y,z [m]
        rot = (0.0, 1.0, 0.0, 0.0), # x,y,z,w [quat] 
        lin_vel = (0.0, 0.0, 0.0), # x,y,z [m/s]
        ang_vel = (0.0, 0.0, 0.0), # x,y,z [rad/s]
        lin_vel_noise = 0.0,
        ang_vel_noise = 0.0,
        # NOTE: theres a difference in default joint_angles from a1_config.py in old codebase to INIT_JOINT_ANGLES in the current definitions.py
        # hips -> all 0 instead of spread at -0.1, 0.1, thighs -> all 0.9 instead of spread at 0.8, 1., calves -> all -1.8 instead of -1.5
    )
    control: ControlConfig = ControlConfig(
        stiffness=dict(joint=20.),
        damping=dict(joint=0.5),
        decimation=4,
        # action_scale = 0.25 now instead of 0.5
    )
    asset: AssetConfig = AssetConfig(
        terminate_after_contacts_on=(), # this is important for recovery, we are almost always in contact!
        # TODO: possibly move terminate and penalize after contact to reward config
        self_collisions=True    
    )
    domain_rand: RecoveryDomainRandConfig = RecoveryDomainRandConfig(
        friction_range=(0.25, 1.25),
        randomize_base_mass=False,
        randomize_gains=True
    )
    rewards: RecoveryRewardsConfig = RecoveryRewardsConfig()
    noise: NoiseConfig = NoiseConfig(
        noise_scales=NoiseConfig.NoiseScalesConfig(
            dof_pos=0.03, #prev 0.01
            ang_vel=0.3, #prev 0.25
        )
    )

    curriculum: CurriculumConfig = CurriculumConfig()
    goal_state: GoalStateConfig = GoalStateConfig()

#########
# Train
#########

@dataclass
class RecoveryAlgorithmConfig(AlgorithmConfig):
    _target_: str = "rsl_rl.algorithms.PPO"

@dataclass
class RecoveryRunnerConfig(RunnerConfig):
    checkpoint: int = 0

@dataclass
class RecoveryTrainConfig(TrainConfig):
    _target_: str = "rsl_rl.runners.OnPolicyRunner"
    algorithm: RecoveryAlgorithmConfig = RecoveryAlgorithmConfig()
    runner: RecoveryRunnerConfig = RecoveryRunnerConfig(
        iterations="${oc.select: iterations,5000}"
    )