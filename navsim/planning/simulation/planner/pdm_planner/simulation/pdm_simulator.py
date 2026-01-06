import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from typing import Optional, Dict
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import (
    BatchKinematicBicycleModel,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import (
    BatchLQRTracker,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)


class PDMSimulator:
    """
    Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation.
    """

    def __init__(self, proposal_sampling: TrajectorySampling, tracker_style: str = 'default', post_style: str = 'none', post_params: Optional[Dict[str, float]] = None):
        """
        Constructor of PDMSimulator.
        :param proposal_sampling: Sampling parameters for proposals
        """

        # time parameters
        self.proposal_sampling = proposal_sampling

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()

        # LQR tracker style selection (configurable via Hydra: simulator.tracker_style=...)
        style = (tracker_style or 'default').lower()
    
        # import ipdb; ipdb.set_trace()
        if style == 'default':
            self._tracker = BatchLQRTracker()
        elif style in ['jitter_highhigh', 'highhigh_jitter']:
            self._tracker = BatchLQRTracker(
                q_longitudinal=[50000],
                r_longitudinal=[1e-6],
                q_lateral=[1, 1e6, 0],
                r_lateral=[1e-8],
                tracking_horizon=2,
                jerk_penalty=1e-8,
                curvature_rate_penalty=1e-8,
            )
        elif style in ['jitter_high', 'high_jitter']:
            self._tracker = BatchLQRTracker(
                q_longitudinal=[500],
                r_longitudinal=[1e-6],
                q_lateral=[1, 5000, 0],
                r_lateral=[1e-6],
                tracking_horizon=3,
                jerk_penalty=1e-6,
                curvature_rate_penalty=1e-6,
            )
        elif style in ['unstable', 'divergent']:
            self._tracker = BatchLQRTracker(
                q_longitudinal=[500],
                r_longitudinal=[1e-6],
                q_lateral=[1, 5000, 0],
                r_lateral=[1e-6],
                tracking_horizon=3,
                jerk_penalty=1e-6,
                curvature_rate_penalty=1e-6,
            )
        elif style in ['aggressive', 'overshoot']:
            self._tracker = BatchLQRTracker(
                q_longitudinal=[200],
                r_longitudinal=[0.1],
                q_lateral=[10, 500, 10],
                r_lateral=[0.01],
                tracking_horizon=5,
            )
        elif style in ['sluggish', 'lazy']:
            self._tracker = BatchLQRTracker(
                q_longitudinal=[5],
                r_longitudinal=[10],
                q_lateral=[0.1, 1, 0],
                r_lateral=[50],
                tracking_horizon=20,
                jerk_penalty=0.1,
                curvature_rate_penalty=0.1,
            )
        elif style in ['drunk', 'driving_drunken']:
            self._tracker = BatchLQRTracker(
                q_longitudinal=[1],
                r_longitudinal=[0.5],
                q_lateral=[0.01, 0.5, 5],
                r_lateral=[0.001],
                jerk_penalty=1e-8,
                curvature_rate_penalty=1e-8,
                tracking_horizon=4,
            )
        elif style in ['safe', 'cautious']:
            self._tracker = BatchLQRTracker(
                q_longitudinal=[1],
                r_longitudinal=[5],
                q_lateral=[10, 20, 5],
                r_lateral=[10],
                tracking_horizon=15,
                jerk_penalty=0.1,
                curvature_rate_penalty=0.1,
            )

        
        else:
            # Fallback to default
            self._tracker = BatchLQRTracker()
#FIXME:
        # post-simulation transform settings (to amplify differences deliberately)
        self._post_style = (post_style or 'none').lower()
        self._post_params = post_params or {}
        
        #==========================强烈强烈抖动 high-high-frequency jitter）=================
        # self._tracker = BatchLQRTracker(
        #     q_longitudinal=[50000],       # 强烈关注速度误差
        #     r_longitudinal=[1e-6],      # 加速度几乎免费

        #     # lateral 权重极端化：heading_error 惩罚极大，steering_rate 几乎免费
        #     q_lateral=[1, 1e6, 0],      
        #     r_lateral=[1e-8],           

        #     tracking_horizon=2,         # 极短视 → 反馈抖动明显
        #     jerk_penalty=1e-8,          
        #     curvature_rate_penalty=1e-8,
        # )

        #==========================强烈抖动 / 神经质风格（high-frequency jitter）=================
        # self._tracker=BatchLQRTracker(
        #     q_longitudinal=[500],   # 强烈关注速度误差
        #     r_longitudinal=[1e-6],  # 加速度代价接近零 → acceleration 不受限制

        #     q_lateral=[1, 5000, 0],  # heading_error 的代价极大 → 大力纠偏
        #     r_lateral=[1e-6],        # steering_rate 几乎免费 → 随便抖

        #     tracking_horizon=3,      # 短视 → 更抖
        #     jerk_penalty=1e-6,
        #     curvature_rate_penalty=1e-6,
        # )
    #     #=========================Style 2：发散风格（unstable / 极端）===========================
    #     self._tracker=BatchLQRTracker(
    #         q_longitudinal=[500],   # 强烈关注速度误差
    #         r_longitudinal=[1e-6],  # 加速度代价接近零 → acceleration 不受限制

    #         q_lateral=[1, 5000, 0],  # heading_error 的代价极大 → 大力纠偏
    #         r_lateral=[1e-6],        # steering_rate 几乎免费 → 随便抖

    #         tracking_horizon=3,      # 短视 → 更抖
    #         jerk_penalty=1e-6,
    #         curvature_rate_penalty=1e-6,
    #     )
    # #     #=========================Style 3：急躁、积极追踪但容易 overshoot（aggressive & oscillatory）===========================
    #     self._tracker=BatchLQRTracker(
    #         q_longitudinal=[200],
    #         r_longitudinal=[0.1],

    #         q_lateral=[10, 500, 10],   # heading_error 权重高 → 过强纠正
    #         r_lateral=[0.01],          # steering_rate 代价小 → 容易过冲

    #         tracking_horizon=5,
    #     )
                
    #    #=========================Style 4：延迟响应、迟钝风格（lazy / sluggish）===========================         
    #     self._tracker=BatchLQRTracker(
    #         q_longitudinal=[5],
    #         r_longitudinal=[10],       # 加速度代价大 → 不想加速

    #         q_lateral=[0.1, 1, 0],     # lateral error 不重要
    #         r_lateral=[50],            # steering_rate 超贵 → 不想转方向盘

    #         tracking_horizon=20,       # 超远 horizon → 平滑但迟钝
    #     )
                
    #     #=========================Style 5：醉驾风格（醉汉顶方向盘）===========================         
    #     self._tracker=BatchLQRTracker(
    #         q_longitudinal=[1],
    #         r_longitudinal=[0.5],

    #         q_lateral=[0.01, 0.5, 5],   # steering_angle 权重大 → 方向盘喜欢保持偏置姿态
    #         r_lateral=[0.001],          # steering_rate 较小 → 易震荡但不剧烈

    #         jerk_penalty=1e-8,
    #         curvature_rate_penalty=1e-8,
    #         tracking_horizon=4,
    #     )
    #     #=========================Style 6：老司机（平稳、顺滑、谨慎、偏向安全）===========================  
    #     self._tracker=BatchLQRTracker(
    #         q_longitudinal=[1],
    #         r_longitudinal=[5],        # 不轻易加速

    #         q_lateral=[10, 20, 5],     # 温和但能纠正
    #         r_lateral=[10],            # steering_rate 代价大 → 方向盘不会乱动

    #         tracking_horizon=15,
    #         jerk_penalty=0.1,
    #         curvature_rate_penalty=0.1,
    #     )        
        
        
#FIXME:
    def simulate_proposals(
        self, states: npt.NDArray[np.float64], initial_ego_state: EgoState
    ) -> npt.NDArray[np.float64]:
        """
        Simulate all proposals over batch-dim
        :param initial_ego_state: ego-vehicle state at current iteration
        :param states: proposal states as array
        :return: simulated proposal states as array
        """

        # TODO: find cleaner way to load parameters
        # set parameters of motion model and tracker
        self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        self._tracker._discretization_time = self.proposal_sampling.interval_length

        proposal_states = states[:, : self.proposal_sampling.num_poses + 1]
        self._tracker.update(proposal_states)

        # state array representation for simulated vehicle states
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = ego_state_to_state_array(initial_ego_state)

        # timing objects
        current_time_point = initial_ego_state.time_point
        delta_time_point = TimeDuration.from_s(self.proposal_sampling.interval_length)

        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + delta_time_point, 1)

        for time_idx in range(1, self.proposal_sampling.num_poses + 1):
            sampling_time: TimePoint = (
                next_iteration.time_point - current_iteration.time_point
            )

            command_states = self._tracker.track_trajectory(
                current_iteration,
                next_iteration,
                simulated_states[:, time_idx - 1],
            )

            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,
            )

            current_iteration = next_iteration
            next_iteration = SimulationIteration(
                current_iteration.time_point + delta_time_point, 1 + time_idx
            )
        # apply post transform to aggressively modify heading/speed if configured
        simulated_states = self._apply_post_transform(simulated_states, initial_ego_state)
        return simulated_states

    def _apply_post_transform(
        self,
        states: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
    ) -> npt.NDArray[np.float64]:
        style = self._post_style
        # import ipdb; ipdb.set_trace()
        if style == 'none':
            return states

        # state layout assumption: [x, y, heading, vx, vy, ...]
        x = states[..., 0]
        y = states[..., 1]
        yaw = states[..., 2]
        vx = states[..., 3]
        vy = states[..., 4]

        # relative yaw to initial
        yaw0 = initial_ego_state.rear_axle.heading
        dyaw = yaw - yaw0

        heading_scale = float(self._post_params.get('heading_scale', 1.0))
        heading_bias = float(self._post_params.get('heading_bias', 0.0))  # [rad]
        speed_scale = float(self._post_params.get('speed_scale', 1.0))
        speed_bias = float(self._post_params.get('speed_bias', 0.0))  # [m/s]
        noise_std = float(self._post_params.get('noise_std', 0.0))

        if style in ['yaw_scale_up', 'yaw_scale_down', 'yaw_scale']:
            yaw = yaw0 + dyaw * heading_scale + heading_bias
            states[..., 2] = yaw
        elif style in ['speed_scale_up', 'speed_scale_down', 'speed_scale']:
            speed = np.sqrt(vx**2 + vy**2)
            speed = speed * speed_scale + speed_bias
            # keep velocity direction, rescale magnitude
            dir_norm = np.sqrt(vx**2 + vy**2) + 1e-9
            vx = vx / dir_norm * speed
            vy = vy / dir_norm * speed
            states[..., 3] = vx
            states[..., 4] = vy
        elif style in ['yaw_speed_extreme', 'aggressive_post']:
            yaw = yaw0 + dyaw * heading_scale + heading_bias
            states[..., 2] = yaw
            speed = np.sqrt(vx**2 + vy**2)
            speed = speed * speed_scale + speed_bias
            dir_norm = np.sqrt(vx**2 + vy**2) + 1e-9
            vx = vx / dir_norm * speed
            vy = vy / dir_norm * speed
            states[..., 3] = vx
            states[..., 4] = vy

        if noise_std > 0.0:
            # add Gaussian noise to heading and velocities
            states[..., 2] = states[..., 2] + np.random.normal(0.0, noise_std, size=states[..., 2].shape)
            states[..., 3] = states[..., 3] + np.random.normal(0.0, noise_std, size=states[..., 3].shape)
            states[..., 4] = states[..., 4] + np.random.normal(0.0, noise_std, size=states[..., 4].shape)

        return states
