import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
import copy
from typing import Optional, Dict, Any
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
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)


class PDMSimulator:
    """
    Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation.
    """

    def __init__(self, proposal_sampling: TrajectorySampling,
                 tracker_style: str = 'default',
                 post_style: str = 'none',
                 post_params: Optional[Dict[str, float]] = None,
                 post_apply_mode: str = 'auto',
                 tracker_params: Optional[Dict[str, Any]] = None,
                 tracker: Optional[BatchLQRTracker] = None):
        """
        Constructor of PDMSimulator.
        :param proposal_sampling: Sampling parameters for proposals
        """

        # time parameters
        self.proposal_sampling = proposal_sampling

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()

        # LQR tracker selection: explicit instance -> params -> style preset
        if tracker is not None:
            print("💗tracker style: custom-instance")
            self._tracker = tracker
        elif tracker_params is not None:
            print("💗tracker style: custom-params")
            self._tracker = BatchLQRTracker(**tracker_params)
        else:
            # LQR tracker style selection (configurable via Hydra: simulator.tracker_style=...)
            style = (tracker_style or 'default').lower()
            print("💗tracker style:", style)
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
        # How to apply post transform:
        # - 'offline': apply once after rollout (legacy behavior)
        # - 'online': apply at every rollout step (affects subsequent dynamics)
        # - 'auto'  : online only when params actually change something (scale/bias/noise)
        self._post_apply_mode = (self._post_params.get('apply_mode', post_apply_mode) or 'auto').lower()


    def _post_effective(self) -> bool:
        """Return True if post params will actually change something vs defaults."""
        style = self._post_style
        if style == 'none':
            return False

        heading_scale = float(self._post_params.get('heading_scale', 1.0))
        heading_bias = float(self._post_params.get('heading_bias', 0.0))
        speed_scale = float(self._post_params.get('speed_scale', 1.0))
        speed_bias = float(self._post_params.get('speed_bias', 0.0))
        noise_std = float(self._post_params.get('noise_std', 0.0))

        # Command/model-level post dynamics (applied online during rollout)
        steer_rate_gain = float(self._post_params.get('steer_rate_gain', 1.0))
        steer_rate_bias = float(self._post_params.get('steer_rate_bias', 0.0))
        accel_gain = float(self._post_params.get('accel_gain', 1.0))
        accel_bias = float(self._post_params.get('accel_bias', 0.0))

        command_delay_steps = int(self._post_params.get('command_delay_steps', 0) or 0)
        command_delay_s = float(self._post_params.get('command_delay_s', 0.0) or 0.0)
        command_lpf_tau = float(self._post_params.get('command_lpf_tau', 0.0) or 0.0)
        command_lpf_alpha = float(self._post_params.get('command_lpf_alpha', 0.0) or 0.0)

        steer_gain_speed_k = float(self._post_params.get('steer_gain_speed_k', 0.0) or 0.0)

        wheelbase_scale = float(self._post_params.get('wheelbase_scale', 1.0) or 1.0)
        accel_time_constant_scale = float(self._post_params.get('accel_time_constant_scale', 1.0) or 1.0)
        steering_angle_time_constant_scale = float(self._post_params.get('steering_angle_time_constant_scale', 1.0) or 1.0)

        return (
            abs(heading_scale - 1.0) > 1e-12
            or abs(heading_bias) > 1e-12
            or abs(speed_scale - 1.0) > 1e-12
            or abs(speed_bias) > 1e-12
            or noise_std > 0.0
            or abs(steer_rate_gain - 1.0) > 1e-12
            or abs(steer_rate_bias) > 1e-12
            or abs(accel_gain - 1.0) > 1e-12
            or abs(accel_bias) > 1e-12
            or command_delay_steps > 0
            or command_delay_s > 1e-12
            or command_lpf_tau > 1e-12
            or command_lpf_alpha > 1e-12
            or abs(steer_gain_speed_k) > 1e-12
            or abs(wheelbase_scale - 1.0) > 1e-12
            or abs(accel_time_constant_scale - 1.0) > 1e-12
            or abs(steering_angle_time_constant_scale - 1.0) > 1e-12
        )


    def _should_apply_post_online(self) -> bool:
        mode = self._post_apply_mode
        if mode in ['online', 'rollout', 'step']:
            return self._post_style != 'none'
        if mode in ['offline', 'final', 'end']:
            return False
        # auto
        return self._post_effective()


    def _apply_post_transform_step(
        self,
        state_t: npt.NDArray[np.float64],
        initial_ego_state: EgoState,
    ) -> npt.NDArray[np.float64]:
        """Apply post transform to a single timestep state array (B, D)."""
        style = self._post_style
        if style == 'none':
            return state_t

        known_styles = {
            'yaw_scale_up', 'yaw_scale_down', 'yaw_scale',
            'speed_scale_up', 'speed_scale_down', 'speed_scale',
            'yaw_speed_extreme', 'aggressive_post',
            'noise', 'gaussian_noise', 'random_noise',
            # Command/model-level dynamics are handled elsewhere; keep list here for
            # legacy safety. (We do not want state-level edits for these.)
            'post_dynamics', 'post_dynamic', 'actuator', 'control', 'command',
        }
        if style not in known_styles:
            return state_t

        # state layout: [x, y, heading, vx, vy, ...]
        yaw = state_t[..., 2]
        vx = state_t[..., 3]
        vy = state_t[..., 4]

        yaw0 = float(initial_ego_state.rear_axle.heading)
        dyaw = yaw - yaw0

        heading_scale = float(self._post_params.get('heading_scale', 1.0))
        heading_bias = float(self._post_params.get('heading_bias', 0.0))
        speed_scale = float(self._post_params.get('speed_scale', 1.0))
        speed_bias = float(self._post_params.get('speed_bias', 0.0))
        noise_std = float(self._post_params.get('noise_std', 0.0))

        # Pure noise mode: mimic legacy behavior (noise on yaw/vx/vy), and let it affect
        # subsequent rollout dynamics without enforcing any additional coupling.
        if style in ['noise', 'gaussian_noise', 'random_noise']:
            if noise_std > 0.0:
                state_t[..., 2] = yaw + np.random.normal(0.0, noise_std, size=yaw.shape)
                state_t[..., 3] = vx + np.random.normal(0.0, noise_std, size=vx.shape)
                state_t[..., 4] = vy + np.random.normal(0.0, noise_std, size=vy.shape)
            return state_t

        # Command/model-level styles are applied before propagation; do nothing here.
        if style in ['post_dynamics', 'post_dynamic', 'actuator', 'control', 'command']:
            return state_t

        # current speed magnitude
        speed = np.sqrt(vx**2 + vy**2)

        # Apply deterministic transforms
        if style in ['yaw_scale_up', 'yaw_scale_down', 'yaw_scale']:
            yaw = yaw0 + dyaw * heading_scale + heading_bias
        elif style in ['speed_scale_up', 'speed_scale_down', 'speed_scale']:
            speed = speed * speed_scale + speed_bias
        elif style in ['yaw_speed_extreme', 'aggressive_post']:
            yaw = yaw0 + dyaw * heading_scale + heading_bias
            speed = speed * speed_scale + speed_bias

        # Apply noise (visual difference + affects subsequent dynamics when online)
        if noise_std > 0.0:
            yaw = yaw + np.random.normal(0.0, noise_std, size=yaw.shape)
            speed = speed + np.random.normal(0.0, noise_std, size=speed.shape)

        # Keep velocities consistent with (post) yaw while preserving (post) speed.
        # This ensures post yaw/speed changes actually affect subsequent rollout.
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)

        state_t[..., 2] = yaw
        state_t[..., 3] = vx
        state_t[..., 4] = vy
        return state_t


    def _apply_command_post_dynamics(
        self,
        command_states: npt.NDArray[np.float64],
        prev_state: npt.NDArray[np.float64],
        *,
        dt: float,
        delay_buf: Optional[npt.NDArray[np.float64]],
        lpf_state: Optional[npt.NDArray[np.float64]],
        init_delay_buf: bool,
    ) -> tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]], bool]:
        """Apply command-space actuator/dynamics mismatch.

        This modifies the tracker outputs (accel, steering_rate) before they are fed into the motion model.
        Implemented effects:
        - steer_rate_gain / steer_rate_bias
        - accel_gain / accel_bias
        - speed-dependent steer gain: steer_gain_speed_k (gain = max(0, 1 - k * speed))
        - command LPF: command_lpf_tau or command_lpf_alpha
        - fixed delay: command_delay_steps or command_delay_s

        Returns:
            cmd_to_apply, updated_delay_buf, updated_lpf_state, updated_init_delay_buf
        """

        cmd = np.array(command_states, copy=True)

        steer_rate_gain = float(self._post_params.get('steer_rate_gain', 1.0))
        steer_rate_bias = float(self._post_params.get('steer_rate_bias', 0.0))
        accel_gain = float(self._post_params.get('accel_gain', 1.0))
        accel_bias = float(self._post_params.get('accel_bias', 0.0))

        # speed-dependent steer gain (understeer at higher speeds)
        steer_gain_speed_k = float(self._post_params.get('steer_gain_speed_k', 0.0) or 0.0)
        if abs(steer_gain_speed_k) > 1e-12:
            speed = np.hypot(prev_state[..., StateIndex.VELOCITY_X], prev_state[..., StateIndex.VELOCITY_Y])
            extra_gain = np.clip(1.0 - steer_gain_speed_k * speed, a_min=0.0, a_max=None)
        else:
            extra_gain = 1.0

        cmd[..., DynamicStateIndex.STEERING_RATE] = (
            cmd[..., DynamicStateIndex.STEERING_RATE] * steer_rate_gain * extra_gain + steer_rate_bias
        )
        cmd[..., DynamicStateIndex.ACCELERATION_X] = (
            cmd[..., DynamicStateIndex.ACCELERATION_X] * accel_gain + accel_bias
        )

        # Low-pass filter (first-order) on commands
        command_lpf_alpha = float(self._post_params.get('command_lpf_alpha', 0.0) or 0.0)
        command_lpf_tau = float(self._post_params.get('command_lpf_tau', 0.0) or 0.0)
        if command_lpf_alpha > 0.0:
            alpha = float(np.clip(command_lpf_alpha, 0.0, 1.0))
        elif command_lpf_tau > 1e-12:
            alpha = float(dt / (dt + command_lpf_tau))
        else:
            alpha = 0.0

        if alpha > 0.0:
            if lpf_state is None:
                lpf_state = np.array(cmd, copy=True)
            else:
                lpf_state = (1.0 - alpha) * lpf_state + alpha * cmd
            cmd = lpf_state

        # Fixed delay on commands
        delay_steps = int(self._post_params.get('command_delay_steps', 0) or 0)
        delay_s = float(self._post_params.get('command_delay_s', 0.0) or 0.0)
        if delay_steps <= 0 and delay_s > 1e-12:
            delay_steps = int(round(delay_s / max(dt, 1e-9)))

        if delay_steps > 0:
            if delay_buf is None or delay_buf.shape[0] != delay_steps:
                delay_buf = np.zeros((delay_steps, cmd.shape[0], cmd.shape[1]), dtype=np.float64)
                init_delay_buf = True
            if init_delay_buf:
                delay_buf[:] = cmd[None, ...]
                init_delay_buf = False

            cmd_to_apply = delay_buf[0]
            delay_buf[:-1] = delay_buf[1:]
            delay_buf[-1] = cmd
            return cmd_to_apply, delay_buf, lpf_state, init_delay_buf

        return cmd, delay_buf, lpf_state, init_delay_buf

        
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
        # print("💗tracker style:",self._tracker_style)
        # Avoid mutating the shared vehicle parameters in-place when modeling wheelbase mismatch.
        vehicle_parameters = copy.deepcopy(initial_ego_state.car_footprint.vehicle_parameters)
        wheelbase_scale = float(self._post_params.get('wheelbase_scale', 1.0) or 1.0)
        if abs(wheelbase_scale - 1.0) > 1e-12 and self._should_apply_post_online():
            vehicle_parameters.wheel_base = float(vehicle_parameters.wheel_base) * wheelbase_scale

        self._motion_model._vehicle = vehicle_parameters
        self._tracker._discretization_time = self.proposal_sampling.interval_length

        # Optionally scale motion-model internal time constants (lag). Only meaningful in online mode.
        orig_accel_tau = float(getattr(self._motion_model, '_accel_time_constant', 0.2))
        orig_steer_tau = float(getattr(self._motion_model, '_steering_angle_time_constant', 0.05))
        accel_tau_scale = float(self._post_params.get('accel_time_constant_scale', 1.0) or 1.0)
        steer_tau_scale = float(self._post_params.get('steering_angle_time_constant_scale', 1.0) or 1.0)

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

        apply_post_online = self._should_apply_post_online()

        # Apply motion-model lag scaling only when using online post.
        if apply_post_online:
            if abs(accel_tau_scale - 1.0) > 1e-12:
                self._motion_model._accel_time_constant = max(0.0, orig_accel_tau * accel_tau_scale)
            if abs(steer_tau_scale - 1.0) > 1e-12:
                self._motion_model._steering_angle_time_constant = max(0.0, orig_steer_tau * steer_tau_scale)

        # Command-level post dynamics state (delay/LPF)
        delay_buf = None
        lpf_state = None
        init_delay_buf = True

        try:
            for time_idx in range(1, self.proposal_sampling.num_poses + 1):
                sampling_time: TimePoint = (
                    next_iteration.time_point - current_iteration.time_point
                )

                command_states = self._tracker.track_trajectory(
                    current_iteration,
                    next_iteration,
                    simulated_states[:, time_idx - 1],
                )

                # Online post dynamics in command space (actuator mismatch, delay, filtering).
                if apply_post_online and self._post_style in [
                    'post_dynamics', 'post_dynamic', 'actuator', 'control', 'command'
                ]:
                    command_states, delay_buf, lpf_state, init_delay_buf = self._apply_command_post_dynamics(
                        command_states,
                        simulated_states[:, time_idx - 1],
                        dt=float(self.proposal_sampling.interval_length),
                        delay_buf=delay_buf,
                        lpf_state=lpf_state,
                        init_delay_buf=init_delay_buf,
                    )

                simulated_states[:, time_idx] = self._motion_model.propagate_state(
                    states=simulated_states[:, time_idx - 1],
                    command_states=command_states,
                    sampling_time=sampling_time,
                )

                # Online post: modify heading/speed at each step so it impacts subsequent dynamics.
                if apply_post_online:
                    simulated_states[:, time_idx] = self._apply_post_transform_step(
                        simulated_states[:, time_idx], initial_ego_state
                    )

                current_iteration = next_iteration
                next_iteration = SimulationIteration(
                    current_iteration.time_point + delta_time_point, 1 + time_idx
                )
        finally:
            # Restore motion-model parameters to avoid leaking across calls.
            self._motion_model._accel_time_constant = orig_accel_tau
            self._motion_model._steering_angle_time_constant = orig_steer_tau

        # Offline post (legacy): apply once after rollout.
        if not apply_post_online:
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
