import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goals, Rewards, Penalties
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 15.])
        self.close_enough_distance = 5.
        self.below_zero_penalty = -20.
        self.close_enough_bonus = 1000.
        self.timeout_penalty = -10
        self.base_reward = 1.
        self.height_reward = 1.
        self.radius_penalty = -10.
        self.cone_sharpness = 1.

    def get_reward(self):
        """Uses current pose of sim to return reward."""
#         height_percentage = self.sim.pose[2] / self.target_pos[2]
        cone_radius = (self.target_pos[2] - self.sim.pose[2]) / self.cone_sharpness
        quadcopter_radius = (self.sim.pose[0] ** 2 + self.sim.pose[1] ** 2) ** 0.5      
        radius_penalty_percentage = (quadcopter_radius - cone_radius) / cone_radius
        if radius_penalty_percentage < 0:
            radius_penalty_percentage = 0
        reward = self.base_reward
        reward += self.height_reward * self.sim.pose[2]
        reward += self.radius_penalty * radius_penalty_percentage
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        distance_from_target = abs(self.sim.pose[:3] - self.target_pos).sum()
        if distance_from_target < self.close_enough_distance:
            reward += self.close_enough_bonus
            done = True
        else:
            if done:
                if self.sim.pose[2] <= 0.:
                    reward += self.below_zero_penalty
                else:
                    reward += self.timeout_penalty
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
