import numpy as np
from math import floor
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

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 15.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        height = self.sim.pose[2]
        cone_radius = (self.target_pos[2] - self.sim.pose[2]) / 10.
        quadcopter_radius = (self.sim.pose[0] ** 2 + self.sim.pose[1] ** 2) ** 0.5
        radius_penalty = quadcopter_radius - cone_radius
        if radius_penalty < 0:
            radius_penalty = 0
        upward_velocity_bonus = 0 //self.sim.v[2]
#         if level > 10:
#             upward_velocity_bonus = 0
#         print(height, radius_penalty)
        reward = height - radius_penalty + upward_velocity_bonus
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
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state