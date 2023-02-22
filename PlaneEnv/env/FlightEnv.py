import numpy as np
from tensorforce.environments import Environment
from .FlightModel import FlightModel


class PlaneEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.FlightModel = FlightModel()
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)
        self.NUM_THRUST = len(self.FlightModel.thrust_act_vec)
        self.NUM_THETA = len(self.FlightModel.theta_act_vec)
        self.max_step_per_episode = 1000
        self.finished = False
        self.crashed = False
        self.episode_end = False
        self.STATES_SIZE = len(self.FlightModel.obs)
        self.V = []

    def states(self):
        return dict(type="float", shape=(self.STATES_SIZE,))

    def actions(self):
        return {
            "thrust": dict(type="int", num_values=self.NUM_THRUST),
            "theta": dict(type="int", num_values=self.NUM_THETA),
        }

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return self.max_step_per_episode

    # Optional
    def close(self):
        super().close()

    def reset(self):
        state = np.zeros(shape=(self.STATES_SIZE,))
        self.FlightModel = FlightModel()
        return state

    def execute(self, actions):
        reward = 0
        nb_timesteps = 30
        for i in range(1, nb_timesteps + 1):
            next_state = self.FlightModel.compute_timestep(actions, nb_timesteps)
            reward += self.reward()
            if self.terminal():
                reward = reward / i
                break
        if i == nb_timesteps:
            reward = reward / nb_timesteps
        # reward = self.reward()
        return next_state, self.terminal(), reward

    def terminal(self):
        self.finished = self.FlightModel.Pos[1] <= 0 #change this to range around 0
        self.episode_end = (self.FlightModel.timestep > self.max_step_per_episode)
        """
        for i in self.FlightModel.Pos_vec[1]:
            if i < 0:
                self.crashed = True
        """
        return self.finished or self.episode_end # or self.crashed

    def reward(self):
        reward = 0
        if self.finished:
            if self.FlightModel.Pos[1] <= 0:# reward = np.log((200-(np.abs(64-self.FlightModel.V[0]))) ** 3) #64 m/s is typical landing speed of Airbus a320. 
                reward = (10000-(self.FlightModel.V[0] + self.FlightModel.V[1]))**2
            else:
                reward = 0
            '''
            if self.FlightModel.Pos[0] < 1000:
                reward = -1.0 * ((5000 - self.FlightModel.Pos[0]) ** 5)
            elif 1000 <= self.FlightModel.Pos[0]: #test this section, deduct for there being too few timesteps
                reward = (5000 - self.FlightModel.Pos[0]) ** 5
            #elif self.FlightModel.Pos[0] >= 40:
            #    reward = np.log(((5000 - self.FlightModel.Pos[0]) ** 2))
            '''
        else:
            reward = -1
        return reward
