import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        self.t0 = 0.0  # initial time
        self.tf = 120  # final time
        self.dt = 0.1  # time step
        self.t = np.arange(self.t0, self.tf+self.dt, self.dt)
        self.observation_space  = spaces.Box(low=np.array([0]), high = np.array([1000]), dtype=np.float64)
        self.action_space = spaces.Box(low = np.array([0]), high=np.array([2]), dtype=np.int16)
        self.X0 = 0.3e5  # cell/ml
        self.S0 = 0.2 # g/L
        self.E0 = 0 # U/L 
        self.i = 0
        # Process conditions

        # Model Parameters
        self.Ks = 0.1 #g/L substrate saturation coefficient
        self.C = 0.000001 # ug/cell - glucose consumption per new cell created (growth coefficient)
        self.MuX = 0.1 # 1/hr
        self.MuE = .000001 # U/(cell*hr)
        self.MuD = 0 # 1/hr
        self.mu_opt = 1.8
        self.T_opt = 37
        self.A_opt = 250
        self.r_t = 3.12
        self.r_a = 4.5
        self.temperature = 32
        self.feed = 0.02
                # Substrate Values
        self.S = np.zeros(int(self.tf/self.dt)+1)
        
        self.enzyme_activity = []
        # Initial Substrate
        self.S[0] = self.S0

        # Cell Values
        self.X = np.zeros(int(self.tf/self.dt)+1)
        # initial Cell concentration value
        self.X[0] = self.X0

        # Enzyme concentration
        self.E0 = 0
        self.E = np.zeros(int(self.tf/self.dt)+1)
        self.E[0] = self.E0
        self.enzyme_state = sum(self.E)
        # Time steps
        self.t = np.arange(self.t0, self.tf+self.dt, self.dt)

        # Change in Enzyme 
        self.E_C = np.zeros(int(self.tf/self.dt))

        self.divide = 10
        self.timesteps = len(self.t)/self.divide
        self.cycle = 0
        print("Initial conditions:")
        print(f"Iteration : {self.i}, temperature: {self.temperature}, enzyme activity: {sum(self.E)}")


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.t0 = 0.0  # initial time
        self.tf = 120  # final time
        self.dt = 0.1  # time step
        self.t = np.arange(self.t0, self.tf+self.dt, self.dt)
        self.i = 0
        self.X0 = 0.3e5  # cell/ml
        self.S0 = 0.2 # g/L
        self.E0 = 0 # U/L 
        # Process conditions

        # Model Parameters
        self.Ks = 0.1 #g/L substrate saturation coefficient
        self.C = 0.000001 # ug/cell - glucose consumption per new cell created (growth coefficient)
        self.MuX = 0.1 # 1/hr
        self.MuE = .000001 # U/(cell*hr)
        self.MuD = 0 # 1/hr
        self.mu_opt = 1.8
        self.T_opt = 37
        self.A_opt = 250
        self.r_t = 3.12
        self.r_a = 4.5
        self.temperature = 32
        self.feed = 0.02
        self.enzyme_state = self.E0

                # Substrate Values
        self.S = np.zeros(int(self.tf/self.dt)+1)

        # Initial Substrate
        self.S[0] = self.S0

        # Cell Values
        self.X = np.zeros(int(self.tf/self.dt)+1)

        # initial Cell concentration value
        self.X[0] = self.X0

        # Enzyme concentration
        self.E0 = 0
        self.E = np.zeros(int(self.tf/self.dt)+1)
        self.E[0] = self.E0
        self.enzyme_state = sum(self.E)
        # Time steps
        self.t = np.arange(self.t0, self.tf+self.dt, self.dt)

        # Change in Enzyme 
        self.E_C = np.zeros(int(self.tf/self.dt))

        self.divide = 10
        self.timesteps = len(self.t)/self.divide
        self.cycle = 0

        observation = np.array([self.E0],dtype=float)
        info = {}

        return observation, info


    def step(self, action):

        action = math.ceil(action[0])
        initial_cordinates = [self.i, self.E[self.i]]  
        if action == 2:
            self.temperature -= 0.5
        else:
            self.temperature += action/2

        MuX =  self.mu_opt*(math.exp(-((self.temperature - self.T_opt)**2)/self.r_t**2))

        dXdt = (MuX *self.S[self.i]) / (self.Ks + self.S[self.i]) * self.X[self.i]
        
        dSdt = ( -self.C * MuX*self.S[self.i]/(self.Ks + self.S[self.i]) * self.X[self.i] )

        delX = dXdt * self.dt
        delS = dSdt * self.dt

        self.X[self.i+1] = self.X[self.i] + delX
        self.S[self.i+1] = self.S[self.i] + delS

        delE = self.MuE * dXdt * self.dt

        step = self.i
        nts = 1

        while step > 0 and nts < 500:
            delE = delE + (self.X[self.i+1] - self.X[self.i]) * (self.MuE)*(1/nts)
            nts += 1
            step -= 1
        
        self.E_C[self.i] = delE
        self.E[self.i+1] = self.E[self.i] +  delE

        if self.i+1 > 51:
            self.E[self.i+1] = self.E[self.i+1] - self.E_C[self.i-50]

        final_cordinates = [self.i+1, self.E[self.i+1]]
        slope = (final_cordinates[1] - initial_cordinates[1]) / (final_cordinates[0] - initial_cordinates[0])

        if self.E[self.i+1] > self.E[self.i]:
            slope = 100 * slope
            reward = 10 + slope
        else:
            reward = 10 * slope

        self.enzyme_state = sum(self.E)

        if self.i >= 1200:
            print("terminating because of iterations")
            done = True
        else:
            done = False
        
        if (self.E[self.i+1] - self.E[self.i]) < 0 :  
            print("terminating because of decrease in enzyme activity")
            done = True

        info = {}
        self.i += 1
        print(f"for iteration {self.i} temperature value is {self.temperature}, action taken : {action} and Enzyme_state : {self.enzyme_state}, reward : {reward}")
        return np.array([self.enzyme_state]), reward, done, info

    def render(self):
        pass

    
    def close(self):
        pass
