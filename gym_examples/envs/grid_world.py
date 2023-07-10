import numpy as np
import gymnasium
from gymnasium import spaces
import math
import pandas as pd


class GridWorldEnv(gymnasium.Env):

    def __init__(self):
        self.t0 = 0.0  # initial time 
        self.tf = 120  # final time minutes
        self.dt = 0.1  # time step
        self.observation_space  = spaces.Box(low=np.array([0]), high = np.array([20]), dtype=np.float64)
        self.action_space = spaces.Box(low = np.array([0]), high=np.array([2]), dtype=np.int16) 
        self.X0 = 0.3e5  # cell/ml
        self.S0 = 0.2 # g/L
        self.E0 = 0 # U/L 
        self.i = 0

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
        self.temperature = np.random.randint(low=32, high=35)

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

        # Time steps
        self.t = np.arange(self.t0, self.tf+self.dt, self.dt)

        # Change in Enzyme 
        self.E_C = np.zeros(int(self.tf/self.dt))

       
        

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.t0 = 0.0  # initial time
        self.tf = 120  # final time
        self.dt = 0.1  # time step
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
        self.temperature = np.random.randint(low=32, high=35)

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

        # Time steps
        self.t = np.arange(self.t0, self.tf+self.dt, self.dt)

        # Change in Enzyme 
        self.E_C = np.zeros(int(self.tf/self.dt))

        # Reset makes the enzyme concetration 0
        observation = np.array([self.E0],dtype=float)
        info = {}

        with open("plot.txt",'w') as f:
            f.truncate(0)
        return observation, info


    def step(self, action):
        # Termincation condition
        if self.i == 1199:
            terminate = True
            return np.array([self.E[self.i]]), 0, terminate, False, {} 
        else:
            terminate = False
            action = math.ceil(action[0])
            initial_cordinates = [self.i, self.E[self.i]]  
            if action == 2:
                self.temperature -= 0.1
            else:
                self.temperature += action/10

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

            if slope < 0:
                reward = 10 * slope
                terminate = True
            else:
                slope = 100 * slope
                reward = 10 + slope

            with open('plot.txt','a') as plotting_file:
                plotting_file.write(f"{self.i},{self.temperature}\n")
            self.i += 1
            data = {
            'experiment_iteration': self.i,
            'cells': self.X[self.i],
            'substrate': self.S[self.i],
            'Enzyme' : self.E[self.i]
            }
            df = pd.DataFrame(data)
            df.to_csv("output_logs.csv",mode='a', index=True)
            return np.array([self.E[self.i]]), reward, terminate, False, {}

    def render(self):
        pass

    
    def close(self):
        pass
