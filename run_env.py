import numpy as np
import gymnasium
from gymnasium import spaces
import math


class GridWorldEnv(gymnasium.Env):

    def run_env(self):
        self.t0 = 0.0  # initial time 
        self.tf = 120  # final time minutes
        self.dt = 0.1  # time step
        self.observation_space  = spaces.Box(low=np.array([0]), high = np.array([1000]), dtype=np.float64)
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
        self.temperature = 32 # Degree celsius

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

        terminate = False
        # Termincation condition
        if self.i >= 1200:
            terminate = True
        else:
            
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

        info = {}
        self.i += 1
        return np.array([self.E[self.i+1]]), reward, terminate, False, info
