import numpy as np
import gymnasium
from gymnasium import spaces
import math


class GridWorldEnv(gymnasium.Env):

    def __init__(self):
        self.observation_space  = spaces.Box(low=np.array([0]), high = np.array([100]), dtype=np.float64)
        self.action_space = spaces.Box(low = np.array([0]), high=np.array([2]), dtype=np.int16)

       # Initial conditions
        self.X0 = 0.3e5  # cell/ml
        self.S0 = 0.2 # g/L
        self.E0 = 0.0 # U/L 
        # Process conditions
        self.T = np.random.randint(low=32, high=35) #'C
        self.A = 250 # RPM

        # Substrate addition : add 0.05 evert 24 hours
        self.sadd = 0.05
        self.sadd_t = 5 # add every 5 hours

        #model parameters
        self.Ks = 0.1
        self.C = 0.000001

        self.MuE = .00000001

        self.MuD = 0

        self.del_t = 0.1
        self.sadd_s = self.sadd_t/self.del_t
        self.t_end = 24*7
        self.tvec = np.arange(0, self.t_end, self.del_t)
        self.ns = len(self.tvec)

        # X S E delE delX
        self.D = np.zeros((self.ns+1, 5))

        self.D[0][0] = self.X0
        self.D[0][1] = self.S0
        self.D[0][2] = self.E0
        self.i = 0

        self.media = 10 # Liters
        self.max_capacity = 50 # Liters
        self.media_added_in_liters = self.sadd/self.S0

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
       # Initial conditions
        self.X0 = 0.3e5  # cell/ml
        self.S0 = 0.2 # g/L
        self.E0 = 0.0 # U/L 
        # Process conditions
        self.T = np.random.randint(low=32, high=35) #'C
        self.A = 250 # RPM

        # Substrate addition : add 0.05 evert 24 hours
        self.sadd = 0.05
        self.sadd_t = 5 # add every 5 hours

        #model parameters
        self.Ks = 0.1
        self.C = 0.000001

        self.MuE = .00000001

        self.MuD = 0

        self.del_t = 0.1
        self.sadd_s = self.sadd_t/self.del_t
        self.t_end = 24*7
        self.tvec = np.arange(0, self.t_end, self.del_t)
        self.ns = len(self.tvec)

        # X S E delE delX
        self.D = np.zeros((self.ns+1, 5))

        self.D[0][0] = self.X0
        self.D[0][1] = self.S0
        self.D[0][2] = self.E0

        self.media = 10 # Liters
        
        # Reset makes    the enzyme concetration 0
        observation = np.array([self.E0],dtype=float)
        info = {}
        self.i = 0
        with open("plot.txt",'w') as f:
            f.truncate(0)
        return observation,info

    def step(self, action):
        if self.i == int(self.t_end/self.del_t):
            terminate = True
            return np.array([self.D[self.i][2]]), 0, terminate, False, {}
        elif self.media >= self.max_capacity:
            terminate = True
            return np.array([self.D[self.i][2]]), 0, terminate, False, {}
        else:
            terminate = False
            action = math.ceil(action[0])
            initial_cordinates = [self.i, self.D[self.i][2]]  
            if action == 2:
                self.T -= 0.1
            else:
                self.T += action/10

            MuX =  0.1*(math.exp(-((self.T - 37)**2)/5**2))
    
            dXdt = (MuX *self.D[self.i][1]) / (self.Ks + self.D[self.i][1]) * self.D[self.i][0]

            dSdt = ( -self.C * MuX*self.D[self.i][1]/(self.Ks+self.D[self.i][1]) * self.D[self.i][0] ) 

            delX = dXdt * self.del_t
            delS = dSdt * self.del_t

            # Change in cells
            self.D[self.i+1][4] = delX

            # time of cell death 
            tcd = 48
            tcd_s = math.floor(tcd/self.del_t)
            cellsub = 0

            if self.i+1 == tcd_s:
                cellsub = self.X0 # seeded cells die on this step

            elif self.i+1 > tcd_s :
                cellsub = self.D[(self.i+1)-tcd][4]

            # Update cells
            self.D[self.i+1][0] = self.D[self.i][0] + delX - cellsub

            # Update substrate 
            if self.i % self.sadd_s == 0:
                delS = delS + self.sadd
                self.media += self.media_added_in_liters
            
            self.D[self.i+1][1] = self.D[self.i][1] + delS

            # Enzyme determination 

            delE = self.MuE * dXdt*self.del_t # new enzyme from fresh cells

            step = self.i
            nts = 1

            t_ev = 20 # hours cells produce enzyme with linear depletion
            s_ev = 20/self.del_t # Timesteps cells can make enzyme

            for j in range(0,int(s_ev)):
                ts = self.i - j
                if ts < 1 :
                    break
                else:
                    delE = delE + (- (self.MuE/s_ev)*j+self.MuE) * self.D[ts][4]

            # Change in enzyme    
            self.D[self.i+1][3] = delE
            
            #Update enzyme variable
            self.D[self.i+1][2] = self.D[self.i][2] + delE

            # Degradation of enzymes
            EDT = 72

            if self.i+1>(EDT/self.del_t+1):
                self.D[self.i+1][2] = self.D[self.i+1][2] - self.D[self.i+1 - int(EDT/self.del_t)][3]
            
            final_cordinates = [self.i+1, self.D[self.i+1][2]]
            
            slope = (final_cordinates[1] - initial_cordinates[1]) / (final_cordinates[0] - initial_cordinates[0])

            if slope < 0:
                reward = 10 * slope
                terminate = True
            else:
                slope = 100 * slope
                reward = 10 + slope

            with open('plot.txt','a') as plotting_file:
                plotting_file.write(f"{self.tvec[self.i]},{self.D[self.i][0]/1e6},{self.D[self.i][1]},{self.D[self.i][2]}\n")
            self.i += 1
            
            return np.array([self.D[self.i][2]]), reward, terminate, False, {}