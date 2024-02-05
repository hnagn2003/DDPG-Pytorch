import numpy as np
import math
class UAVEnv():
    def __init__(self):
        # state = [alpha, q, theta, M, h]
        self.last_action = None
        self.Rho = 1
        self.V_s = 340
        self.int_e_2 = 0
        self.dt = 0.001
        self.T_end = 20
        self.a_zc = 30
        #state
        self.alpha = 0
        self.q = 0
        self.theta = 0
        self.M = 2
        self.h = 5000
        self.last_observation = self.alpha, self.M, self.h
        #Table 1
        self.g = 9.8 #Gravitational acceleration
        self.S =  0.0409 #Reference Area
        self.m =  204.02 #mass
        self.Iyy = 247.439 #Moment of Inertia
        self.d =  0.2286 #Reference Distance
        #Table 2
        self.a_a = 0.3
        self.a_n = 19.373
        self.b_n = -31.023
        self.c_n = -9.717
        self.d_n = -1.948
        self.a_m = 40.44
        self.b_m = -64.015
        self.c_m = 2.922
        self.d_m = -11.803
        #Table 3
        self.k_a = 1
        self.k_delta = 0.1
        self.a_zmax = 100
        self.delta_max_dot = 1.5
    def reset(self):
        # TODO reset observation (angle of attack, Mach number, height)
        pass
    def get_observation(self):
        return self.alpha, self.M, self.h
    def is_terminate(self):
        # TODO
        return False
    def get_state(self): # 
        return {self.a_z, self.alpha, self.q, self.theta, self.M, self.h}
    def step(self, action):
        # TODO return new observation based on action, reward, check terminate
        self.last_observation = self.get_observation()
        k_DC, k_A, k_I, k_g = action
    
        for t in np.arange(0, self.T_end, self.dt):
            # Calculate Autopilot
            e_1 = k_DC - self.a_z  # somehow we gain a_z
            e_2 = k_A * e_1 - self.q
            int_e_2 = int_e_2 + e_2 * self.dt
            e_3 = k_I * int_e_2 - self.q
            delta = k_g * e_3
            
            # Equation 17
            C_A = self.a_a
            C_N = self.a_n*(self.alpha)**3+self.b_m*self.alpha*abs(self.alpha)+self.c_n*(2-self.M/3)*self.alpha+self.d_n*delta
            C_M = self.a_m*(self.alpha)**3+self.b_m*self.alpha*abs(self.alpha)+self.c_m*(-7+8*self.M/3)*self.alpha+self.d_m*delta
            # Equation 16
            V = self.M * self.V_s
            Q = 0.5 * self.Rho * V**2

            alpha_dot = Q*self.S/self.m/V*(C_N*math.sin(self.alpha)+C_A*math.cos(self.alpha))+self.g/V*math.cos(gamma)+self.q  # TODO
            q_dot = Q*self.S*self.d/self.Iyy*C_M      # TODO
            theta_dot = self.q  # TODO
            M_dot = Q*self.S/self.m*self.V_s(C_N*math.sin(self.alpha)+C_A*math.cos(self.alpha)-self.g/self.V_s*math.sin(gamma))      # TODO
            h_dot = V*math.sin(gamma)     # TODO

            # Calculate Flight Vehicle outputs (Equation 16)
            gamma = self.theta - self.alpha

            # Integration using Euler’s method
            self.a_z = V * (theta_dot - alpha_dot)
            self.alpha = self.alpha + alpha_dot * self.dt
            self.q = self.q + q_dot * self.dt
            self.theta = self.theta + theta_dot * self.dt
            self.M = self.M + M_dot * self.dt
            self.h = self.h + h_dot * self.dt
        reward = -self.k_a*((self.a_z-self.a_zc)/self.a_zmax)**2-self.k_delta*0
        self.observation = self.get_observation()
        # return self.get_state(self.get_observation(), action), self.get_observation(), reward, self.is_terminate()
        return self.observation, self.last_observation, reward, self.is_terminate()