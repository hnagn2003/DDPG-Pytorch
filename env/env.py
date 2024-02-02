import numpy as np
import math
class UAVEnv():
    def __init__(self):
        # state = [alpha, q, theta, M, h]
        self.cur_obs = 0, 2, 5000
        self.Rho = 1
        self.V_s = 340
        self.int_e_2 = 0
        self.dt = 0.001
        self.T_end = 20
        self.a_zc = 30
        
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
        # TODO return observation
        pass
    def is_terminate(self):
        # TODO
        return False
    def get_state(self, observation, action): # 
        k_DC, k_A, k_I, k_g = action
        alpha, M, h = observation
        q = None # TODO

        for t in np.arange(0, self.T_end, self.dt):
            # Calculate Autopilot
            e_1 = k_DC - self.a_z  # somehow we gain a_z
            e_2 = k_A * e_1 - q
            int_e_2 = int_e_2 + e_2 * self.dt
            e_3 = k_I * int_e_2 - q
            delta = k_g * e_3
            
            # Equation 17
            C_A = self.a_a
            C_N = self.a_n*(alpha)**3+self.b_m*alpha*abs(alpha)+self.c_n*(2-M/3)*alpha+self.d_n*delta
            C_M = self.a_m*(alpha)**3+self.b_m*alpha*abs(alpha)+self.c_m*(-7+8*M/3)*alpha+self.d_m*delta
            # Equation 16
            V = M * self.V_s
            Q = 0.5 * self.Rho * V**2

            alpha_dot = Q*self.S/self.m/V*(C_N*math.sin(alpha)+C_A*math.cos(alpha))+self.g/V*math.cos(gamma)+q  # TODO
            q_dot = Q*self.S*self.d/self.Iyy*C_M      # TODO
            theta_dot = q  # TODO
            M_dot = Q*self.S/self.m*self.V_s(C_N*math.sin(alpha)+C_A*math.cos(alpha)-self.g/self.V_s*math.sin(gamma))      # TODO
            h_dot = V*math.sin(gamma)     # TODO

            # Calculate Flight Vehicle outputs (Equation 16)
            gamma = theta - alpha

            # Integration using Euler’s method
            self.a_z = V * (theta_dot - alpha_dot)
            alpha = alpha + alpha_dot * self.dt
            q = q + q_dot * self.dt
            theta = theta + theta_dot * self.dt
            M = M + M_dot * self.dt
            h = h + h_dot * self.dt
        return {self.a_z, alpha, q, theta, M, h}
    def step(self, action):
        # TODO return new observation based on action, reward, check terminate
        reward = -self.k_a*((self.a_z-self.a_zc)/self.a_zmax)**2-self.k_delta*0
        return self.get_observation(), reward, self.is_terminate()
