
import numpy as np 
import torch
import torch.nn.functional as F
# https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OURandomProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OURandomProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
        
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device=torch.device("cpu")):
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.terminates = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self.size, self.current_index = 0, 0

    def add(self, state, action, reward, next_state, terminate):
        self.states[self.current_index] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.current_index] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.current_index] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.current_index] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.terminates[self.current_index] = torch.tensor(terminate, dtype=torch.float32, device=self.device)

        self.current_index = (self.current_index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        print(self.size, batch_size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], self.terminates[indices]
