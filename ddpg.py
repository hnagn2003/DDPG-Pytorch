from models.backbone import Actor, Critic
import torch
from torch.optim import Adam
from utils import OURandomProcess, ReplayBuffer
import argparse
import torch.nn.functional as F
import numpy as np

class DDPG():
    def __init__(self, observation_size, action_size, args) -> None:
        if args.seed > 0:
            self.seed(args.seed)
        self.batch_size = args.batch_size
        self.observation_size = observation_size
        self.action_size= action_size
        self.tau = args.tau # soft update factor
        self.discount = args.discount
        self.reciprocal = 1.0 / args.epsilon
        self.epsilon = 0.9 # prob of exploration rather than exploitation
        self.decay_epsilon = 1e-5
        self.last_state = None
        self.last_action = None
        self.is_training = True
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w': 3e-3
        }
        self.actor = Actor(self.observation_size, self.action_size, **net_cfg)
        self.target_actor = Actor(self.observation_size, self.action_size, **net_cfg)
        self.actor_optimizer  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.observation_size, self.action_size, **net_cfg)
        self.target_critic = Critic(self.observation_size, self.action_size, **net_cfg)
        self.critic_optimizer  = Adam(self.critic.parameters(), lr=args.rate)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        
        # experience buffer
        self.buffer_size = self.batch_size*100
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.observation_size, self.action_size)
        self.action_choice = OURandomProcess(size=self.action_size, theta=0.15)
        # TODO normalize state
        # TODO (optionals) cuda, load save weights,...
    def get_state(observation): # 
        # TODO giai ptvp 16-20, return state
        return {'observation': observation, 'pitch_rate': None, 'pitch_angle': None}
    
    def soft_update(tau, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
            
    def update_policy(self):
        # sample batch
        print(self.replay_buffer.sample(self.batch_size))
        observation, action, reward, next_observation, terminate = self.replay_buffer.sample(self.batch_size)
        state = self.get_state(observation)
        next_state = self.get_state(next_observation)
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            next_q_values = self.target_critic(next_state, next_action)
        target_q_value =  reward + self.discount*next_q_values
        
        # update critic
        self.critic.zero_grad()
        pred_q_value = self.critic(state, action)
        td_loss = F.mse_loss(pred_q_value, target_q_value)
        td_loss.backward()
        self.critic_optimizer.step()
        # update actor
        self.actor.zero_grad()
        policy_loss = -self.critic(state, self.actor(state)).mean()
        policy_loss.backward()
        self.actor_optimizer.step()
        # soft update target
        self.soft_update(self.tau, self.target_actor, self.actor)
        self.soft_update(self.tau, self.target_critic, self.critic)
        
    # def eval(self):
    #     self.actor.eval()
    #     self.target_actor.eval()
    #     self.critic.eval()
    #     self.target_critic.eval()
        
    def go_random(self):
        action = np.random.uniform(-1.,1.,self.action_size)
        self.last_action = action
        return action
    
    def select_action(self, last_state, decay_epsilon=True):
        state_tensor = torch.tensor(np.array([last_state]), dtype=torch.float32)
        action = self.actor(state_tensor).squeeze().detach().numpy()
        if self.is_training:
            epsilon_decay = max(self.epsilon, 0)
            action += epsilon_decay * self.random_process.sample()
        action = np.clip(action, -1.0, 1.0) # to range -1 1
        if decay_epsilon:
            self.epsilon -= self.decay_epsilon
        action += epsilon_decay * self.random_process.sample()
        self.last_action = action
        return action
    
    def reset(self, obs):
        self.last_state = obs
        self.random_process.reset_states()
#test 
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=-1, type=int, help='')
parser.add_argument('--batch_size', default=32, type=int, help='')
parser.add_argument('--tau', default=0.001, type=float, help='')
parser.add_argument('--discount', default=0.01, type=float, help='')
parser.add_argument('--epsilon', default=1.0, type=float, help='')
parser.add_argument('--hidden1', default=64, type=int, help='')
parser.add_argument('--hidden2', default=64, type=int, help='')
parser.add_argument('--prate', default=0.001, type=float, help='')
parser.add_argument('--rate', default=0.001, type=float, help='')
args = parser.parse_args()


ddpg = DDPG(4, 4, args)
for episode in range(50):
    # perform actions in the env and get response state, action, reward, next_observation, terminate
    ddpg.replay_buffer.add(1, 2, 3, 4, 5)
ddpg.update_policy()