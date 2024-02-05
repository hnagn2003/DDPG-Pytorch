import argparse
from env.env import UAVEnv
from copy import deepcopy
from ddpg import DDPG
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
parser.add_argument('--num_of_explorations', default=20, type=int, help='')

def train(args, num_iters, agent, env, max_episode_length=None):
    agent.is_training = True
    iter = 0
    episode = 0
    episode_steps = 0
    episode_reward = 0
    observation = None
    while iter < num_iters:
        #reset env each start of episode
        print("----------------iter ", iter, "------------------------------")
        # if observation is None:
        #     observation = deepcopy(env.reset())
        #     agent.reset(observation)
            
        # first, agent explore in a fixed of steps
        if iter <= args.num_of_explorations:
            action = agent.go_random()
        # then follow the policies
        else:
            action = agent.select_action(observation)
        
        #env response
        observation, next_observation, reward, terminate = env.step(action)
        next_observation = deepcopy(next_observation)
        if max_episode_length and episode_steps >= max_episode_length -1:
            terminate = True
        # TODO store transition
        agent.replay_buffer.add(observation, next_observation, action, reward, terminate)        

        # agent update policy
        if iter > args.warmup :
            agent.update_policy()
        iter += 1
        episode_steps += 1
        episode_reward += reward
        agent.replay_buffer.add()
        #end of episode
        if terminate:

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0
            episode += 1
num_iters = 10
agent = DDPG()
env = UAVEnv()
train(parser.parse_args(), num_iters=num_iters, agent=agent, env=env)