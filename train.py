import argparse
from env.env import UAVEnv
import torch
from ddpg import DDPG
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=-1, type=int, help='')


def train(num_iters, agent, env, val_steps, output, max_episode_length):
    iter = 0
    episode = 0
    episode_step = 0
    obervation = None
    while iter < num_iters:
        #reset env each start of episode
    pass
    

num_iters = 10
env = UAVEnv()
train