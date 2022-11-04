from EMADQN import My_AL
from single_agent.utils_common import agg_double_list

import sys
sys.path.append("../highway-env")


import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env




if __name__ == '__main__':
    MAX_EPISODES = 200  # 20000
    EPISODES_BEFORE_TRAIN = 10
    EVAL_EPISODES = 3
    EVAL_INTERVAL = 10  # 200

    # max steps in each episode, prevent from running too   long
    MAX_STEPS = 100

    MEMORY_CAPACITY = 10000  # 1000000
    BATCH_SIZE = 128
    CRITIC_LOSS = "mse"
    MAX_GRAD_NORM = None

    REWARD_DISCOUNTED_GAMMA = 0.99
    EPSILON_START = 0.99
    EPSILON_END = 0.05
    EPSILON_DECAY = 200  # 50000
    env = gym.make('merge-multi-agent-v0')
    # env_eval = gym.make('merge-multi-agent-v0')
    state_dim = env.n_s
    action_dim = env.n_a

    my_al = My_AL(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_dim, action_dim=action_dim,
              batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN)
    my_al.train(200)
