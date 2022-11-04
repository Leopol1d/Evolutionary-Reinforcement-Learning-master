# TIME:2022/11/2 14:18
# DEVELOPER:LEOPOLD

import sys
sys.path.append("..")

from core import my_runner
import highway_env


from single_agent.Model_common import ActorNetwork, CriticNetwork
import torch
from single_agent.Memory_common import OnPolicyReplayMemory
import gym
import numpy as np
# import highway_env
import random
env = gym.make('merge-multi-agent-v0')
n_agent = len(env.controlled_vehicles)
memory = OnPolicyReplayMemory(10000)

state, _ = env.reset()

while True:
    actions = [0] * n_agent
    for agent_id in range(n_agent):
        actions[agent_id] = np.random.choice(env.n_a)
    next_state, global_reward, done, info = env.step(tuple(actions))

    reward = list(info["regional_rewards"])


    memory.push(state, actions, reward, next_state, done)
    # print('memory: ', memory.memory)
    # next_state = torch.FloatTensor(next_state)
    state = next_state
    if memory.__len__() >= 10:
        batch = memory.sample(3)
        # print('batch: ', batch)

    print(memory.__len__())
