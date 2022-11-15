from copy import deepcopy

import torch

from EMAPPO import My_AL
from mappo import MAPPO
from single_agent.utils_common import agg_double_list

import sys
sys.path.append("..")

import gym
import numpy as np
import matplotlib.pyplot as plt
# import highway_env
import argparse
import configparser
import os
from datetime import datetime
from single_agent.utils import agg_double_list, copy_file_ppo, init_dir
import time
from single_agent.Model_common import ActorNetwork, CriticNetwork
from torch import nn


def get_params(actor):
    """
    Returns parameters of the actor
    """
    return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                               actor.parameters()]))

def to_numpy(var):
    return  var.data.numpy()


def get_size(actor):

    return get_params(actor).shape[0]

if __name__ == '__main__':

    # actor_hidden_size = 3
    # actor1 = ActorNetwork(5, actor_hidden_size,2, nn.functional.log_softmax)
    # num_params = len(list(actor1.parameters()))
    # print('num_params: ', num_params)
    # print(get_size(actor1))
    # print(get_params(actor1))
    # print('----------------------------------------------------')
    # for i, param in enumerate(actor1.parameters()):
    #     print(i, 'param.data: ', param.data)
    #     print(i, param.shape)
    # parents = 10
    # weights = np.array([np.log((parents + 1) / i)
    #                          for i in range(1, parents + 1)])
    # print(weights)
    # weights /= weights.sum()
    # print(weights)
    num_params = 38
    sigma = 1e-3
    cov = sigma * np.ones(num_params)
    print('cov: ', cov)
    epsilon_half = np.random.randn(2, 38)
    epsilon = np.concatenate([epsilon_half, - epsilon_half])
    print(epsilon)
    inds = epsilon * np.sqrt(cov)
    print(inds.shape)
    new_damp = 1e-3 * 0.95 + 0.05 * 1e-5
    print(new_damp)




