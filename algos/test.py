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

    actor_hidden_size = 3
    # actor1 = ActorNetwork(5, actor_hidden_size,2, nn.functional.log_softmax)
    # num_params = len(list(actor1.parameters()))
    # actor1_param = actor1.get_params()
    # print(actor1_param)
    # actor2 = ActorNetwork(5, actor_hidden_size,2, nn.functional.log_softmax)
    # actor2.set_params(actor1_param)
    # for p1, p2 in zip(actor1.parameters(), actor2.parameters()):
    #     print(p1.data == p2.data)
    #     print('p1.data: ', p1.data)
    #     print('p2.data: ', p2.data)
    # pop = []
    # temp = []
    # for _ in range(5):
    #     pop.append(ActorNetwork(5, actor_hidden_size, 2, nn.functional.log_softmax).cuda())
    #     temp.append(ActorNetwork(5, actor_hidden_size, 2, nn.functional.log_softmax).cuda())
    #
    # # print(pop[4])
    #
    # # for i in pop[-2:]:
    # #     for param in i.parameters():
    # #         print(param)
    # #     print('--------------------------------')
    #
    # es_params = []
    # for actor in pop:
    #     es_params.append(actor.get_params())
    #
    # for i, param in enumerate(es_params):
    #     temp[i].set_params(param)
    # for p, t in zip(pop, temp):
    #     for p1, p2 in zip(p.parameters(), t.parameters()):
    #         print(p1 == p2)
    # a = ActorNetwork(5, 128, 2, nn.functional.log_softmax)
    # print(a.get_size())