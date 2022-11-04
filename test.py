import numpy

import highway_env
import numpy as np
import gym
import torch
env = gym.make('merge-multi-agent-v0')
import matplotlib.pyplot as plt

mix_rate = [0.5, 0.5]
print(len(mix_rate))


class A:
    def __init__(self, a):
        self.a = a
        # self.learner.actor = a ** 2
if __name__ == '__main__':
    x = A(2)
    print(x.a)