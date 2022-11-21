import numpy

# import highway_env
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt

# ends = [400, 200, 100, 200]
# # print(sum(ends[:3]))
# Merging_lane_cost_100 = - np.exp(-(100 - 100) ** 2 / ( # r_m
#                      10* ends[2]))
# Merging_lane_cost_0 = - np.exp(-(0 - 100) ** 2 / ( # r_m
#                      10* ends[2]))
# print(Merging_lane_cost_100)
# print(Merging_lane_cost_0)
a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.mean(a, axis=1))