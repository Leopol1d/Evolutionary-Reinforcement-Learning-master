from EMAPPO import My_AL
from mappo import MAPPO
from single_agent.utils_common import agg_double_list

import sys
sys.path.append("..")

import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env
import argparse
import configparser
import os
from datetime import datetime
from single_agent.utils import agg_double_list, copy_file_ppo, init_dir
a = 9
rewards_mu = True if a > 0 else False
a = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580]
print(a[4])
