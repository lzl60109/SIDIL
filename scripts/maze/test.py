import gym
import mujoco_maze
import pickle
import numpy as np
from src.stitching.utils import read_npz
from src.envs.utils import make_env
from tianshou.data import ReplayBuffer, Batch
from tianshou.env import DummyVectorEnv

file = "data/demos/maze/GoalAntWallRoom5x11-v1/GoalAntWallRoom5x11_aug_trajectories_20_200_0.9.pkl"
file1 = '/mnt/gdut/sam_zgm/LZLiang/icra_2025/data/demos/adroit/AdroitDoor-v0/AdroitDoor_aug_trajectories_20_200_0.9.pkl'
file2 = '/mnt/gdut/sam_zgm/LZLiang/icra_2025/data/demos/maze/GoalPointWallRoom5x11-v1/20episodes_1.pkl'

data = pickle.load(open(file, "rb"))
data1 = pickle.load(open(file1, "rb"))
data2 = pickle.load(open(file2, "rb"))
# print(len(data2['obs'][0]))

print(data.keys())