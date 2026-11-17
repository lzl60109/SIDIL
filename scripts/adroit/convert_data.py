import argparse
import os
import gym
from copy import deepcopy
import numpy as np
import pickle
import d4rl

def read_npz(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, (list, dict, tuple)):
        res = data
    elif isinstance(data, np.lib.npyio.NpzFile):
        res = dict(data)
        data.close()
    else:
        print("incompatible type of data to unzip...")
        res = None
    return res


def argsparser():
    parser = argparse.ArgumentParser("replay trajectories")
    parser.add_argument('--env_id', help='environment ID', default='door-v0')
    parser.add_argument('--expert_path', type=str, help='path of the current trajectories',
                        default="/mnt/gdut/sam_zgm/LZLiang/icra_2025/data/demos/adroit/AdroitDoor-v0/door-v0_demos.pickle")
    parser.add_argument('--output', help='output filename of converted trajectories', 
                        default='/mnt/gdut/sam_zgm/LZLiang/icra_2025/data/demos/adroit/AdroitDoor-v0/door-v0_demos_new.pickle')
    return parser.parse_args()


def convert_dataset(env_id, fname):
    """
    Convert the dataset in a compatible form according to the openai datasets
    :param env_id:
    environment the dataset is run onto
    :param fname:
    path to the dataset file
    :return:
    the dataset dictionary with the converted dataset
    """

    data = read_npz(fname)
    if not data:
        # is emtpy
        print("dataset is empty")
        return data
    if isinstance(data, dict) and 'obs' in data and 'acs' in data:
        # data is in correct format
        return data
    elif isinstance(data, list) and isinstance(data[0], dict) and 'observations' in data[0] and 'actions' in data[0] \
            and 'rewards' in data[0]:
        res = {'obs': [], 'act': [], 'rew': [], 'done': [], 'obs_next': [], 'init_state_dict':[]}
        env = gym.make(env_id)
        for traj in data:
            env.reset()
            if 'init_state_dict' in traj:
                env.set_env_state(traj['init_state_dict'])

            res['obs'].append(traj['observations'])
            res['act'].append(traj['actions'])
            res['init_state_dict'].append(traj['init_state_dict'])

            actions = traj['actions']
            next_obs = []
            rews = []
            do = []
            for t in range(actions.shape[0]):
                observation, reward, done, info = env.step(actions[t])
                next_obs.append(observation)
                rews.append(reward)
                do.append(done)
            res['rew'].append(np.array(rews))
            res['done'].append(np.array(do))
            res['obs_next'].append(np.array(next_obs))

        return res
    else:
        print("Unexpected type of data")
        return data

            


def main(args):
    data = convert_dataset(args.env_id, args.expert_path)
    if data:
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    args = argsparser()
    main(args)
