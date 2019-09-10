import gym
from gym import wrappers
import numpy as np
import random
from utils.atari_wrappers import *

def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i) 
    np.random.seed(i)
    random.seed(i)


def get_env(env_name, seed, save_path):
    env = gym.make(env_name)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '%s/tmp' % save_path
    env = wrappers.Monitor(env, expt_dir, video_callable=False, force=True)
    env = wrap_deepmind(env)

    return env


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
