import os
import torch
import torch.optim as optim
import argparse
from collections import namedtuple

from src.model import Dueling_DQN
from src import dqn, raa_dqn
from utils.atari_wrappers import *
from utils.gym_setup import *
from utils.schedules import *

# Global Variables
# Extended data table 1 of nature paper
BATCH_SIZE = 32
SAMPLE_SIZE = 128
REPLAY_BUFFER_SIZE = 1000000
FRAME_HISTORY_LEN = 4
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)
LEARNING_STARTS = 50000


def atari_learn(args):
    OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
    optimizer = OptimizerSpec(constructor=optim.RMSprop,
                              kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS))

    save_path = "logs/{}/{}/seed-{}".format(args.env_name, args.agent_name, args.seed)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    env = get_env(args.env_name, args.seed, save_path)

    if args.agent_name == 'DuelingDQN_RAA':
        raa_dqn.dqn_learning(
            env=env,
            q_func=Dueling_DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            max_steps=args.max_steps,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            sample_size=SAMPLE_SIZE,
            gamma=GAMMA,
            beta=args.beta,
            reg_scale=args.reg_scale,
            use_restart=args.use_restart,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=args.target_update_freq,
            save_path=save_path
        )
    else:
        dqn.dqn_learning(
            env=env,
            q_func=Dueling_DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            max_steps=args.max_steps,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=args.target_update_freq,
            save_path=save_path
        )
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL agents for atari')
    parser.add_argument("--env_name", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--agent_name", default="DuelingDQN", help="DuelingDQN, DuelingDQN_RAA")
    parser.add_argument("--seed", type=int, default=123, help="seed for initialization")
    parser.add_argument("--gpu", type=int, default=0, help="ID of GPU to be used")
    parser.add_argument("--beta", type=float, default=0.05, help="Coefficient for progressive update")
    parser.add_argument("--reg_scale", type=float, default=0.1, help="Scale of regularization for anderson acceleration")
    parser.add_argument("--max_steps", type=float, default=20e6, help="Max time steps to run environment")
    parser.add_argument("--use_restart", action="store_true", help="Whether to use the adaptvie restart strategy")
    parser.add_argument("--target_update_freq", type=int, default=10000, help="frequency to update target network")
    args = parser.parse_args()

    # command
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Run training
    print("----------------------------------------------")
    print("Training on %s with %s" % (args.env_name, args.agent_name))
    print("----------------------------------------------")

    atari_learn(args)

