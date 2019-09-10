import torch
import sys
import gym.spaces
import itertools
import numpy as np
import random
from utils.replay_buffer import *
from utils.schedules import *
from utils.gym_setup import *
from src.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dqn_learning(env,
                 q_func,
                 optimizer_spec,
                 exploration=LinearSchedule(1000000, 0.1),
                 max_steps=20e6,
                 replay_buffer_size=1000000,
                 batch_size=32,
                 gamma=0.99,
                 learning_starts=50000,
                 learning_freq=4,
                 frame_history_len=4,
                 target_update_freq=10000,
                 save_path=None):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    max_steps: float
        Maximal steps.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    # Set the logger
    logger = Logger(save_path)

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
        in_channels = input_shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
        in_channels = input_shape[2]
    num_actions = env.action_space.n
    
    # define Q target and Q
    Q = q_func(in_channels, num_actions).to(device)
    Q_target = q_func(in_channels, num_actions).to(device)

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ######

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    SAVE_MODEL_EVERY_N_STEPS = 100000
    saved_scalars = []
    stop = False
    clipped_error = torch.FloatTensor([0]).to(device)

    for t in itertools.count():
        # 1. Step the env and store the transition
        # store last frame, returned idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observations to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()  # torch

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = observations.unsqueeze(0) / 255.0
                with torch.no_grad():
                    q_value_all_actions = Q(obs)
                action = (q_value_all_actions.data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]

        obs, reward, done, info = env.step(action)

        # clipping the reward, noted in nature paper
        reward = np.clip(reward, -1.0, 1.0)

        # store effect of action
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = env.reset()

        # update last_obs
        last_obs = obs

        # 2. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_t = obs_t / 255.0
            act_t = torch.LongTensor(act_t).to(device)
            rew_t = torch.FloatTensor(rew_t).to(device)
            obs_tp1 = obs_tp1 / 255.0
            done_mask = done_mask

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            q_values = Q(obs_t)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            # get the Q values for best actions in obs_tp1 
            # based off frozen Q network
            # max(Q(s', a', theta_i_frozen)) wrt a'
            q_tp1_values = Q_target(obs_tp1).detach()
            q_s_a_prime, a_prime = q_tp1_values.max(1)

            # if current state is end of episode, then there is no next Q value
            q_s_a_prime = (1 - done_mask) * q_s_a_prime 

            # Compute Bellman error
            # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
            error = rew_t + gamma * q_s_a_prime - q_s_a

            # clip the error and flip
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()
            q_s_a.backward(clipped_error.data)

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

        # 3. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if save_path is not None:
                torch.save(Q.state_dict(), '%s/net.pth' % save_path)

        if t % LOG_EVERY_N_STEPS == 0:
            underlying_env = get_wrapper_by_name(env, "Monitor")
            internal_steps = underlying_env.get_total_steps()
            stop = (internal_steps >= max_steps)
            episode_rewards = underlying_env.get_episode_rewards()
            num_episode = len(episode_rewards)

            if num_episode > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                saved_scalars.append([t, internal_steps, num_episode, mean_episode_reward,
                                      clipped_error.mean().data.cpu().numpy()])
                np.save('%s/scalars.npy' % save_path, saved_scalars)

            print("---------------------------------")
            print("Wrapped - Atari (steps) %d-%d" % (t, internal_steps))
            print("episodes %d" % num_episode)
            print("mean episode reward %f" % mean_episode_reward)
            print("best mean episode reward %f" % best_mean_episode_reward)
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # ============ TensorBoard logging ============#
            info = {'num_episodes': len(episode_rewards),
                    'exploration': exploration.value(t),
                    'mean_episode_reward_last_100': mean_episode_reward
                    }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

        # 4. Check the stop criteria
        if stop:
            break
