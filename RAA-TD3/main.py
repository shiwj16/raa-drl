import numpy as np
import torch
import gym
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import src.replay as replay
import src.TD3 as TD3
import src.RAATD3_v0 as RAATD3
# import src.RAATD3_v1 as RAATD3


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        index = 0
        while not done and index < args.episode_size:
            index += 1
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="Base_AATD3")          # Policy name
    parser.add_argument("--env_name", default="Pendulum-v0")            # OpenAI gym environment name Pendulum-v0
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)     # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e4, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_steps", default=1e6, type=float)         # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")           # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)          # Batch size for both actor and critic
    parser.add_argument("--episode_size", default=1000, type=int)       # Episode size for tracking
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate,default: 0.005
    parser.add_argument("--policy_noise", default=0.2, type=float)      # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
    # for regularized anderson acceleration
    parser.add_argument("--use_restart", action="store_true")           # Whether to use the restart technique
    parser.add_argument("--beta", default=0.1, type=float)              # Coefficient of consistency between actor and critic
    parser.add_argument("--reg_scale", default=0.001, type=float)       # Scale of regularization for anderson acceleration
    parser.add_argument("--num", default=5, type=int)                   # Maximal number for previous critics
    parser.add_argument("--aa_batch", default=400, type=int)            # The batch size for calculating the weights in AA

    args = parser.parse_args()

    data_path = "./logs/{}/{}-num={}/seed-{}/data".format(args.env_name, args.policy_name, args.num, args.seed)
    model_path = "./logs/{}/{}-num={}/seed-{}/model".format(args.env_name, args.policy_name, args.num, args.seed)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if args.save_models and not os.path.exists(model_path):
        os.makedirs(model_path)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "RAA-TD3":
        policy = RAATD3.RAATD3(state_dim, action_dim, max_action, args.use_restart, 
                                    args.beta, args.reg_scale, args.num, args.aa_batch)
    elif args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    else:
        raise NotImplementedError

    replay_buffer = replay.ReplayBuffer()
    
    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]

    global_step = 0
    steps_since_eval = 0
    episode_num = 0
    done = True
    residuals = []

    while global_step < args.max_steps:
        if done or epi_step==args.episode_size:
            if global_step != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") %
                      (global_step, episode_num, epi_step, epi_reward))
                residual = policy.train(replay_buffer, epi_step, args.batch_size, args.discount, args.tau,
                                        args.policy_noise, args.noise_clip, args.policy_freq)

            # Evaluate episode
            if steps_since_eval >= args.eval_freq:
                steps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))
                residuals.append(residual)
                np.save(os.path.join(data_path, 'eval.npy'), evaluations)
                np.save(os.path.join(data_path, 'residual.npy'), residuals)

                if args.save_models: policy.save(model_path)

            # Reset environment
            obs = env.reset()
            done = False
            epi_reward = 0
            epi_step = 0
            episode_num += 1

        # Select action randomly or according to policy
        if global_step < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0: 
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if epi_step + 1 == args.episode_size else float(done) # env._max_episode_steps
        epi_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        epi_step += 1
        global_step += 1
        steps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy))
    if args.save_models: policy.save(model_path)
    np.save(os.path.join(data_path, 'eval.npy'), evaluations)
