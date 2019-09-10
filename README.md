# Regularized Anderson Acceleration for Off-Policy Deep Reinforcement Learning

Regularized Anderson Acceleration (RAA) is a general acceleration framework for off-policy deep reinforcement learning. 
The algorithm is based on the paper [Regularized Anderson Acceleration for Off-Policy Deep Reinforcement Learning]
(https://arxiv.org/pdf/1909.03245.pdf) presented at NeurIPS 2019.

This implementation uses [PyTorch](https://github.com/pytorch/pytorch) and Python 3.6.
Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks and [Atari 2600] (https://github.com/openai/atari-py)
in [OpenAI Gym v2](https://github.com/openai/gym). 

### Usage
The paper results can be reproduced exactly by running:

For RAA-DuelingDQN
```
./RAA-DuelingDQN/run_atari.sh
```
For RAA-TD3
```
./RAA-TD3/run_mujoco.sh
```

Hyper-parameters can be modified with different arguments to main.py. We include an implementation of DuelingDQN (./RAA-DuelingDQN/src/dqn.py) for 
easy comparison of hyper-parameters with RAA-DuelingDQN and an implementation of TD3 (./RAA-TD3/src/TD3.py) for 
easy comparison of hyper-parameters with RAA-TD3. 

### Results
Learning curves found in the paper are found under ./learning_curves. 
Some experimental data and saved models are found under ./RAA-DuelingDQN/logs and ./RAA-TD3/logs.
Numerical results can be found in the paper, or from the learning curves.

### Reference
```
@article{wenjie2019regularized,
  title={Regularized Anderson Acceleration for Off-Policy Deep Reinforcement Learning},
  author={Wenjie Shi, Shiji Song, Hui Wu, Ya-Chu Hsu, Cheng Wu, Gao Huang},
  booktitle={Advances In Neural Information Processing Systems},
  year={2019}
}
```


