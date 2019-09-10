#!/bin/bash

# Script to reproduce results


for ((i=101;i<106;i+=1))
do
  python main.py \
  --env_name="QbertNoFrameskip-v4" \
  --agent_name="DuelingDQN_RAA" \
  --seed=$i \
  --gpu=5 \
  --beta=0.05 \
  --use_restart \
  --reg_scale=0.1 \
  --target_update_freq=2000 \
  --max_steps=15000000

  python main.py \
  --env_name="SpaceInvadersNoFrameskip-v4" \
  --agent_name="DuelingDQN_RAA" \
  --seed=$i \
  --gpu=5 \
  --beta=0.05 \
  --use_restart \
  --reg_scale=0.1 \
  --target_update_freq=2000 \
  --max_steps=15000000
  
  python main.py \
  --env_name="BreakoutNoFrameskip-v4" \
  --agent_name="DuelingDQN_RAA" \
  --seed=$i \
  --gpu=5 \
  --beta=0.05 \
  --use_restart \
  --reg_scale=0.1 \
  --target_update_freq=2000 \
  --max_steps=30000000
  
  python main.py \
  --env_name="EnduroNoFrameskip-v4" \
  --agent_name="DuelingDQN_RAA" \
  --seed=$i \
  --gpu=5 \
  --beta=0.05 \
  --use_restart \
  --reg_scale=0.1 \
  --target_update_freq=2000 \
  --max_steps=15000000
done
