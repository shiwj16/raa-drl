#!/bin/bash

# Script to reproduce results

for ((i=101;i<108;i+=1))
do
  nohup python main.py \
  --policy_name="RAA-TD3" \
  --env_name="Walker2d-v2" \
  --seed=$i \
  --save_models \
  --use_restart \
  --num=5 \
  --beta=0.1 \
  --reg_scale=0.001 \
  --max_steps=1000000 \

  nohup python main.py \
  --policy_name="RAA-TD3" \
  --env_name="Ant-v2" \
  --seed=$i \
  --save_models \
  --use_restart \
  --num=5 \
  --beta=0.1 \
  --reg_scale=0.001 \
  --max_steps=1000000 \
  
  nohup python main.py \
  --policy_name="RAA-TD3" \
  --env_name="Hopper-v2" \
  --seed=$i \
  --save_models \
  --use_restart \
  --num=5 \
  --beta=0.1 \
  --reg_scale=0.001 \
  --max_steps=1000000 \
  
  nohup python main.py \
  --policy_name="RAA-TD3" \
  --env_name="HalfCheetah-v2" \
  --seed=$i \
  --save_models \
  --use_restart \
  --num=5 \
  --beta=0.1 \
  --reg_scale=0.001 \
  --max_steps=1000000 \
done

