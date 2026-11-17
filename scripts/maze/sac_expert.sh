#!/bin/bash

DEVICE=cuda:1
# Possible values=
# GoalPointRegionUMaze-v2
# GoalPointWallRoom5x11-v1
# GoalPointCorridor7x7-v2
# GoalAntRegionUMaze-v2
# GoalAntWallRoom5x11-v1
# GoalAntCorridor7x7-v2
ENV=GoalAntWallRoom5x11-v1
CKPT=""

# Setup
# pip install -e .
wandb login "$WANDB_API_KEY"

CMD=(python -u experiments/sac_expert.py
  +experiment=sac_expert/goalenv
  device="$DEVICE"
  env.id="$ENV"
  policy.checkpoint_path="$CKPT"
  logger.wandb.project='sac_expert'
  logger.wandb.name="${ENV}"
)

echo "[Executing command] ${CMD[*]}"
"${CMD[@]}"
