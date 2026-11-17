import sys
from argparse import Namespace
from typing import Callable, Dict, Optional, Union

import pickle
import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import ImitationPolicy

from src.logger import WandBLogger
from src.utils import threshold_linear_fn


def BCTrainer(
    policy: ImitationPolicy,
    demo_buffer: ReplayBuffer,
    cfg: Union[Namespace, DictConfig],
    dynamics_model=None,
    demos: Batch = None,
    logger: Optional[WandBLogger] = None,
    checkpoint_fn: Optional[Callable[[None], None]] = None,
) -> Dict[str, Union[float, str]]:

    timestamp = {
        "epoch": 0,
        "policy_grad_step": 0,
        "dynamics_epoch": 0,
        "dynamics_grad_step": 0,
    }
    
    # Main training loop
    for epoch in tqdm.trange(
        timestamp["epoch"],        # */600
        cfg.policy.train.n_epoch,  # 600
        desc="Epoch",
        file=sys.stdout,
        disable=cfg.policy.train.step_per_epoch >= 1000,
    ):
        # Dynamics Model
        model_buffer = demos

        # # demos.shape = (1000000, 1, 39)
        # model_buffer = ReplayBuffer(size=np.prod(demos.obs.shape[:2]))  # size = 1000000 * 1
        # for n in range(traces.obs.shape[1]):      # 1000000
        #     for t in range(traces.obs.shape[0]):  # 1
        #         batch = Batch(
        #             obs=demos.obs[t, n],
        #             act=demos.act[t, n],
        #             # rew=demos.rew[t, n],
        #             # done=demos.done[t, n],
        #             # obs_next=demos.obs_next[t, n],
        #         )
        #         model_buffer.add(batch)

        # Update policy
        pbar = tqdm.tqdm(
            total=cfg.policy.train.step_per_epoch,
            desc=f"Epoch #{epoch}",
            file=sys.stdout,  # print tqdm to stdout
            disable=cfg.policy.train.step_per_epoch < 1000,
        )
        iters = 0
        policy.train()
        with pbar:
            while iters < pbar.total:
                if cfg.demonstration.mode == "stitching" :
                    train_result = policy.update(
                        cfg.policy.batch_size, demo_buffer, model_buffer)
                else:
                    train_result = policy.update(
                        cfg.policy.batch_size, demo_buffer)

                # Update timestamps
                pbar.update(1)
                iters += 1
                timestamp["policy_grad_step"] += 1

                logger.log_mean("train", train_result, timestamp)
                pbar.set_postfix(
                    loss=f"{train_result['loss']:.3f}",
                )
        policy.eval()

        # Save policy
        logger.save(timestamp, checkpoint_fn, is_best=False)

        timestamp["epoch"] += 1

    return timestamp
