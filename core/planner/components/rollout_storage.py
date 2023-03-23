import numpy as np
import gzip
import pickle
import h5py
import os
import copy
from collections import deque

from core.utils.general_utils import (
    AttrDict,
    RecursiveAverageMeter,
    ParamDict,
    listdict2dictlist,
)


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""

    def __init__(self):
        self.rollouts = []

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts  # rollout storage should not be empty
        stats = RecursiveAverageMeter()
        for rollout in self.rollouts:
            stat = AttrDict(avg_reward=np.stack(rollout.reward).sum())
            info = [list(filter(None, ele)) for ele in rollout.info]
            info = [ele for ele in info if ele]
            if info:
                info = listdict2dictlist([item for sublist in info for item in sublist])
                for key in info:
                    name = "avg_" + key
                    stat[name] = np.array(info[key]).sum()
            stats.update(stat)
        return stats.avg

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]
