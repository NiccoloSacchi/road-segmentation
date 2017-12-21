import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

class Sampler(object):

    def __init__(self,
                 datasets,
                 minibatch_size,
                 num_input_channels):

        self.minibatch_size = minibatch_size
        self.num_input_channels = num_input_channels
        self.num_samples = len(datasets[0])
        self.iters_per_epoch = self.num_samples // self.minibatch_size
        self.datasets = datasets

        self._indices = None
        self._indices_epoch = None

    def __getitem__(self, index):
        return self.get_minibatch(index)

    def get_minibatch(self, index):

        epoch, i = divmod(index, self.iters_per_epoch)

        self._shuffle(epoch)

        minibatch_elements = self._indices[i * self.minibatch_size : (i + 1) * self.minibatch_size]

        # Required for H5 compatibility.
        minibatch = tuple(np.array([d[e] for e in minibatch_elements]) for d in self.datasets)

        return minibatch

    def shuffle(self, index, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = np.random.randint(i + 1)
            lst[i], lst[j] = lst[j], lst[i]

        return lst

    def _shuffle(self, epoch):

        if self._indices_epoch == epoch:
            return

        self._indices = np.arange(self.num_samples)
        self.shuffle(epoch, self._indices)
        self._indices_epoch = epoch
