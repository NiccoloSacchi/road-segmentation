import logging
import numpy as np

from . import network, trainer, samplers
from .transformation import all_transformations, rotation, crop_central

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.optim as optim

logger = logging.getLogger(__name__)

def my_setup(unet,
                training_x, training_y,
                num_input_channels = 3,
                batch_size = 1,
                loss_weights=None,
                learning_rate=1e-4,
                save_path=None,
                save_every=None):

    ndims           = unet.config.ndims
    num_channels    = unet.config.num_input_channels
    num_classes     = unet.config.num_classes

    if loss_weights is None:
        loss_weights = [np.ones(training_y[0].shape) for ty_i in training_y]

    # Sampler, solver and trainer
    logger.info("Creating sampler...")

    datasets  = [training_x, training_y, loss_weights]
    sampler   = samplers.Sampler(datasets,
                                 minibatch_size=batch_size,num_input_channels=num_channels)

    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

    def batch_transform(x_b,y_b,w_b):

        x = np.transpose(x_b, (0, 3, 1, 2))

        d_rand = np.random.randint(360)
        h_rand = np.random.random()*2
        l_rand = np.random.random()*2

        x = rotation(x,d_rand)
        y = rotation(y_b,d_rand)
        w = rotation(w_b,d_rand)

        x = crop_central(x,h_rand,l_rand)
        y = crop_central(y_b,h_rand,l_rand)
        w = crop_central(w_b,h_rand,l_rand)

        x = np.float32(x)
        y = np.int64(y)
        w = np.float32(w)

        return x,y,w

    def training_step(niter, sampler, unet, optimizer):

        # Get the minibatch
        x, y, w = sampler.get_minibatch(niter)
        x, y, w = batch_transform(x, y, w)

        # ignore the label equal to 2
        mask = y == 2
        y[mask] = 0
        w[mask] = 0

        # Convert to pytorch
        x2 = Variable(torch.from_numpy(np.ascontiguousarray(x)).cuda())
        y2 = Variable(torch.from_numpy(np.ascontiguousarray(y)).cuda())
        w2 = Variable(torch.from_numpy(np.ascontiguousarray(w)).cuda())

        optimizer.zero_grad()
        pred = unet(x2)

        loss = unet.unet_cross_entropy_labels(pred, y2)
        loss.backward()

        optimizer.step()

        return {"loss": float(loss.data.cpu().numpy())}

    logger.info("Creating trainer...")
    unet_trainer = trainer.Trainer(lambda niter : training_step(niter, sampler, unet, optimizer),
                                   save_every=save_every or sampler.iters_per_epoch,
                                   save_path=save_path,
                                   managed_objects=trainer.managed_objects({"network": unet,
                                                                            "optimizer": optimizer}),
                                   test_function=None,
                                   test_every=None)

    return unet_trainer, sampler, optimizer
