import os, sys
import copy
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import cv2
from autolab_core import BinaryClassificationResult

from core.components.base_model import BaseModel
from core.modules.losses import L2Loss, KLDivLoss, BCELogitsLoss
from core.modules.subnetworks import Encoder, Decoder, Predictor
from core.utils.general_utils import (
    AttrDict,
    ParamDict,
    remove_spatial,
    get_clipped_optimizer,
    GetIntermediatesSequential,
)
from core.utils.pytorch_utils import TensorModule, RAdam, ten2ar, ar2ten, map2torch
from core.modules.variational_inference import (
    ProbabilisticModel,
    Gaussian,
    get_fixed_prior,
)
from core.components.checkpointer import (
    CheckpointHandler,
    save_cmd,
    save_git,
    get_config_path,
)
from core.modules.layers import LayerBuilderParams
from core.utils.vis_utils import make_image_strip


class VAE(BaseModel, ProbabilisticModel):
    """Simple recurrent forward predictor network with image encoder and decoder."""

    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self.device = self._hp.device if "device" in self._hp else None
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(
            self._hp.use_convs, self._hp.normalization
        )

        # set up beta tuning (use fixed beta by default)
        if self._hp.target_kl is None:
            self._log_beta = TensorModule(
                np.log(self._hp.fixed_beta)
                * torch.ones(1, requires_grad=False, device=self._hp.device)
            )
        else:
            self._log_beta = TensorModule(
                torch.zeros(1, requires_grad=True, device=self._hp.device)
            )
            self.beta_opt = self._get_beta_opt()

        self.build_network()

    @contextmanager
    def val_mode(self):
        # self.switch_to_prior()
        yield
        self.switch_to_inference()

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = ParamDict(
            {
                "use_skips": False,
                "skips_stride": 1,
                "add_weighted_pixel_copy": False,  # if True, adds pixel copying stream for decoder
                "pixel_shift_decoder": False,
                "use_convs": True,
                "normalization": "batch",
                "use_wide_img": False,
            }
        )

        # Network size
        default_dict.update(
            {
                "img_sz": 32,  # image resolution
                "img_width": 64,  # used only when use_wide_img is True
                "img_height": 64,  # used only when use_wide_img is True
                "input_nc": 3,  # number of input feature maps
                "ngf": 128,  # number of feature maps in shallowest level
                "nz_enc": 32,  # number of dimensions in encoder-latent space
                "nz_vae": 32,  # number of dimensions in vae-latent space
                "nz_mid": 32,  # number of dimensions for internal feature spaces
                "n_processing_layers": 3,  # Number of layers in MLPs
            }
        )

        # Loss weights
        default_dict.update(
            {
                "img_mse_weight": 1.0,
                "fixed_beta": 1.0,
                "target_kl": None,
            }
        )

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        self.encoder = Encoder(self._hp)
        self.enc2z = torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        self.z2enc = torch.nn.Linear(self._hp.nz_vae, self._hp.nz_enc)
        self.decoder = Decoder(self._hp)

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        # encode
        enc_input = self.encoder(inputs.images[:, 0])

        # sample z
        output.q = Gaussian(self.enc2z(remove_spatial(enc_input)))
        if self._sample_prior:
            output.z = get_fixed_prior(output.q).sample()
        else:
            output.z = output.q.sample()

        # decode
        output.output_imgs = self.decoder(
            self.z2enc(output.z).view(enc_input.shape)
        ).images
        output.output_imgs = output.output_imgs[
            :, None
        ]  # make it sequence to make downstream usage easier
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        losses.img_mse = L2Loss(self._hp.img_mse_weight)(
            model_output.output_imgs[:, 0], inputs.images[:, 0]
        )

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta[0].detach())(
            model_output.q, get_fixed_prior(model_output.q)
        )

        # Update Beta
        if self.training:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        self._logger.log_scalar(self.beta[0], "beta", step, phase)
        if log_images:
            # log reconstructions / prior samples
            ch = inputs.images.shape[2]
            for i in range(int(ch // 3)):
                img_strip = make_image_strip(
                    [
                        inputs.images[:, 0, i * 3 : (i + 1) * 3],
                        model_output.output_imgs[:, 0, i * 3 : (i + 1) * 3],
                    ]
                )
                self._logger.log_images(
                    img_strip[None], "generation_{}".format(i), step, phase
                )

    def _get_beta_opt(self):
        return get_clipped_optimizer(
            filter(lambda p: p.requires_grad, self._log_beta.parameters()),
            lr=3e-4,
            optimizer_type=RAdam,
            betas=(0.9, 0.999),
            gradient_clip=None,
        )

    def _update_beta(self, kl_div):
        """Updates beta with dual gradient descent."""
        if self._hp.target_kl is not None:
            beta_loss = self.beta * (self._hp.target_kl - kl_div).detach().mean()
            self.beta_opt.zero_grad()
            beta_loss.backward()
            self.beta_opt.step()

    def forward_encoder(self, inputs):
        enc = self.encoder(inputs)
        q = Gaussian(self.enc2z(remove_spatial(enc)))
        return q.mu

    @property
    def resolution(self):
        return self._hp.img_sz

    @property
    def beta(self):
        return self._log_beta().exp()
