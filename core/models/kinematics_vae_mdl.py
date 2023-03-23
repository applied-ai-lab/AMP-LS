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
from core.modules.geco_modules import GECO
from core.utils.vis_utils import make_image_strip
from core.utils.transform_utils import ortho6d2mat
from core.models.vae_mdl import VAE
from core.data.scene_collision.src.robot import Robot


class KinematicsVAE(VAE):
    """
    VAE to learn structured latent representation for robot kinematics.
    """

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = ParamDict({"use_convs": False})

        # Network size
        default_dict.update(
            {
                "state_dim": 1,
                "activation_fn": nn.LeakyReLU(0.2, inplace=True),
                "asset_path": "./core/data/scene_collision/assets/",
            }
        )

        # Loss weights
        default_dict.update(
            {
                "state_mse_weight": 1.0,
                "pos_mse_weight": 1.0,
                "rot_mse_weight": 1.0,
                "geco_params": None,
            }
        )

        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def build_network(self):
        """Defines the network architecture"""
        self._hp.builder = LayerBuilderParams(
            self._hp.use_convs,
            self._hp.normalization,
            activation_fn=self._hp.activation_fn,
        )

        self.encoder = Encoder(self._hp)
        self.enc2z = torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        self.z2enc = torch.nn.Linear(self._hp.nz_vae, self._hp.nz_enc)
        self.decoder = Decoder(self._hp)
        self.robot = Robot(
            os.path.join(self._hp.asset_path, "./data/panda/panda.urdf"),
            "panda_link8",
            device=self._hp.device,
        )
        if self._hp.geco_params is not None:
            self.geco = GECO(self._hp.geco_params)

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        # encode
        enc_input = self.encoder(inputs.states)

        # sample z
        output.q = Gaussian(self.enc2z(enc_input))
        if self._sample_prior:
            output.z = get_fixed_prior(output.q).sample()
        else:
            output.z = output.q.sample()

        # decode
        output.decoded_output = self.decoder(self.z2enc(output.z)).images
        return output

    def encode(self, inputs):
        """Runs forward pass of encoder"""
        output = AttrDict()

        # encode
        enc_input = self.encoder(inputs.states)

        # sample z
        output.q = Gaussian(self.enc2z(enc_input))
        if self._sample_prior:
            output.z = get_fixed_prior(output.q).sample()
        else:
            output.z = output.q.sample()
        return output

    def decode(self, z):
        """Runs forward pass of decoder given embedding."""
        return self.decoder(self.z2enc(z)).images

    def loss(self, model_output, inputs):
        """Loss computation."""
        losses = AttrDict()

        # reconstruction loss
        losses.state_mse = L2Loss(self._hp.state_mse_weight)(
            model_output.decoded_output[:, :, :7], inputs.states[:, :, :7]
        )  # (joint + pos)
        losses.pos_mse = L2Loss(self._hp.pos_mse_weight)(
            model_output.decoded_output[:, :, 7:10], inputs.states[:, :, 7:10]
        )  # (joint + pos)
        losses.rot_mse = L2Loss(self._hp.rot_mse_weight)(
            model_output.decoded_output[:, :, 10:], inputs.states[:, :, 10:]
        )  # (joint + pos)

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta[0].detach())(
            model_output.q, get_fixed_prior(model_output.q)
        )

        joint_states = model_output.decoded_output[:, 0, :7]
        joint_states = torch.cat(
            [joint_states, 0.04 * torch.ones_like(joint_states)], dim=-1
        )
        self.robot.set_joint_cfg(joint_states)
        ee_pose = self.robot.ee_pose
        ee_pos = ee_pose[:, :3, 3]
        ee_rot = ee_pose[:, :3, :3]

        # Update Beta
        if self.training:
            self._update_beta(losses.kl_loss.value)

        if self._hp.geco_params is not None:
            state_loss = (
                losses.state_mse.value * losses.state_mse.weight
                + losses.rot_mse.value * losses.rot_mse.weight
                + losses.pos_mse.weight * losses.pos_mse.value
            )
            losses.total = AttrDict(
                value=self.geco.loss(state_loss, losses.kl_loss.value)
            )
        else:
            losses.total = self._compute_total_loss(losses)
        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        """Optionally visualizes outputs.
        :arg model_output: output of the model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg losses: output of the model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        super()._log_losses(losses, step, log_images, phase)
        state_total_loss = (
            losses.state_mse.value * losses.state_mse.weight
            + losses.rot_mse.value * losses.rot_mse.weight
            + losses.pos_mse.weight * losses.pos_mse.value
        )
        self._logger.log_scalar(state_total_loss, "state_total_loss", step, phase)

        if self._hp.geco_params is not None:
            self._logger.log_scalar(self.geco.geco_lambda, "beta", step, phase)
        else:
            self._logger.log_scalar(self.beta[0], "beta", step, phase)

    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)

        if self._hp.geco_params is not None:
            d["geco"] = self.geco.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "geco" in state_dict:
            geco_params = state_dict.pop("geco")
        if self._hp.geco_params is not None:
            self.geco.load_state_dict(geco_params)
        super().load_state_dict(state_dict, *args, **kwargs)
