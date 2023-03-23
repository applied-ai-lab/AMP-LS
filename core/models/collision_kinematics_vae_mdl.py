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
from core.utils.pytorch_utils import (
    TensorModule,
    RAdam,
    ten2ar,
    ar2ten,
    map2torch,
    make_one_hot,
)
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
from core.models.kinematics_vae_mdl import KinematicsVAE
from core.modules.scene_encoder_modules import SceneCollisionEncoder


class CollisionKinematicsVAE(KinematicsVAE):
    """
    Kinematics VAE with collision predictor
    """

    def __init__(self, params, logger=None):
        super().__init__(params, logger)
        # define bounds of point clouds
        self.bounds = nn.Parameter(
            ar2ten(np.array(self._hp.bounds).astype(np.float32), device=self.device),
            requires_grad=False,
        )

        # voxel size
        self.vox_size = nn.Parameter(
            ar2ten(np.array(self._hp.vox_size).astype(np.float32), device=self.device),
            requires_grad=False,
        )

        # number of voxels in the bounded scene
        self.num_voxels = nn.Parameter(
            ((self.bounds[1] - self.bounds[0]) / self.vox_size).long(),
            requires_grad=False,
        )

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = ParamDict(
            {
                "pretrained_path": None,  # pretrained kinematics VAE weight path
                "bounds": [[-0.5, -0.8, 0.24], [0.5, 0.8, 0.6]],
                "vox_size": [0.125, 0.1, 0.09],
                "scene_collision_weight_path": None,
                "loss_pct": 0.1,
                "n_obj_points": 1024,
                "nz_latent_enc": 128,
                "nz_latent_mid": 128,
                "concat_z": False,
                "detach_encoder": False,
                "relative_geometry": True,
                "log_pc_interval": 1000,
            }
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        """Defines the network architecture."""
        super().build_network()
        if self._hp.pretrained_path is not None:
            self.load_model_weights(self, self._hp.pretrained_path)
        # self.scene_collision_net = SceneCollisionNet(bounds=self._hp.bounds, vox_size=self._hp.vox_size)
        self.scene_collision_encoder = SceneCollisionEncoder(
            bounds=self._hp.bounds, vox_size=self._hp.vox_size
        )
        if self._hp.scene_collision_weight_path is not None:
            self.scene_collision_encoder.load_state_dict(
                torch.load(self._hp.scene_collision_weight_path)["model_state_dict"],
                strict=False,
            )
        self.scene_encoder = Encoder(self._update_scene_encoder_params())

        input_dim = 256 * 10
        if self._hp.concat_z:
            input_dim += self._hp.nz_vae
        self.collision_predictor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = super().forward(inputs)
        output.collision = self.classify_collision(inputs, output.z)
        return output

    def get_scene_features(self, inputs):
        """
        Output scene embeddings given point cloud input.
        """
        scene_features = self.scene_collision_encoder(inputs.scene_pc[0])
        return scene_features

    def classify_collision(self, inputs, z, latent_detach=True):
        """
        Forward pass of a classifier.
        """
        if latent_detach:
            z = z.detach()

        if "scene_features" in inputs.keys():
            scene_features = inputs.scene_features
        else:
            scene_features = self.get_scene_features(inputs)

        if self._hp.scene_collision_weight_path is not None and self._hp.detach_encoder:
            scene_features = scene_features.detach()

        in_bounds = (inputs.trans > self.bounds[0] + 1e-5).all(dim=-1)
        in_bounds &= (inputs.trans < self.bounds[1] - 1e-5).all(dim=-1)
        inputs.trans[~in_bounds] = 0.0
        trans_inds = self.voxel_inds(inputs.trans, scale=2).long()

        tr_vox_centers = (
            self._inds_from_flat(trans_inds, scale=2) * self.vox_size * 2
            + self.vox_size / 2
            + self.bounds[0]
        )
        trans_offsets = inputs.trans - tr_vox_centers
        vox_trans_features = (
            scene_features[..., trans_inds[:, :, torch.arange(inputs.trans.shape[2])]]
            .squeeze(0)
            .permute(1, 2, 3, 0)
        )

        if self._hp.relative_geometry:
            vox_trans_features = torch.cat(
                [
                    vox_trans_features,
                    trans_offsets,
                    inputs.rots,
                    z.unsqueeze(2).repeat(1, 1, vox_trans_features.shape[2], 1),
                ],
                dim=-1,
            )
        else:
            vox_trans_features = torch.cat(
                [
                    vox_trans_features,
                    z.unsqueeze(2).repeat(1, 1, vox_trans_features.shape[2], 1),
                ],
                dim=-1,
            )

        scene_features = self.scene_encoder(vox_trans_features)
        scene_features = scene_features.reshape(
            (scene_features.shape[0], scene_features.shape[1], -1)
        )

        if self._hp.concat_z:
            scene_features = torch.cat([scene_features, z], dim=-1)
        collision = self.collision_predictor(scene_features).squeeze(-1)
        return collision

    def voxel_inds(self, xyz, scale=1):
        inds = ((xyz - self.bounds[0]) // (scale * self.vox_size)).int()
        return self._inds_to_flat(inds, scale=scale)

    def _inds_to_flat(self, inds, scale=1):
        flat_inds = inds * torch.cuda.IntTensor(
            [
                self.num_voxels[1:].prod() // (scale**2),
                self.num_voxels[2] // scale,
                1,
            ],
            device=self.device,
        )
        return flat_inds.sum(axis=-1)

    def _inds_from_flat(self, flat_inds, scale=1):
        ind0 = flat_inds // (self.num_voxels[1:].prod() // (scale**2))
        ind1 = (flat_inds % (self.num_voxels[1:].prod() // (scale**2))) // (
            self.num_voxels[2] // scale
        )
        ind2 = (flat_inds % (self.num_voxels[1:].prod() // (scale**2))) % (
            self.num_voxels[2] // scale
        )
        return torch.stack((ind0, ind1, ind2), dim=-1)

    def loss(self, model_output, inputs):
        losses = AttrDict()
        collision_losses = nn.BCEWithLogitsLoss(reduction="none")(
            model_output.collision, inputs.collision.float()
        )
        top_losses, _ = torch.topk(
            collision_losses,
            int(collision_losses.size(1) * self._hp.loss_pct),
            sorted=False,
        )
        rand_losses = collision_losses[
            :,
            torch.randint(
                collision_losses.size(1),
                (int(collision_losses.size(1) * self._hp.loss_pct),),
            ),
        ]
        collision_loss = 0.5 * (top_losses.mean() + rand_losses.mean())
        losses.collision_loss = AttrDict(weight=1.0, value=collision_loss)
        losses.total = self._compute_total_loss(losses)
        return losses

    def load_model_weights(self, model, checkpoint, epoch="latest"):
        """Loads weights for a given model from the given checkpoint directory."""
        checkpoint_dir = (
            checkpoint
            if os.path.basename(checkpoint) == "weights"
            else os.path.join(checkpoint, "weights")
        )  # checkpts in 'weights' dir
        checkpoint_path = CheckpointHandler.get_resume_ckpt_file(epoch, checkpoint_dir)
        CheckpointHandler.load_weights(checkpoint_path, model=model)

    def decode(self, z, inputs, latent_detach=False):
        collision = self.classify_collision(inputs, z, latent_detach=latent_detach)
        output = AttrDict(
            states=self.decoder(self.z2enc(z)).images, collision=collision
        )
        return output

    def decode_states(self, z):
        return self.decoder(self.z2enc(z)).images

    def _update_scene_encoder_params(self):
        params = copy.deepcopy(self._hp)
        state_dim = 1024 + self._hp.nz_vae
        if self._hp.relative_geometry:
            state_dim += 9

        return params.overwrite(
            AttrDict(
                use_convs=False,
                use_skips=False,  # no skip connections needed bc we are not reconstructing
                builder=LayerBuilderParams(
                    use_convs=False,
                    normalization=self._hp.normalization,
                    d=3,
                    activation_fn=nn.ReLU(inplace=True),
                ),
                nz_enc=256,
                ngf=128,
                nz_mid=512,
                n_processing_layers=1,
                state_dim=state_dim,
            )
        )

    def _update_latent_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(
            AttrDict(
                use_convs=False,
                use_skips=False,  # no skip connections needed bc we are not reconstructing
                builder=LayerBuilderParams(
                    use_convs=False,
                    normalization=self._hp.normalization,
                    d=3,
                    activation_fn=nn.ReLU(inplace=True),
                ),
                nz_enc=self._hp.nz_latent_enc,
                nz_mid=self._hp.nz_latent_mid,
                n_processing_layers=1,
                state_dim=7,
            )
        )

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        if phase != "train" or (
            phase == "train" and step % self._hp.log_pc_interval == 0
        ):
            self._logger.log_3d_obj(ten2ar(inputs.scene_pc[0]), "pc", step, phase)
        pos_ratio = torch.sum(inputs.collision.float()) / inputs.collision.shape[1]
        pred = torch.sigmoid(model_output.collision)
        bcr = BinaryClassificationResult(
            ten2ar(pred.squeeze(0)), ten2ar(inputs.collision.squeeze(0))
        )
        collision_origin_loss = nn.BCEWithLogitsLoss()(
            model_output.collision, inputs.collision.float()
        )
        self._logger.log_scalar(
            collision_origin_loss, "collision_origin_loss", step, phase
        )
        self._logger.log_scalar(bcr.accuracy, "accuracy", step, phase)
        self._logger.log_scalar(bcr.f1_score, "f1_score", step, phase)
        self._logger.log_scalar(bcr.tpr, "tpr", step, phase)
        self._logger.log_scalar(bcr.ap_score, "ap_score", step, phase)
        self._logger.log_scalar(pos_ratio, "pos_ratio", step, phase)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "geco" in state_dict:
            state_dict.pop("geco")
        super().load_state_dict(state_dict, *args, **kwargs)
