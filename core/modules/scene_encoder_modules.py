import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from torch.cuda.amp import autocast

OBJ_NPOINTS = [256, 64, None]
OBJ_RADII = [0.02, 0.04, None]
OBJ_NSAMPLES = [64, 128, None]
OBJ_MLPS = [[0, 64, 128], [128, 128, 256], [256, 256, 512]]
SCENE_PT_MLP = [3, 128, 256]
SCENE_VOX_MLP = [256, 512, 1024, 512]
CLS_FC = [2057, 1024, 256]

import math
from typing import List, Tuple

import torch
import torch.nn as nn


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def init_sine(dim_in, layer, c, w0, first=False):
    w_std = (1 / dim_in) if first else (math.sqrt(c / dim_in) / w0)
    layer.weight.data.uniform_(-w_std, w_std)
    if layer.bias is not None:
        layer.bias.data.uniform_(-w_std, w_std)


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],
        *,
        bn: bool = False,
        activation="relu",
        preact: bool = False,
        first: bool = False,
        name: str = "",
        instance_norm: bool = False,
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0))
                    else None,
                    preact=preact,
                    instance_norm=instance_norm,
                    first=first,
                ),
            )


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        activation,
        bn,
        init,
        conv=None,
        batch_norm=None,
        bias=True,
        preact=False,
        name="",
        instance_norm=False,
        instance_norm_func=None,
        first=False,
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        activation = Sine() if activation == "sine" else nn.ReLU(inplace=True)
        if isinstance(activation, Sine):
            # init_sine(conv_unit.in_channels, conv_unit, c=6., w0=activation.w0, first=first)
            init(conv_unit.weight)
            if bias:
                nn.init.constant_(conv_unit.bias, 0)
        else:
            init(conv_unit.weight)
            if bias:
                nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(
                    out_size, affine=False, track_running_stats=False
                )
            else:
                in_unit = instance_norm_func(
                    in_size, affine=False, track_running_stats=False
                )

        if preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

            if not bn and instance_norm:
                self.add_module(name + "in", in_unit)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

            if not bn and instance_norm:
                self.add_module(name + "in", in_unit)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class Conv1d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        activation="relu",
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
        first=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d,
            first=first,
        )


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        activation="relu",
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
        first=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d,
            first=first,
        )


class Conv3d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Tuple[int, int, int] = (1, 1, 1),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        activation="relu",
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
        first=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=BatchNorm3d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm3d,
            first=first,
        )


class FC(nn.Sequential):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        activation="relu",
        bn: bool = False,
        init=None,
        preact: bool = False,
        name: str = "",
        first=False,
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if activation == "sine":
            activation = Sine()
            # init_sine(fc.in_features, fc, c=6., w0=activation.w0, first=first)
            if init is not None:
                init(fc.weight)
            if not bn:
                nn.init.constant_(fc.bias, 0)
        else:
            if activation is not None:
                activation = nn.ReLU(inplace=True)
            if init is not None:
                init(fc.weight)
            if not bn:
                nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "fc", fc)

        if not preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + "activation", activation)


class SceneCollisionEncoder(nn.Module):
    def __init__(self, bounds, vox_size):
        super().__init__()
        self.bounds = nn.Parameter(
            torch.from_numpy(np.asarray(bounds)).float(), requires_grad=False
        )
        self.vox_size = nn.Parameter(
            torch.from_numpy(np.asarray(vox_size)).float(), requires_grad=False
        )
        self.num_voxels = nn.Parameter(
            ((self.bounds[1] - self.bounds[0]) / self.vox_size).long(),
            requires_grad=False,
        )

        self.scene_pt_mlp = nn.Sequential()
        for i in range(len(SCENE_PT_MLP) - 1):
            self.scene_pt_mlp.add_module(
                "pt_layer{}".format(i),
                Conv1d(SCENE_PT_MLP[i], SCENE_PT_MLP[i + 1], first=(i == 0)),
            )

        self.scene_vox_mlp = nn.ModuleList()
        for i in range(len(SCENE_VOX_MLP) - 1):
            scene_conv = nn.Sequential()
            if SCENE_VOX_MLP[i + 1] > SCENE_VOX_MLP[i]:
                scene_conv.add_module(
                    "3d_conv_layer{}".format(i),
                    Conv3d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                )
                scene_conv.add_module(
                    "3d_max_layer{}".format(i), nn.MaxPool3d(2, stride=2)
                )
            else:
                scene_conv.add_module(
                    "3d_convt_layer{}".format(i),
                    nn.ConvTranspose3d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            self.scene_vox_mlp.append(scene_conv)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def _inds_to_flat(self, inds, scale=1):
        flat_inds = inds * torch.cuda.IntTensor(
            [
                self.num_voxels[1:].prod() // (scale**2),
                self.num_voxels[2] // scale,
                1,
            ],
            device=self.num_voxels.device,
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

    def voxel_inds(self, xyz, scale=1):
        inds = ((xyz - self.bounds[0]) // (scale * self.vox_size)).int()
        return self._inds_to_flat(inds, scale=scale)

    def get_scene_features(self, scene_pc):
        scene_xyz, scene_features = self._break_up_pc(scene_pc)
        scene_inds = self.voxel_inds(scene_xyz)

        # Featurize scene points and max pool over voxels
        scene_vox_centers = (
            self._inds_from_flat(scene_inds) * self.vox_size
            + self.vox_size / 2
            + self.bounds[0]
        )
        scene_xyz_centered = (scene_pc[..., :3] - scene_vox_centers).transpose(2, 1)
        if scene_features is not None:
            scene_features = self.scene_pt_mlp(
                torch.cat((scene_xyz_centered, scene_features), dim=1)
            )
        else:
            scene_features = self.scene_pt_mlp(scene_xyz_centered)
        max_vox_features = torch.zeros(
            (*scene_features.shape[:2], self.num_voxels.prod())
        ).to(scene_pc.device)
        if scene_inds.max() >= self.num_voxels.prod():
            print(
                scene_xyz[range(len(scene_pc)), scene_inds.max(axis=-1)[1]],
                scene_inds.max(),
            )
        assert scene_inds.max() < self.num_voxels.prod()
        assert scene_inds.min() >= 0

        with autocast(enabled=False):
            max_vox_features[..., : scene_inds.max() + 1] = torch_scatter.scatter_max(
                scene_features.float(), scene_inds[:, None, :]
            )[0]
        max_vox_features = max_vox_features.reshape(
            *max_vox_features.shape[:2], *self.num_voxels.int()
        )

        # 3D conv over voxels
        l_vox_features = [max_vox_features]
        for i in range(len(self.scene_vox_mlp)):
            li_vox_features = self.scene_vox_mlp[i](l_vox_features[i])
            l_vox_features.append(li_vox_features)

        # Stack features from different levels
        stack_vox_features = torch.cat((l_vox_features[1], l_vox_features[-1]), dim=1)
        stack_vox_features = stack_vox_features.reshape(
            *stack_vox_features.shape[:2], -1
        )
        return stack_vox_features

    # scene_pc: (b, n_scene_pts, 6); obj_pc: (b, n_obj_pts, 6);
    # trans: (b, q, 3); rots: (b, q, 6)
    def forward(self, scene_pc):
        scene_features = self.get_scene_features(scene_pc)
        return scene_features
