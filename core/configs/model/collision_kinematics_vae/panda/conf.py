import os

from core.models.collision_kinematics_vae_mdl import CollisionKinematicsVAE
from core.components.logger import Logger
from core.utils.general_utils import AttrDict
from core.configs.default_data_configs.scene_collision import data_spec
from core.data.scene_collision.scene_collision_data_loader import SceneCollisionDataset
from core.components.evaluator import (
    ImageEvaluator,
    DummyEvaluator,
    MultiImageEvaluator,
)
from core.utils.aug_utils import *

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    "model": CollisionKinematicsVAE,
    "model_test": CollisionKinematicsVAE,
    "logger": Logger,
    "logger_test": Logger,
    "evaluator": DummyEvaluator,
    "data_dir": "./datasets/panda/scene_shapenet_x1.5",
    "num_epochs": 100,
    "epoch_cycles_train": 1,
    "batch_size": 1,
    "lr": 1e-3,
    "optimizer": "sgd",
    "momentum": 0.9,
}
configuration = AttrDict(configuration)

model_config = {
    "has_rot": True,
    "state_dim": 16,
    "nz_enc": 512,
    "nz_mid": 512,
    "nz_vae": 7,
    "activation_fn": nn.ELU(inplace=True),
    "normalization": "none",
    "pretrained_path": "./experiments/model/kinematics_vae/panda/geco.g0.0002.z512.latent7.st16.6d.kcrl0.split.pr.iterD",
    "bounds": [[-0.5, -0.8, 0.24], [0.5, 0.8, 0.6]],
    # 'bounds':[[-0.5, -0.8, 0.14], [0.5, 0.8, 0.5]],
    "concat_z": False,
    "relative_geometry": False,
}
model_config = AttrDict(model_config)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.dataset_class = SceneCollisionDataset
data_config.dataset_spec.subseq_len = 1
data_config.dataset_spec.meshes = "./datasets/shapenet_x1.5"
data_config.dataset_spec.query_size = 2048
data_config.dataset_spec.n_scene_points = 32768
data_config.dataset_spec.n_obj_low = (4,)
data_config.dataset_spec.n_obj_high = (8,)
data_config.dataset_spec.intrinsics = AttrDict(
    frame="camera",
    fx=616.36529541,
    fy=616.20294189,
    cx=310.25881958,
    cy=236.59980774,
    skew=0.0,
    width=640,
    height=480,
)

data_config.dataset_spec.extrinsics = AttrDict(
    azimuth=[-0.2, 0.2],
    elevation=[0.6, 1.0],
    radius=[1.5, 2.0],
)
data_config.dataset_spec.bounds = model_config.bounds
data_config.dataset_spec.has_rot = model_config["has_rot"]
data_config.dataset_spec.robot_urdf = (
    "./core/data/scene_collision/assets/data/panda/panda.urdf"
)
data_config.dataset_spec.camera_pose_mean = False
# data_config.dataset_spec.data_size = 1000
# data_config.dataset_spec.table_dims = np.array([1.0, 1.6, 0.3])
