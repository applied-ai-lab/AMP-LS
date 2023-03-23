import os
from collections.abc import Iterable

import h5py
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tra
from autolab_core import ColorImage, DepthImage
from core.utils.general_utils import ForkedPdb


class ObjectPlacementNotFound(Exception):
    pass


class SceneManager:
    def __init__(
        self,
        dataset_folder,
        table_dims=np.array([1.0, 1.6, 0.5]),
        renderer=None,
        cat=None,
    ):
        self._dataset_path = dataset_folder
        obj_info = h5py.File(os.path.join(self._dataset_path, "object_info.hdf5"), "r")
        self.mesh_info = obj_info["meshes"]
        self.categories = obj_info["categories"]

        self._collision_manager = trimesh.collision.CollisionManager()
        if renderer is not None and not isinstance(renderer, SceneRenderer):
            raise ValueError("renderer must be of type SceneRenderer")
        self._renderer = renderer

        self.objs = {}

        self._gravity_axis = 2
        self._table_dims = table_dims
        self._table_pose = np.eye(4)
        self._cat = cat

    @property
    def camera_pose(self):
        if self._renderer is None:
            raise ValueError("SceneManager does not contain a renderer!")
        return self._renderer.camera_pose

    @camera_pose.setter
    def camera_pose(self, cam_pose):
        if self._renderer is None:
            raise ValueError("SceneManager does not contain a renderer!")
        self._renderer.camera_pose = cam_pose

    @property
    def table_bounds(self):
        if not hasattr(self, "_table_bounds"):
            lbs = self._table_pose[:3, 3] - 0.5 * self._table_dims
            ubs = self._table_pose[:3, 3] + 0.5 * self._table_dims
            lbs[self._gravity_axis] = ubs[self._gravity_axis]
            ubs[self._gravity_axis] += 0.001
            lbs[self._gravity_axis] += 0.001
            self._table_bounds = (lbs, ubs)
        return self._table_bounds

    @table_bounds.setter
    def table_bounds(self, bounds):
        if not isinstance(bounds, np.ndarray):
            bounds = np.asarray(bounds)
        if bounds.shape != (2, 3):
            raise ValueError("Bounds is incorrect shape, should be (2, 3)")
        self._table_bounds = bounds

    def collides(self):
        return self._collision_manager.in_collision_internal()

    def min_distance(self, obj_manager):
        return self._collision_manager.min_distance_other(obj_manager)

    def add_object(self, name, mesh, info={}, pose=None, color=None):
        if name in self.objs:
            raise ValueError("Duplicate name: object {} already exists".format(name))

        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        color = np.asarray((0.7, 0.7, 0.7)) if color is None else np.asarray(color)
        mesh.visual.face_colors = np.tile(
            np.reshape(color, [1, 3]), [mesh.faces.shape[0], 1]
        )
        self.objs[name] = {"mesh": mesh, "pose": pose}
        if len(info) != 0:
            path = info["path"][()].decode("utf-8")
            self.objs[name]["path"] = path
        if "grasps" in info:
            self.objs[name]["grasps"] = info["grasps"][()]
        self._collision_manager.add_object(
            name,
            mesh,
            transform=pose,
        )
        if self._renderer is not None:
            self._renderer.add_object(name, mesh, pose)

        return True

    def remove_object(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self._collision_manager.remove_object(name)
        if self._renderer is not None:
            self._renderer.remove_object(name)
        del self.objs[name]

    def sample_obj(self, cat=None, obj=None):
        if cat is None:
            cat = np.random.choice(list(self.categories.keys()))
        elif isinstance(cat, Iterable):
            cat = np.random.choice(cat)
        if obj is None:
            obj = np.random.choice(list(self.categories[cat]))
        try:
            mesh_path = os.path.join(
                self._dataset_path, self.mesh_info[obj]["path"].asstr()[()]
            )
            mesh = trimesh.load(mesh_path, force="mesh")
            mesh.metadata["key"] = obj
            mesh.metadata["path"] = mesh_path
            info = self.mesh_info[obj]
        except (ValueError, TypeError):
            mesh = None
            info = None

        return mesh, info

    def place_obj(self, obj_id, mesh, info, max_attempts=5):
        self.add_object(obj_id, mesh, info=info)
        for _ in range(max_attempts):
            rand_stp = self._random_object_pose(info, *self.table_bounds)
            self.set_object_pose(obj_id, rand_stp)
            if not self.collides():
                return True

        self.remove_object(obj_id)
        return False

    def sample_and_place_obj(self, obj_id, max_attempts=1):
        for _ in range(max_attempts):
            obj_mesh, obj_info = self.sample_obj(cat=self._cat)
            if not obj_mesh:
                continue
            if self.place_obj(obj_id, obj_mesh, obj_info):
                break
            else:
                continue

    def arrange_table(self):
        # Create and add table mesh
        table_mesh = trimesh.creation.box(self._table_dims)
        table_mesh.metadata["key"] = "table"
        self.add_object(
            name="table",
            mesh=table_mesh,
            pose=self._table_pose,
        )

    def arrange_scene(self, num_objects, max_attempts=1):
        # Create and add table mesh
        table_mesh = trimesh.creation.box(self._table_dims)
        table_mesh.metadata["key"] = "table"
        self.add_object(
            name="table",
            mesh=table_mesh,
            pose=self._table_pose,
        )

        # Sample and add objects
        for i in range(num_objects):
            obj_id = "obj_{:d}".format(i + 1)
            self.sample_and_place_obj(obj_id, max_attempts=max_attempts)

    def arrange_manual_scene(self, obj_info):
        # Create and add table mesh
        table_mesh = trimesh.creation.box(self._table_dims)
        table_mesh.metadata["key"] = "table"
        self.add_object(
            name="table",
            mesh=table_mesh,
            pose=self._table_pose,
        )

        for i, key in enumerate(obj_info.keys()):
            obj_id = "obj_{:d}".format(i + 1)
            cat, obj = key.split("/")
            pose = obj_info[key]
            pose[2, 3] -= 0.25 - self._table_dims[2] / 2.0
            obj_mesh, info = self.sample_obj(obj=obj)
            if not obj_mesh:
                print("Mesh is not found")
                continue

            self.add_object(obj_id, obj_mesh, pose=pose, info=info)
            if self.collides():
                print("Collision found")
            #     self.remove_object(obj_id)
            #     return False
        return True

    def set_table_pose(self, table_pose):
        self._table_pose = table_pose

    def arrange_simple_scene(self, num_objects, max_attempts=5):
        # Create and add table mesh
        table_mesh = trimesh.creation.box(self._table_dims)
        table_mesh.metadata["key"] = "table"
        self.add_object(
            name="table",
            mesh=table_mesh,
            pose=self._table_pose,
        )

        for i in range(num_objects):
            cuboid_dim = np.random.uniform(low=[0.1, 0.1, 0.1], high=[0.3, 0.3, 0.3])
            cuboid_mesh = trimesh.creation.box(cuboid_dim)
            cuboid_mesh.metadata["key"] = "obj_{}".format(i + 1)
            pose = np.eye(4)
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                xy_pos = np.random.uniform(
                    low=[
                        -self._table_dims[0] / 2.0 + cuboid_dim[0] / 2.0,
                        -self._table_dims[1] / 2.0 + cuboid_dim[1] / 2.0,
                    ],
                    high=[
                        self._table_dims[0] / 2.0 - cuboid_dim[0] / 2.0,
                        self._table_dims[1] / 2.0 - cuboid_dim[1] / 2.0,
                    ],
                )
                pose[:3, 3] = np.concatenate(
                    [
                        xy_pos,
                        np.array(
                            [self._table_dims[2] / 2.0 + cuboid_dim[2] / 2.0 + 0.001]
                        ),
                    ]
                )
                self.add_object(
                    name="obj_{}".format(i + 1), mesh=cuboid_mesh, pose=pose
                )
                if self.collides():
                    self.remove_object("obj_{}".format(i + 1))
                    attempts += 1
                else:
                    placed = True

    def arrange_fixed_scene(self, offset=0):
        # Create and add table mesh
        table_mesh = trimesh.creation.box(self._table_dims)
        table_mesh.metadata["key"] = "table"
        self.add_object(
            name="table",
            mesh=table_mesh,
            pose=self._table_pose,
        )

        cuboid_dim = np.array([0.3, 0.1, 0.3])
        cuboid_mesh = trimesh.creation.box(cuboid_dim)
        cuboid_mesh.metadata["key"] = "obj_0"
        pose = np.eye(4)
        # pose[:3, 3] = [0.4, 0.2, 0.15]
        pose[:3, 3] = [
            -0.2 + offset,
            -0.3,
            self._table_dims[2] / 2.0 + cuboid_dim[2] / 2.0 + 0.001,
        ]
        self.add_object(name="obj_0", mesh=cuboid_mesh, pose=pose)

        cuboid_dim = np.array([0.3, 0.1, 0.3])
        cuboid_mesh = trimesh.creation.box(cuboid_dim)
        cuboid_mesh.metadata["key"] = "obj_1"
        pose = np.eye(4)
        pose[:3, 3] = [
            -0.2,
            0.2,
            self._table_dims[2] / 2.0 + cuboid_dim[2] / 2.0 + 0.001,
        ]
        # pose[:3, 3] = [0.4, -0.3, 0.15]
        self.add_object(name="obj_1", mesh=cuboid_mesh, pose=pose)

    def arrange_single_obs_scene(self):
        # Create and add table mesh
        table_mesh = trimesh.creation.box(self._table_dims)
        table_mesh.metadata["key"] = "table"
        self.add_object(
            name="table",
            mesh=table_mesh,
            pose=self._table_pose,
        )

        cuboid_dim = np.array([0.3, 0.1, 0.3])
        cuboid_mesh = trimesh.creation.box(cuboid_dim)
        cuboid_mesh.metadata["key"] = "obj_0"
        pose = np.eye(4)
        pose[:3, 3] = [
            0.0,
            0.0,
            self._table_dims[2] / 2.0 + cuboid_dim[2] / 2.0 + 0.001,
        ]
        self.add_object(name="obj_0", mesh=cuboid_mesh, pose=pose)

    def get_object_pose(self, name):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        return self.objs[name]["pose"]

    def set_object_pose(self, name, pose):
        if name not in self.objs:
            raise ValueError("object {} needs to be added first".format(name))
        self.objs[name]["pose"] = pose
        self._collision_manager.set_transform(
            name,
            pose,
        )
        if self._renderer is not None:
            self._renderer.set_object_pose(name, pose)

    def render_points(self):
        if self._renderer is not None:
            return self._renderer.render_points()

    def reset(self):
        if self._renderer is not None:
            self._renderer.reset()

        for name in self.objs:
            self._collision_manager.remove_object(name)

        self.objs = {}

    def _random_object_pose(self, obj_info, lbs, ubs):
        stps, probs = obj_info["stps"][()], obj_info["probs"][()]
        pose = stps[np.random.choice(len(stps), p=probs)].copy()
        pose[:3, 3] += np.random.uniform(lbs, ubs)
        z_rot = tra.rotation_matrix(
            2 * np.pi * np.random.rand(), [0, 0, 1], point=pose[:3, 3]
        )
        return z_rot @ pose


class SceneRenderer:
    def __init__(self):
        self._scene = pyrender.Scene()
        self._node_dict = {}
        self._camera_intr = None
        self._camera_node = None
        self._light_node = None
        self._renderer = None

    def create_camera(self, intr, znear, zfar):
        cam = pyrender.IntrinsicsCamera(intr.fx, intr.fy, intr.cx, intr.cy, znear, zfar)
        self._camera_intr = intr
        self._camera_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        self._scene.add_node(self._camera_node)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        self._light_node = pyrender.Node(light=light, matrix=np.eye(4))
        self._scene.add_node(self._light_node)
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=intr.width,
            viewport_height=intr.height,
            point_size=5.0,
        )

    @property
    def camera_pose(self):
        if self._camera_node is None:
            return None
        return self._camera_node.matrix

    @camera_pose.setter
    def camera_pose(self, cam_pose):
        if self._camera_node is None:
            raise ValueError("No camera in scene!")
        self._scene.set_pose(self._camera_node, cam_pose)
        self._scene.set_pose(self._light_node, cam_pose)

    def render_rgbd(self, depth_only=False):
        if depth_only:
            depth = self._renderer.render(self._scene, pyrender.RenderFlags.DEPTH_ONLY)
            color = None
            depth = DepthImage(depth, frame="camera")
        else:
            color, depth = self._renderer.render(self._scene)
            color = ColorImage(color, frame="camera")
            depth = DepthImage(depth, frame="camera")

        return color, depth

    def render_segmentation(self, full_depth=None):
        if full_depth is None:
            _, full_depth = self.render_rgbd(depth_only=True)

        self.hide_objects()
        output = np.zeros(full_depth.data.shape, dtype=np.uint8)
        for i, obj_name in enumerate(self._node_dict):
            self._node_dict[obj_name].mesh.is_visible = True
            _, depth = self.render_rgbd(depth_only=True)
            mask = np.logical_and(
                (np.abs(depth.data - full_depth.data) < 1e-6),
                np.abs(full_depth.data) > 0,
            )
            if np.any(output[mask] != 0):
                raise ValueError("wrong label")
            output[mask] = i + 1
            self._node_dict[obj_name].mesh.is_visible = False
        self.show_objects()

        return output, ["BACKGROUND"] + list(self._node_dict.keys())

    def render_points(self):
        _, depth = self.render_rgbd(depth_only=True)
        point_norm_cloud = depth.point_normal_cloud(self._camera_intr)

        pts = point_norm_cloud.points.data.T.reshape(depth.height, depth.width, 3)
        norms = point_norm_cloud.normals.data.T.reshape(depth.height, depth.width, 3)
        cp = self.camera_pose
        cp[:, 1:3] *= -1

        pt_mask = np.logical_and(
            np.linalg.norm(pts, axis=-1) != 0.0,
            np.linalg.norm(norms, axis=-1) != 0.0,
        )
        pts = tra.transform_points(pts[pt_mask], cp)
        return pts.astype(np.float32)

    def add_object(self, name, mesh, pose=None):
        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        node = pyrender.Node(
            name=name,
            mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False),
            matrix=pose,
        )
        self._node_dict[name] = node
        self._scene.add_node(node)

    def add_points(self, points, name, pose=None, color=None, radius=0.005):
        points = np.asanyarray(points)
        if points.ndim == 1:
            points = np.array([points])

        if pose is None:
            pose = np.eye(4)
        else:
            pose = pose.matrix

        color = np.asanyarray(color, dtype=np.float) if color is not None else None

        # If color specified per point, use sprites
        if color is not None and color.ndim > 1:
            self._renderer.point_size = 1000 * radius
            m = pyrender.Mesh.from_points(points, colors=color)
        # otherwise, we can make pretty spheres
        else:
            mesh = trimesh.creation.uv_sphere(radius, [20, 20])
            if color is not None:
                mesh.visual.vertex_colors = color
            poses = None
            poses = np.tile(np.eye(4), (len(points), 1)).reshape(len(points), 4, 4)
            poses[:, :3, 3::4] = points[:, :, None]
            m = pyrender.Mesh.from_trimesh(mesh, poses=poses)

        node = pyrender.Node(mesh=m, name=name, matrix=pose)
        self._node_dict[name] = node
        self._scene.add_node(node)

    def set_object_pose(self, name, pose):
        self._scene.set_pose(self._node_dict[name], pose)

    def has_object(self, name):
        return name in self._node_dict

    def remove_object(self, name):
        self._scene.remove_node(self._node_dict[name])
        del self._node_dict[name]

    def show_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = True

    def toggle_wireframe(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.primitives[0].material.wireframe ^= True

    def hide_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = False

    def reset(self):
        for name in self._node_dict:
            self._scene.remove_node(self._node_dict[name])
        self._node_dict = {}
