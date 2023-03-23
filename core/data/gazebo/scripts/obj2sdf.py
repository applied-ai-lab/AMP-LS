import os, sys
import numpy as np
import h5py
import tqdm

import trimesh


obj_info = h5py.File(
    os.path.join(os.environ["DATA_DIR"], "shapenet_x1.5/object_info.hdf5"), "r"
)

for cat in obj_info["categories"]:
    print("Processing Category {}".format(cat))
    for filename in tqdm.tqdm(obj_info["categories"][cat]):
        sdf_model_file_text = "<?xml version='1.0'?>\n"
        object_name = cat + "_" + filename.decode("utf-8")
        model_path = os.path.join(cat, filename.decode("utf-8") + ".obj")
        sdf_model_file_text += (
            '<sdf version="1.6"> \n \
                <static>1</static> \n \
                <model name="'
            + object_name
            + '"> \n \
                   <pose>0 0 0 0 0 0</pose> \n \
                    <link name="link"> \n \
                        <collision name="collision"> \n \
                            <geometry> \n \
                                <mesh> \n \
                                    <uri>model://'
            + model_path
            + '</uri> \n \
                                </mesh> \n \
                            </geometry> \n \
                        </collision>  \n \
                        <visual name="visual"> \n \
                            <material> \n \
                                <script> \n \
                                    <uri>file://media/materials/scripts/gazebo.material</uri>\n \
                                    <name>Gazebo/Green</name>\n \
                                </script>\n \
                            </material>\n \
                            <geometry> \n \
                                <mesh> \n \
                                    <uri>model://'
            + model_path
            + "</uri> \n \
                                </mesh> \n \
                            </geometry> \n \
                        </visual> \n \
                    </link> \n \
                    <static>1</static> \n \
                </model> \n \
            </sdf>"
        )
        f = open(
            os.path.join(
                os.environ["DATA_DIR"],
                "shapenet_x1.5",
                cat,
                filename.decode("utf-8") + ".sdf",
            ),
            "w",
        )
        f.write(sdf_model_file_text)
        f.close()
