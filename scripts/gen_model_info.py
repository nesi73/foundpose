# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates the 3D bounding box and the diameter of 3D object models."""
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

def calc_model_info(object_id, output_path, model_path):
    models_info = {}
    misc.log("Processing model of object {}...".format(1))

    model = inout.load_ply(model_path)

    xs, ys, zs = model["pts"][:,0], model["pts"][:,1], model["pts"][:,2]
    bbox = misc.calc_3d_bbox(xs, ys, zs)

    # Calculated diameter.
    diameter = misc.calc_pts_diameter(model["pts"])

    models_info[object_id] = {
        "min_x": bbox[0],
        "min_y": bbox[1],
        "min_z": bbox[2],
        "size_x": bbox[3],
        "size_y": bbox[4],
        "size_z": bbox[5],
        "diameter": diameter,
    }

    print(models_info)
    # Save the calculated info about the object models.
    inout.save_json(output_path, models_info)

dataset_path = config.datasets_path
calc_model_info(1, dataset_path + "/lmo/models/models_info.json", dataset_path + "/lmo/models/obj_000001.ply")