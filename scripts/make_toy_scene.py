import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def make_furniture(category):
    width, depth, height = (1.0, 2.0, 0.5) if category == "bed" else (0.7, 0.7, 0.45)
    if category == "sofa":
        width = 1.8
    mesh = o3d.geometry.TriangleMesh.create_box(width, depth, height)
    mesh.translate((-width / 2, -depth / 2, 0))
    if category in ("chair", "armchair", "sofa"):
        back = o3d.geometry.TriangleMesh.create_box(width, 0.08, 0.45)
        back.translate((-width / 2, depth / 2 - 0.08, height))
        mesh += back
    cloud = mesh.sample_points_uniformly(number_of_points=4000)
    points = np.asarray(cloud.points)
    top = np.flatnonzero(np.abs(points[:, 2] - height) < 1e-5)
    standpoints = [
        [0, -depth / 2 - 0.6, 0], [width / 2 + 0.6, 0, 0],
        [-width / 2 - 0.6, 0, 0], [0, depth / 2 + 0.6, 0],
    ]
    return mesh, cloud, top, standpoints


def main():
    parser = argparse.ArgumentParser(description="Generate procedural furniture for pipeline smoke tests, not paper benchmarks.")
    parser.add_argument("--output", type=Path, default=Path("scene_data"))
    args = parser.parse_args()
    root = args.output
    if (root / "obj_info_3dfront.json").exists() or (root / "3dfront/model_info.json").exists():
        raise FileExistsError("Refusing to overwrite an existing scene dataset.")
    categories = ["chair", "armchair", "sofa", "stool", "bed", "table", "desk", "stand", "cabinet", "shelf", "wardrobe", "lamp"]
    objects = {}
    models = {}
    for category in categories:
        identifier = "toy_" + category
        directory = root / "3dfront/obj_models" / identifier
        directory.mkdir(parents=True, exist_ok=False)
        mesh, cloud, top, standpoints = make_furniture(category)
        o3d.io.write_triangle_mesh(str(directory / "mesh.obj"), mesh)
        o3d.io.write_point_cloud(str(directory / "point_cloud_seg.ply"), cloud)
        (directory / "seat_indices.txt").write_text("\n".join(map(str, top.tolist())) + "\n")
        info = {
            "raw_id": identifier, "category": category, "super_category": category,
            "scale": 1, "rotate": 0, "standpoint": standpoints,
        }
        objects[category] = {identifier: info}
        models[identifier] = info
    (root / "obj_info_3dfront.json").write_text(json.dumps(objects, indent=2) + "\n")
    (root / "3dfront/model_info.json").write_text(json.dumps(models, indent=2) + "\n")
    layout = {
        "obj_angles": [0, 0], "obj_trans": [[0, 0, 0], [3, 0, 0]],
        "obj_raw_ids": ["toy_chair", "toy_bed"], "obj_labels": ["chair", "bed"],
        "obj_names": ["chair1", "bed1"], "obj_scales": [[1, 1, 1], [1, 1, 1]],
        "room_width": 8, "room_length": 8,
    }
    directory = root / "3dfront/layouts"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "example_room.json").write_text(json.dumps(layout, indent=2) + "\n")
    print("Generated procedural furniture in {}. Not a replacement for benchmark assets.".format(root))


if __name__ == "__main__":
    main()
