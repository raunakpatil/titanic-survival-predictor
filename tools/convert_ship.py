"""Convert the Titanic .3ds model into the small GLB used by the 3D ship viewer.

    python tools/convert_ship.py titanic.3ds ship_viewer/titanic.glb [--faces 150000] [--cell 0.25]

The source model has ~2.4M triangles spread over 12k objects. Vertex clustering
(snapping to a grid of ``--cell`` model units) collapses details too small to see,
then quadric decimation brings each material down to its share of ``--faces``.
Needs: pip install numpy trimesh fast-simplification
"""
import argparse
import struct

import fast_simplification
import numpy as np
import trimesh


def read_3ds(path):
    """Minimal 3DS reader: per-object vertices, faces, face→material groups and material colours."""
    data = open(path, "rb").read()
    objects, materials = [], {}

    def cstr(off):
        end = data.index(b"\0", off)
        return data[off:end].decode("latin1"), end + 1

    def walk(off, end, ctx):
        while off < end - 6:
            cid, length = struct.unpack_from("<HI", data, off)
            if length < 6:
                break
            body, chunk_end = off + 6, off + length
            if cid in (0x4D4D, 0x3D3D, 0x4100):          # main, editor, triangle mesh
                walk(body, chunk_end, ctx)
            elif cid == 0xAFFF:                           # material block
                mat = {}
                walk(body, chunk_end, {"mat": mat})
                if "name" in mat:
                    materials[mat["name"]] = mat
            elif cid == 0x4000:                           # named object
                name, start = cstr(body)
                obj = {"name": name, "v": None, "f": None, "groups": []}
                objects.append(obj)
                walk(start, chunk_end, {"obj": obj})
            elif cid == 0x4110:                           # vertex list
                n = struct.unpack_from("<H", data, body)[0]
                ctx["obj"]["v"] = np.frombuffer(data, "<f4", n * 3, body + 2).reshape(n, 3)
            elif cid == 0x4120:                           # face list (+ nested material groups)
                n = struct.unpack_from("<H", data, body)[0]
                faces = np.frombuffer(data, "<u2", n * 4, body + 2).reshape(n, 4)
                ctx["obj"]["f"] = faces[:, :3].astype(np.int64)
                walk(body + 2 + n * 8, chunk_end, ctx)
            elif cid == 0x4130:                           # faces using a material
                name, start = cstr(body)
                k = struct.unpack_from("<H", data, start)[0]
                ctx["obj"]["groups"].append((name, np.frombuffer(data, "<u2", k, start + 2).astype(np.int64)))
            elif cid == 0xA000:                           # material name
                ctx["mat"]["name"] = cstr(body)[0]
            elif cid == 0xA020:                           # diffuse colour
                sub, sub_len = struct.unpack_from("<HI", data, body)
                if sub == 0x0011:
                    ctx["mat"]["diffuse"] = tuple(data[body + 6:body + 9])
                elif sub == 0x0010:
                    ctx["mat"]["diffuse"] = tuple(int(c * 255) for c in struct.unpack_from("<3f", data, body + 6))
            off = chunk_end

    walk(0, len(data), {})
    return objects, materials


def triangles_by_material(objects):
    by_mat = {}
    for obj in objects:
        if obj["f"] is None or obj["name"].startswith("3d_man"):   # skip the scale figures
            continue
        tris = obj["v"][obj["f"]]
        assigned = np.zeros(len(tris), bool)
        for mat, idx in obj["groups"]:
            idx = idx[idx < len(tris)]
            assigned[idx] = True
            by_mat.setdefault(mat, []).append(tris[idx])
        if (~assigned).any():
            by_mat.setdefault("FrontCol", []).append(tris[~assigned])
    return {m: np.vstack(t) for m, t in by_mat.items()}


def cluster(tris, cell):
    """Snap vertices to a grid and drop collapsed or duplicate triangles."""
    q = np.round(tris.reshape(-1, 3) / cell).astype(np.int64)
    verts, inv = np.unique(q, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return trimesh.Trimesh(verts * cell, faces[np.sort(first)], process=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--faces", type=int, default=150_000, help="total triangle budget")
    ap.add_argument("--cell", type=float, default=0.25, help="vertex clustering grid size (model units)")
    args = ap.parse_args()

    objects, materials = read_3ds(args.src)
    meshes = {m: cluster(t, args.cell) for m, t in triangles_by_material(objects).items()}
    total = sum(len(m.faces) for m in meshes.values())

    scene = trimesh.Scene()
    for name, mesh in sorted(meshes.items(), key=lambda kv: -len(kv[1].faces)):
        target = max(int(args.faces * len(mesh.faces) / total), 100)
        if len(mesh.faces) > target:
            v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                                target_reduction=1 - target / len(mesh.faces))
            mesh = trimesh.Trimesh(v, f, process=False)
        rgb = materials.get(name, {}).get("diffuse", (200, 200, 200))
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=name, baseColorFactor=[*rgb, 255], metallicFactor=0.0, roughnessFactor=0.85, doubleSided=True))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    # Z-up → glTF Y-up; bow/stern centred on x=0, keel at y=0
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    lo, hi = scene.bounds
    scene.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    scene.export(args.out)
    faces = sum(len(g.faces) for g in scene.geometry.values())
    print(f"{total:,} clustered faces -> {faces:,}; bounds {np.round(scene.bounds, 1).tolist()}")


if __name__ == "__main__":
    main()
