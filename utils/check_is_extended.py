import numpy as np

good_pts = {}
vertices = np.empty(shape=())


def check_is_extended(*args, **kwargs):
    pts, edges, proj_len = args
    if not len(edges):
        return False

    candidate_pts = np.where(proj_len < 0, edges[:, 0], edges[:, 1])
    for idx in candidate_pts:
        tid = good_pts.get(idx)
        if tid is None:
            continue
        target_pts = vertices[np.fromiter(tid, count=len(tid), dtype=np.int_)]
        center_pts = vertices[idx]
        edges_vec = target_pts - center_pts
        edges_len = np.linalg.norm(edges_vec, axis=-1, keepdims=True)
        edges_vec /= edges_len
        pts_vec = pts - center_pts
        proj_len = np.matmul(edges_vec, np.expand_dims(pts_vec, 1)).squeeze(1)
        is_in = proj_len <= edges_len
        if np.all(is_in):
            return True

    return False
