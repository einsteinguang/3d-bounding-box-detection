import torch
import numpy as np
import torch.nn.functional as F


def extract_6d_params_from_corners(corners_8x3):
    """
    corners_8x3: Tensor of shape (8, 3) for a single bounding box.
                 Assumes corners are in the same order used by plot_bbox_and_pointcloud().

    Returns: (center, dims, orient_6d)
      center: (3,)    - the average of all corners
      dims:   (3,)    - (length_x, length_y, length_z)
      orient_6d: (6,) - first two columns of the box's 3x3 rotation
    """
    # Compute center as the mean of all corners
    center = corners_8x3.mean(dim=0)  # (3,)

    # Suppose corner(0) is "origin," corner(1) differs in x, corner(3) in y, corner(4) in z.
    xdir = corners_8x3[1] - corners_8x3[0]  # (3,)
    ydir = corners_8x3[3] - corners_8x3[0]  # (3,)
    zdir = corners_8x3[4] - corners_8x3[0]  # (3,)

    # Their lengths => dims
    dx = xdir.norm(p=2)
    dy = ydir.norm(p=2)
    dz = zdir.norm(p=2)
    dims = torch.stack([dx, dy, dz], dim=0)  # (3,)

    # Create the rotation matrix columns [b1, b2, b3],
    # but we only store the first 2 columns in 6D form.
    # b1 = normalized xdir
    b1 = xdir / (dx + 1e-8)

    # b2 = make ydir orthonormal to b1
    # subtract out proj of ydir on b1, then normalize
    ydir_ortho = ydir - (ydir.dot(b1))*b1
    b2 = ydir_ortho / (ydir_ortho.norm(p=2) + 1e-8)

    # We do NOT store b3 explicitly because 6D representation uses only first 2 columns.
    # The "Gram-Schmidt" procedure would define b3 = b1 x b2 inside the reconstruction.
    # Put b1 and b2 into a 6D vector
    orient_6d = torch.cat([b1, b2], dim=0)  # shape (6,)

    return center, dims, orient_6d


def extract_6d_ordered_params_from_corners(corners_8x3):
    """
    corners_8x3: Tensor of shape (8, 3) for a single bounding box.
                 Assumes corners are in the same order used by plot_bbox_and_pointcloud().

    Returns: (center, dims, orient_6d)
      center: (3,)    - the average of all corners
      dims:   (3,)    - (length_x, length_y, length_z) sorted descending (largest first)
      orient_6d: (6,) - first two columns of the box's 3x3 rotation, corresponding to the largest two dims
    """
    # Compute center as the mean of all corners.
    center = corners_8x3.mean(dim=0)  # (3,)

    # Compute three directional vectors relative to corner 0.
    # We assume that corner0 is the "origin" and:
    #   corner1 differs along one axis, corner3 along another, and corner4 along the third.
    xdir = corners_8x3[1] - corners_8x3[0]  # (3,)
    ydir = corners_8x3[3] - corners_8x3[0]  # (3,)
    zdir = corners_8x3[4] - corners_8x3[0]  # (3,)

    # Compute the norms of these directions.
    dx = xdir.norm(p=2)
    dy = ydir.norm(p=2)
    dz = zdir.norm(p=2)

    # Stack the dims into a tensor and determine the order (largest first).
    dims_tensor = torch.stack([dx, dy, dz], dim=0)  # shape (3,)
    order = torch.argsort(dims_tensor, descending=True)  # indices sorted descending

    # Reorder the dimensions and corresponding direction vectors.
    dims_sorted = dims_tensor[order]  # sorted dims: (3,)
    # Create a list of the direction vectors.
    dirs = [xdir, ydir, zdir]
    ordered_dirs = [dirs[i] for i in order]

    # Define b1 as the normalized vector of the largest direction.
    b1 = ordered_dirs[0] / (ordered_dirs[0].norm(p=2) + 1e-8)
    # For b2, take the second largest direction and orthogonalize it against b1.
    b2_temp = ordered_dirs[1] - (ordered_dirs[1].dot(b1)) * b1
    b2 = b2_temp / (b2_temp.norm(p=2) + 1e-8)

    # The 6D orientation is the concatenation of b1 and b2.
    orient_6d = torch.cat([b1, b2], dim=0)  # shape (6,)

    return center, dims_sorted, orient_6d


def reconstruct_8_corners_from_6d(center, dims, orient_6d):
    """
    center:     (3,)   e.g. predicted or GT box center
    dims:       (3,)   e.g. predicted or GT box dimension (dx, dy, dz)
    orient_6d:  (6,)   The first two columns of a rotation matrix
                       from the "6D continuous orientation" representation.
    Returns corners_8x3: (8,3) the 8 corners in the same order as used by plot_bbox_and_pointcloud().
    """
    # Recover b1, b2 from orient_6d
    b1 = orient_6d[0:3]
    b2 = orient_6d[3:6]

    # Safely normalize
    b1 = b1 / (b1.norm() + 1e-8)
    b2 = b2 - (b2.dot(b1)) * b1
    b2 = b2 / (b2.norm() + 1e-8)

    # b3 = b1 x b2  (right-handed)
    b3 = torch.cross(b1, b2)

    # Build local axes scaled by dims
    dx, dy, dz = dims
    # half extents in each direction
    hx = dx / 2.0
    hy = dy / 2.0
    hz = dz / 2.0

    # Construct 8 corners in the same layout
    # We'll interpret "bottom face" as z- => minus hz, "top face" as z+ => plus hz
    # Then edges: (0->1) is along +x, (0->3) is along +y, (0->4) is along +z
    c = center
    corners = []
    corners.append(c + (-hx)*b1 + (-hy)*b2 + ( hz)*b3)
    corners.append(c + ( hx)*b1 + (-hy)*b2 + ( hz)*b3)
    corners.append(c + ( hx)*b1 + ( hy)*b2 + ( hz)*b3)
    corners.append(c + (-hx)*b1 + ( hy)*b2 + ( hz)*b3)
    corners.append(c + (-hx)*b1 + (-hy)*b2 + (-hz)*b3)
    corners.append(c + ( hx)*b1 + (-hy)*b2 + (-hz)*b3)
    corners.append(c + ( hx)*b1 + ( hy)*b2 + (-hz)*b3)
    corners.append(c + (-hx)*b1 + ( hy)*b2 + (-hz)*b3)
    return torch.stack(corners, dim=0)  # (8,3)


def estimate_pc_radius(pc):
    """
    pc: torch.tensor shape (N, 3)
    Returns: radius of a sphere that encloses 99% of the point cloud
    """
    # Compute the center of the point cloud
    pc_center = pc.mean(dim=0)
    # Compute the distance from the center to each point
    distances = torch.norm(pc - pc_center, dim=1)
    # Sort the distances in ascending order
    sorted_distances, _ = torch.sort(distances)
    # Find the distance that encloses 99% of the points
    radius = sorted_distances[int(0.99 * len(sorted_distances))]
    return radius, pc_center


def compute_loss(pred, center, dims, gt_6d):
    # pc: tensor of shape (N, 3)
    # pred: shape (12,) [center(3), dims(3), orient_6d(6)]

    # scale the difference by mean dimension
    loss_center = F.l1_loss(pred[:3], center)
    loss_dims = F.l1_loss(pred[3:6], dims)
    loss_orient = F.l1_loss(pred[6:], gt_6d)

    return {"center": loss_center, "dims": loss_dims, "orient": loss_orient}