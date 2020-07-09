import numpy as np


def project_points(calib, points3d):
    """
    Projects 3D points using a calibration matrix.

    Parameters:
        points3d: ndarray of shape (nPoints, 3)
    """
    assert points3d.ndim == 2 and points3d.shape[1] == 3

    p = np.empty((len(points3d), 2))
    p[:, 0] = points3d[:, 0] / points3d[:, 2] * calib[0, 0] + calib[0, 2]
    p[:, 1] = points3d[:, 1] / points3d[:, 2] * calib[1, 1] + calib[1, 2]

    return p


def calibration_matrix(points2d, points3d):
    """
    Calculates camera calibration matrix (no distortion) from 3D points and their projection.
    Only works if all points are away from the camera, eg all z coordinates>0.

    Returns:
        calib, reprojection error, x residuals, y residuals, x singular values, y singular values
    """
    assert points2d.ndim == 2 and points2d.shape[1] == 2
    assert points3d.ndim == 2 and points3d.shape[1] == 3

    A = np.column_stack([points3d[:, 0] / points3d[:, 2], np.ones(len(points3d))])
    px, resx, _, sx = np.linalg.lstsq(A, points2d[:, 0], rcond=None)

    A = np.column_stack([points3d[:, 1] / points3d[:, 2], np.ones(len(points3d))])
    py, resy, _, sy = np.linalg.lstsq(A, points2d[:, 1], rcond=None)

    calib = np.eye(3)
    calib[0, 0] = px[0]
    calib[1, 1] = py[0]
    calib[0, 2] = px[1]
    calib[1, 2] = py[1]

    # Calculate mean reprojection error
    # p = np.empty((len(points3d), 2))
    # p[:, 0] = points3d[:, 0] / points3d[:, 2] * calib[0, 0] + calib[0, 2]
    # p[:, 1] = points3d[:, 1] / points3d[:, 2] * calib[1, 1] + calib[1, 2]
    p = project_points(calib, points3d)
    reproj = np.mean(np.abs(points2d - p))

    return calib, reproj, resx, resy, sx, sy
