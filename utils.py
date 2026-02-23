import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

def Pi(ph):
    """
    Convert from homogeneous to inhomogeneous coordinates
    ph: (point dimension x number of points)
    """
    p = ph[:-1] / ph[-1]
    return p

def PiInv(p):
    """
    Convert from inhomogeneous to homogeneous coordinates
    p: (point dimension x number of points)
    """
    if p.ndim == 1:
        return np.append(p, 1)
    
    ph = np.vstack((p, np.ones(p.shape[-1])))
    return ph

def projectpoints(K, R, t, Q, distCoeffs=None):
    """
    K: camera matrix
    (R, t): pose of the camera
    Q: 3 x n matrix; n points in 3D to be projected into the camera
    """
    if distCoeffs is None:
        return K @ np.column_stack((R, t)) @ PiInv(Q)
    else:
        delta_r = lambda r: sum(k*r**(2*i+2) for i,k in enumerate(distCoeffs))
        dist = lambda p: p * (1 + delta_r(np.linalg.norm(p, axis=0)))

        p1 = np.column_stack((R, t)) @ PiInv(Q)
        p2 = PiInv(dist(Pi(p1)))
        p3 = K @ p2
        return p3