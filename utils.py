import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import sys
import importlib

def box3d(n=16): # Given
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
    if Q.shape[0] == 4:
        Q = Pi(Q)
    
    if distCoeffs is None:
        return K @ np.column_stack((R, t)) @ PiInv(Q)
    else:
        delta_r = lambda r: sum(k*r**(2*i+2) for i,k in enumerate(distCoeffs))
        dist = lambda p: p * (1 + delta_r(np.linalg.norm(p, axis=0)))

        p1 = np.column_stack((R, t)) @ PiInv(Q)
        p2 = PiInv(dist(Pi(p1)))
        p3 = K @ p2
        return p3

def undistortImage(im, K, distCoeffs):
    delta_r = lambda r: sum(k * r**(2*i+2) for i,k in enumerate(distCoeffs))
    dist = lambda p: p * (1.0 + delta_r(np.linalg.norm(p, axis=0)))

    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)
    # q = ...
    # q_d = ...
    # p_d = ...

    # q: 2 x N normalized coords (undistorted)
    q = Pi(np.linalg.inv(K) @ p)

    # q_d: 2 x N normalized coords (distorted)
    q_d = dist(q)

    # p_d: 3 x N pixel homogeneous coords in the INPUT (distorted) image
    p_d = K @ PiInv(q_d)


    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    # plt.figure()
    # plt.scatter(*p_d[:2])
    # plt.show()
    assert (p_d[2]==1).all(), 'You did a mistake somewhere'
    im_undistorted = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)
    return im_undistorted

def hest(q1, q2, normalize=False):
    """
    Find H that best approximates q1 = H @ q2
    q1 and q2 are inhom.
    """
    assert q1.shape == q2.shape
    assert q1.shape[0] == 2

    q2 = PiInv(q2)
    if normalize:
        T1 = normalize2d(q1)
        T2 = normalize2d(q2)
        q1 = T1 @ PiInv(q1)
        q2 = T2 @ q2
    
    B_list = []
    for i in range(q1.shape[1]):
        q1i_cross =  np.array([
            [0, -1, q1[1,i]],
            [1, 0, -q1[0,i]],
            [-q1[1,i], q1[0,i], 0]
        ], dtype=float)
        B_i = np.kron(q2[:, i], q1i_cross)
        B_list.append(B_i)
    
    B = np.concatenate(B_list, axis=0)
    U, S, VT = np.linalg.svd(B)
    H = VT[-1,:].reshape((3, 3)).T
    if normalize:
        H = np.linalg.inv(T1) @ H @ T2

    return H

def normalize2d(p):
    """p is inhom."""
    assert p.shape[0] in (2, 3)
    if p.shape[0] == 3:
        p = Pi(p)

    mu = np.mean(p, axis=1)
    sig = np.std(p, axis=1)
    Tinv = np.array([
        [sig[0], 0, mu[0]],
        [0, sig[1], mu[1]],
        [0, 0, 1]
    ])
    T = np.linalg.inv(Tinv)
    return T

def warpImage(im, H): # Given
    imWarp = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
    return imWarp

def CrossOp(p):
    assert p.shape[0] == 3
    assert p.size == 3
    p_cross = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ], dtype=float)
    return p_cross

def DrawLine(l, shape): # Given
    #Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame
    def in_frame(l_im):
        q = np.cross(l.flatten(), l_im)
        q = q[:2]/q[2]
        if all(q>=0) and all(q+1<=shape[1::-1]):
            return q
    lines = [[1, 0, 0], [0, 1, 0], [1, 0, 1-shape[1]], [0, 1, 1-shape[0]]]
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    if (len(P)==0):
        print("Line is completely outside image")
    plt.plot(*np.array(P).T)

def triangulate(q_list, P_list):
    """
    Triangulate a single 3D point that has been seen by n different cameras.
    Return the triangulation of the point in 3D using the linear algorithm.
    q_list: (q1, q2,..., qn) - pixel coordinates
    P_list: (P1, P2,..., Pn) - projection matrices
    """
    assert len(q_list) == len(P_list)
    B = []
    for q, P in zip(q_list, P_list):
        B.append(np.stack((P[2]*q[0] - P[0], P[2]*q[1] - P[1]), axis=0))
    
    B = np.concatenate(B, axis=0)
    U, S, VT = np.linalg.svd(B)
    Q = VT[-1,:]
    return Q

def pest(q, Q, normalize=False):
    """
    Estimate projection matrix P using DLT.
    q: 2D projections, hom.
    Q: 3D points, hom.

    Normalizing 2D points:
    qi = P @ Qi
    T @ qi = (T @ P) @ Qi
    qi_norm = P_norm @ Qi
    """
    assert q.shape[1] == Q.shape[1]
    assert q.shape[0] == 3
    assert Q.shape[0] == 4

    if normalize:
        T = normalize2d(q)
        q = T @ q
    
    B = []
    q = q / q[-1]
    print(q)
    Q = Q / Q[-1]
    for i in range(Q.shape[1]):
        Bi = np.kron(Q[:, i], CrossOp(q[:, i]))
        assert Bi.shape == (3, 12)
        B.append(Bi)
    
    B = np.concatenate(B, axis=0)

    U, S, VT = np.linalg.svd(B)
    P = VT[-1].reshape((4, 3)).T

    print('sing.:', S.max(), S.min())

    if normalize:
        P = np.linalg.inv(T) @ P
    return P