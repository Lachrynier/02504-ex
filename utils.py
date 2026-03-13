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

def ensure_inhom(p, dim, *, eps=1e-12):
    """
    Return inhom coords of dimension `dim`.

    Accepts:
      - (dim, N) or (dim,)       inhom
      - (dim+1, N) or (dim+1,)   hom

    Returns:
      - (dim, N) or (dim,)
    """
    p = np.asarray(p)

    # 1D case
    if p.ndim == 1:
        if p.shape[0] == dim:
            return p
        if p.shape[0] == dim + 1:
            w = p[-1]
            if abs(w) < eps:
                raise ValueError("Homogeneous w is ~0; cannot dehomogenize.")
            return p[:-1] / w
        raise ValueError(f"Expected length {dim} or {dim+1}, got {p.shape[0]}.")

    # 2D case (dim x N)
    if p.ndim == 2:
        if p.shape[0] == dim:
            return p
        if p.shape[0] == dim + 1:
            w = p[-1:]
            if np.any(np.abs(w) < eps):
                raise ValueError("Some homogeneous w are ~0; cannot dehomogenize.")
            return p[:-1] / w
        raise ValueError(f"Expected shape ({dim}, N) or ({dim+1}, N), got {p.shape}.")

    raise ValueError(f"Expected 1D or 2D array, got ndim={p.ndim}.")

def ensure_hom(p, dim):
    """
    Return hom coords of dimension `dim+1`.

    Accepts:
      - (dim, N) or (dim,)       inhom
      - (dim+1, N) or (dim+1,)   hom

    Returns:
      - (dim+1, N) or (dim+1,)
    """
    p = np.asarray(p)

    # 1D case
    if p.ndim == 1:
        if p.shape[0] == dim + 1:
            return p
        if p.shape[0] == dim:
            return np.append(p, 1.0)
        raise ValueError(f"Expected length {dim} or {dim+1}, got {p.shape[0]}.")

    # 2D case
    if p.ndim == 2:
        if p.shape[0] == dim + 1:
            return p
        if p.shape[0] == dim:
            ones = np.ones((1, p.shape[1]), dtype=p.dtype)
            return np.vstack((p, ones))
        raise ValueError(f"Expected shape ({dim}, N) or ({dim+1}, N), got {p.shape}.")

    raise ValueError(f"Expected 1D or 2D array, got ndim={p.ndim}.")

def projectpoints(K, R, t, Q, distCoeffs=None):
    """
    K: camera matrix
    (R, t): pose of the camera
    Q: 3 x n matrix; n points in 3D to be projected into the camera
    """
    Q = ensure_inhom(Q, dim=3)
    
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
    q1 = ensure_inhom(q1, dim=2)
    q2 = ensure_hom(q2, dim=2)

    assert q1.shape[1] == q2.shape[1]

    if normalize:
        T1 = normalize2d(q1)
        T2 = normalize2d(q2)
        q1 = Pi(T1 @ PiInv(q1))
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

def normalize2d(p, debug=False):
    """p is inhom."""
    p = ensure_inhom(p, dim=2)

    mu = np.mean(p, axis=1)
    sig = np.std(p, axis=1)
    Tinv = np.array([
        [sig[0], 0, mu[0]],
        [0, sig[1], mu[1]],
        [0, 0, 1]
    ])
    T = np.linalg.inv(Tinv)

    if debug:
        assert np.all(np.isclose(np.mean(Pi(T @ PiInv(p)), axis=1), 0))
        assert np.all(np.isclose(np.std(Pi(T @ PiInv(p)), axis=1), 1))
    return T

def warpImage(im, H): # Given
    imWarp = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
    return imWarp

def CrossOp(p):
    assert p.shape[0] == 3
    assert p.size == 3
    p = p.ravel() # otherwise if (3,1), indexing gives a 1x1 array instead of a scalar
    p_cross = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ], dtype=float)
    return p_cross

def ref2ref(R1, t1, R2, t2):
    """R, t maps from the reference frame of camera one to the reference frame of camera two (their relative transformation)"""
    R = R2 @ R1.T
    t = t2 - R2 @ R1.T @ t1
    return R, t

def essential_matrix(R, t):
    E = CrossOp(t) @ R
    return E

def fundamental_matrix(R, t, K1, K2):
    """F = K2^(-T) @ E @ K1^(-1) in q2.T @ F @ q1 = 0"""
    E = essential_matrix(R, t)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

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
        q = ensure_inhom(q, dim=2)
        assert P.shape == (3, 4)

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
    q = ensure_hom(q, dim=2)
    Q = ensure_hom(Q, dim=3)

    if normalize:
        T = normalize2d(q)
        q = T @ q
    
    B = []
    q = q / q[-1]
    Q = Q / Q[-1]
    for i in range(Q.shape[1]):
        Bi = np.kron(Q[:, i], CrossOp(q[:, i]))
        assert Bi.shape == (3, 12)
        B.append(Bi)
    
    B = np.concatenate(B, axis=0)

    U, S, VT = np.linalg.svd(B)
    P = VT[-1].reshape((4, 3)).T

    print('max and min singular values:', S.max(), S.min())

    if normalize:
        P = np.linalg.inv(T) @ P
    return P

def checkerboard_points(n, m):
    """
    Returns the points Q_ij = [i-(n-1)/2, j-(m-1)/2, 0]
        for i=0,...,n-1 and j=0,...,m-1
    out_shape is (3, n*m)
    """
    i = np.arange(n) - (n - 1) / 2
    j = np.arange(m) - (m - 1) / 2

    # changed to match convention of cv2.findChessboardCorners, which corresponds to row major flattening of all the points
    x = np.tile(i, m)
    y = np.repeat(j, n)
    z = np.zeros(n * m)

    return np.vstack((x, y, z))