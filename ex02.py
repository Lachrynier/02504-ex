import sys
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
        p = PiInv(p)

    mu = np.mean(p, axis=1)
    sig = np.std(p, axis=1)
    Tinv = np.array([
        [sig[0], 0, mu[0]],
        [0, sig[1], mu[1]],
        [0, 0, 1]
    ])
    T = np.linalg.inv(Tinv)
    return T

def warpImage(im, H):
    imWarp = cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))
    return imWarp

# E2.1
f = 600
alpha = 1
beta = 0
delta_x = 400
delta_y = 400

R = np.eye(3)
t = np.array([0,0.2,1.5])

K = np.array([
    [f, beta*f, delta_x],
    [0, alpha*f, delta_y],
    [0, 0, 1]
])

P = box3d(n=16)
q = Pi(projectpoints(K, R, t, P))


print(np.sum((q[0] > 800) | (q[1] > 800)))


P1 = np.array([-0.5,-0.5,-0.5])
q1 = Pi(projectpoints(K, R, t, P1[:,None]))
print(q1)

plt.figure()
plt.scatter(*q)
plt.scatter(*q1, color='red')
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(*P)
ax.scatter(*(np.column_stack((R, t)) @ PiInv(P)), color='red')
ax.set_aspect('equal', adjustable='box')
plt.show()

# E2.2
k = [-0.2]
q = Pi(projectpoints(K, R, t, P, k))
q1 = Pi(projectpoints(K, R, t, P1[:,None], k))
print(q1)

print(np.sum((q[0] > 800) | (q[1] > 800)))

plt.figure()
plt.scatter(*q)
plt.scatter(*q1, edgecolors='red')
plt.show()

# E2.3
k3 = -0.245031
k5 = 0.071524
k7 = -0.00994978
distCoeffs = [k3, k5, k7]

im = cv2.imread('data/gopro_robot.jpg')[:,:,::-1].astype(float) / 255
delta_y, delta_x = tuple(s//2 for s in im.shape[:2])
f = 0.455732 * im.shape[1]
alpha = 1
beta = 0

K = np.array([
    [f, beta*f, delta_x],
    [0, alpha*f, delta_y],
    [0, 0, 1]
])
print(K)

# E2.4
im_undist = undistortImage(im, K, distCoeffs)
fig,ax = plt.subplots(1,2,figsize=(12,6))
ax[0].imshow(im)
ax[1].imshow(im_undist)
plt.show()

# E2.5
q2 = np.array([[1,0,2,2],[1,3,3,4]], dtype=float)
H = np.array([
    [-2,0,1],
    [1,-2,0],
    [0,0,3]
], dtype=float)
q1 = Pi(H @ PiInv(q2))
print(q1)

# E2.6
H_est = hest(q1, q2)
print(f'||H||_F: {np.linalg.norm(H_est)}')
with np.printoptions(precision=3, suppress=True):
    print(H_est)
    print(H_est * (H[-1,-1]/H_est[-1,-1]))
print(Pi(H_est @ PiInv(q2)))

# E2.7
T = normalize2d(q2)

# E2.8
H_est = hest(q1, q2, normalize=True)
with np.printoptions(precision=3, suppress=True):
    print(H_est)
    print(H_est * (H[-1,-1]/H_est[-1,-1]))

# E2.9
q2 = np.random.randn(2, 100)
q2h = np.vstack((q2, np.ones((1, 100))))
H_true = np.random.randn(3,3)
q1h = H_true@q2h
q1 = Pi(q1h)

print(f'H_true:\n', H_true)

H_est = hest(q1, q2, normalize=False)
with np.printoptions(precision=9, suppress=True):
    print(f'H_est (non-normalized):\n', H_est * (H_true[-1,-1]/H_est[-1,-1]))

H_est = hest(q1, q2, normalize=True)
with np.printoptions(precision=9, suppress=True):
    print(f'H_est (normalized):\n', H_est * (H_true[-1,-1]/H_est[-1,-1]))

# E2.10
im1 = cv2.imread('data/paper1.jpg')[:,:,::-1].astype(float) / 255
im2 = cv2.imread('data/paper2.jpg')[::-1,::-1,::-1].astype(float) / 255

# plt.imshow(im1); plt.show()
# plt.imshow(im2); plt.show()

# plt.imshow(im1); p = plt.ginput(4); print(repr(np.array(p))); plt.close()
# plt.imshow(im2); p = plt.ginput(4); print(repr(np.array(p))); plt.close()

q1 = np.array([[2358.89851719,  861.21508185],
       [ 581.66211375, 2198.21861472],
       [ 997.43760263, 3099.06550729],
       [2481.18542569, 2173.76123302]]).T
q2 = np.array([[2611.62479475, 2088.16039707],
       [ 866.99823357, 1370.74386724],
       [ 324.85960591, 1880.27265263],
       [1482.50900632, 2630.29902473]]).T

H_est = hest(q1, q2, normalize=True)
q1_est = Pi(H_est @ PiInv(q2))

fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(im1)
ax[0].plot(*q1_est, 'r.', markersize=8)
ax[1].imshow(im2)
ax[1].plot(*q2, 'r.', markersize=8)
plt.show()

# testing
# get the points
# plt.imshow(im2); p2 = np.array(plt.ginput(3)).T; print(repr(p2)); plt.close()

p2 = np.array([[ 614.27195601, 2415.96574116, 1213.47780763],
       [2944.16875653, 1582.70784197,  926.43476638]])

p1 = Pi(H_est @ PiInv(p2))

fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].imshow(im1)
ax[0].plot(*p1, 'r.', markersize=8)
ax[1].imshow(im2)
ax[1].plot(*p2, 'r.', markersize=8)
plt.show()

# E2.11
im1_est = warpImage(im2, H_est)
plt.imshow(im1_est)
plt.show()