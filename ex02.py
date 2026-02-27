import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from utils import *

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