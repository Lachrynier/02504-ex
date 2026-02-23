import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

### E9-10
print(sys.version)
print(cv2.__version__)

im_path = "data/cat-image.jpg"

im = cv2.imread(im_path)[:,:,::-1]
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

plt.imshow(im)
plt.show()

### E11
import itertools as it
def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

points = box3d(16)
print(points.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*points)
plt.show()

### E12
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

p = np.arange(6).reshape(2,3)
print(p)

ph = PiInv(p)
print(ph)

print(Pi(ph))
print(Pi(3*ph))

### E13
def projectpoints(K, R, t, Q):
    """
    K: camera matrix
    (R, t): pose of the camera
    Q: 3 x n matrix; n points in 3D to be projected into the camera
    """
    return K @ np.column_stack((R, t)) @ PiInv(Q)

Q = box3d(16)
K = np.eye(3)
R = np.eye(3)
t = np.array([0,0,4])

points = Pi(projectpoints(K,R,t,Q))

plt.figure()
plt.scatter(*points)
plt.show()

if 0:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    # ax.view_init(elev=10, azim=20, roll=-90)
    ax.scatter(*points)
    ax.scatter(*(np.column_stack((R, t)) @ PiInv(Q)), color='red')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

### E14
Q = box3d(16)
rot_mat = lambda theta: np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
R = rot_mat(np.deg2rad(30))
t = np.array([0,0,4])

points = Pi(projectpoints(K,R,t,Q))

plt.figure()
plt.scatter(*points)
plt.show()

if 0:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    # ax.view_init(elev=10, azim=20, roll=-90)
    ax.scatter(*points)
    ax.scatter(*(np.column_stack((R, t)) @ PiInv(Q)), color='red')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

### E15 as in sol

Q = box3d(16)
rot_mat = lambda theta: np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
R = rot_mat(np.deg2rad(-30))
t = np.array([0,2,4])

points = Pi(projectpoints(K,R,t,Q))

plt.figure()
plt.scatter(*points)
plt.show()

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(*points)
# plt.show()

### E15 play around

Q = box3d(16)
# K = np.array([[1,0,10],[0,1,10],[0,0,1]])
rot_mat = lambda theta: np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
R = rot_mat(np.deg2rad(0))
t = np.array([5,5,4])

points = Pi(projectpoints(K,R,t,Q))

plt.figure()
plt.scatter(*points)
plt.show()