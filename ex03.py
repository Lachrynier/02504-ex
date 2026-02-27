from scipy.spatial.transform import Rotation

from utils import *; importlib.reload(sys.modules['utils']); from utils import *

# %load_ext autoreload
# %autoreload 2

### Epipolar geometry
K1 = np.array([
    [1000, 0, 300],
    [0, 1000, 200],
    [0, 0, 1],
], dtype=float)
K2 = K1

R1 = np.eye(3)
t1 = np.zeros(3)

R2 = Rotation.from_euler('xyz', [0.7, -0.5, 0.8]).as_matrix()
t2 = np.array([0.2, 2, 1])


# E3.1
Q = np.array([1, 0.5, 4, 1])

q1 = Pi(projectpoints(K1, R1, t1, Q))
q2 = Pi(projectpoints(K2, R2, t2, Q))

print(f'q1: {q1}')
print(f'q2: {q2}')

# E3.2
p1, p2 = np.random.rand(2, 3)
print(f'CrossOp: {CrossOp(p1) @ p2}')
print(f'np.cross: {np.cross(p1, p2)}')

# E3.3
# "R, t maps from the reference frame of camera one to the reference frame of camera two (their relative transformation)"
# p1 = R1 @ Q + t1 maps from world reference frame to camera 1 reference frame. we want to go from camera 1 to world, and then follow that by world to camera 2
# Q = np.linalg.inv(R1) @ (p1 - t1) = R1.T @ (p1 - t1)
# p2 = R2 @ Q + t2 = R2 @ (R1.T @ (p1 - t1)) + t2 = R2 @ R1.T @ p1 - R2 @ R1.T @ t1 + t2

R = R2 @ R1.T
t = -R2 @ R1.T @ t1 + t2
E = CrossOp(t) @ R # essential matrix
F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1) # fundamental matrix
print(f'F:\n{F}')

# E3.4
l = F @ PiInv(q1)
print(f'l: {-5.285/l[-1] * l}')

# E3.5
# They are projections of the same point in space,
# so since there are no errors in our example,
# the theory dictates why it is the case.
print(f'q2.T @ l = {PiInv(q2) @ l}')



# E3.8-3.10
# E3.8
data = np.load('data/TwoImageDataCar.npy', allow_pickle=True).item()
im1 = data['im1']
R1 = data['R1']
t1 = data['t1']
im2 = data['im2']
R2 = data['R2']
t2 = data['t2']
K1 = data['K']
K2 = K1

def e9_and_10():
    # E3.9
    R = R2 @ R1.T
    t = -R2 @ R1.T @ t1 + t2
    E = CrossOp(t) @ R # essential matrix
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1) # fundamental matrix
    print(f'F:\n{0.01174/F[-1,-1] * F}')

    plt.imshow(im1); q1 = np.array(plt.ginput(1)).T; print(repr(q1)); plt.close()

    print(f'Clicked q1: {q1}')

    l2 = F @ PiInv(q1)
    print(f'l2: {l2}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im1)
    ax[0].plot(*q1, 'gx')
    ax[1].imshow(im2)
    plt.sca(ax[1])
    DrawLine(l2, im2.shape)
    plt.show()

    # E3.10
    plt.imshow(im2); q2 = np.array(plt.ginput(1)).T; print(repr(q2)); plt.close()

    print(f'Clicked q2: {q2}')

    l1 = PiInv(q2).T @ F
    print(f'l2: {l2}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im1)
    plt.sca(ax[0])
    DrawLine(l1, im1.shape)
    ax[1].imshow(im2)
    ax[1].plot(*q2, 'gx')
    plt.show()

# e9_and_10()

# E3.11
# q1 = P1 @ Q = K1 @ (R1 @ Q + t1) = K1 @ R1 @ Q + K1 @ t1
# Q = np.linalg.inv(K1 @ R1) @ (q1 - K1 @ t1)
q1 = np.array([[338.74561933], [255.69887482]]).squeeze()
Q = np.linalg.inv(K1 @ R1) @ (PiInv(q1) - K1 @ t1)
print(Pi(projectpoints(K1, R1, t1, Q))) # sanity check

p1 = R1 @ Q + t1
p1_line = p1[:,None] * np.linspace(1, 2)
Q_line = np.linalg.inv(R1) @ (p1_line - t1[:,None])
q_line = Pi(projectpoints(K2, R2, t2, Q_line))
plt.plot(*q_line)
plt.show()


Q_true = Q_line[:,-1]
q2 = Pi(projectpoints(K2, R2, t2, Q_true))

P1 = K1 @ np.column_stack((R1, t1))
P2 = K2 @ np.column_stack((R2, t2))
Q_est = triangulate((q1, q2), (P1, P2))

print(f'Q_true: {Q_true}')
print(f'Q_est: {Pi(Q_est)}')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(im1)
ax[0].plot(*q1, 'gx')
ax[1].imshow(im2)
ax[1].plot(*q2, 'gx')
plt.show()

# recovers Q_true successfully,
# but not the same points on the image
# this is due to the 3D point having different depth than
# the physical point marked on the image, as it might be
# behind or in front of it.