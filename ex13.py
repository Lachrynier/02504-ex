from utils import *; importlib.reload(sys.modules['utils']); from utils import *

DATA_PATH = Path('data/casper')

c = np.load(DATA_PATH / 'calib.npy', allow_pickle=True).item()
K0 = c['K0']
K1 = c['K1']
d0 = c['d0']
d1 = c['d1']
R = c['R']
t = c['t']

n1 = 40
n2 = 41

# E13.2
im0 = cv2.imread(DATA_PATH / "sequence/frames0_0.png")
size = (im0.shape[1], im0.shape[0])
stereo = cv2.stereoRectify(c['K0'], c['d0'], c['K1'],
c['d1'], size, c['R'], c['t'], flags=0)
R0, R1, P0, P1 = stereo[:4]
maps0 = cv2.initUndistortRectifyMap(c['K0'], c['d0'], R0, P0, size, cv2.CV_32FC2)
maps1 = cv2.initUndistortRectifyMap(c['K1'], c['d1'], R1, P1, size, cv2.CV_32FC2)

ims0 = []
ims1 = []

for i in range(26):
    im0 = cv2.imread(DATA_PATH / f'sequence/frames0_{i}.png')
    im0 = cv2.cvtColor(im0.astype(np.float32) / 255, cv2.COLOR_BGR2GRAY)
    im1 = cv2.imread(DATA_PATH / f'sequence/frames1_{i}.png')
    im1 = cv2.cvtColor(im1.astype(np.float32) / 255, cv2.COLOR_BGR2GRAY)
    im0 = cv2.remap(im0, *maps0, cv2.INTER_LINEAR)
    im1 = cv2.remap(im1, *maps1, cv2.INTER_LINEAR)
    ims0.append(im0)
    ims1.append(im1)

ims0 = np.array(ims0)
ims1 = np.array(ims1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(ims0[0], cmap='gray')
ax[1].imshow(ims1[0], cmap='gray')
plt.show()

# E13.3
def unwrap(ims):
    primary = ims[2:18]
    secondary = ims[18:26]
    fft_primary = np.fft.rfft(primary, axis=0)
    theta_primary = np.angle(fft_primary[1])
    fft_secondary = np.fft.rfft(secondary, axis=0)
    theta_secondary = np.angle(fft_secondary[1])
    theta_c = (theta_secondary - theta_primary) % (2*np.pi)
    order_primary = np.round((n1*theta_c - theta_primary) / (2*np.pi))
    theta_est = (2*np.pi * order_primary + theta_primary) / n1
    return theta_est

theta0 = unwrap(ims0)
theta1 = unwrap(ims1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
c0 = ax[0].imshow(theta0)
fig.colorbar(c0, ax=ax[0])
c1 = ax[1].imshow(theta1)
fig.colorbar(c1, ax=ax[1])
plt.show()


# E13.4
for obj in [ims0, ims1]:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    c0 = ax[0].imshow(obj[0])
    fig.colorbar(c0, ax=ax[0])
    c1 = ax[1].imshow(obj[1])
    fig.colorbar(c1, ax=ax[1])
    plt.show()

thresh1 = 15 / 255
thresh2 = 15 / 255
mask0 = (ims0[0] - ims0[1]) > thresh1
mask1 = (ims1[0] - ims1[1]) > thresh2

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(mask0)
ax[1].imshow(mask1)
plt.show()

# E13.5
disparity = np.zeros(theta0.shape)
q0s = []
q1s = []

infs = np.array([np.inf, -np.inf])
for i0 in range(mask0.shape[0]):
    for j0 in range(mask0.shape[1]):
        if not mask0[i0, j0]:
            continue
        
        closeness = np.abs(theta0[i0, j0] - theta1[i0])
        closeness[~mask1[i0]] = np.inf
        j1 = np.argmin(closeness)
        disparity[i0, j0] = j0 - j1

        q0s.append([j0, i0])
        q1s.append([j1, i0])

q0s = np.array(q0s).T.astype(np.float32)
q1s = np.array(q1s).T.astype(np.float32)

fig, ax = plt.subplots()

# cim = ax.imshow(disparity)
cim = ax.imshow(cv2.medianBlur(disparity.astype(np.float32), 5))
fig.colorbar(cim, ax=ax)
plt.show()

# E13.6
Q_all = Pi(cv2.triangulatePoints(P0, P1, q0s, q1s))

valid = Q_all[2] > 0
Q = Q_all[:, valid]

# https://github.com/isl-org/Open3D/issues/6872#issuecomment-2615797521
# export XDG_SESSION_TYPE=x11
import os
os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(Q.T)
o3d.visualization.draw_geometries([pcd])

# E13.7
q0s_valid = q0s[:, valid]
q1s_valid = q1s[:, valid]

im0_color = cv2.imread(DATA_PATH / "sequence/frames0_0.png")
im0_color = cv2.cvtColor(im0_color, cv2.COLOR_BGR2RGB)
im0_color = im0_color.astype(np.float32) / 255.0
im0_color_rect = cv2.remap(im0_color, *maps0, cv2.INTER_LINEAR)

# q0s_valid has shape 2 x N:
# q0s_valid[0] = x coordinates / columns
# q0s_valid[1] = y coordinates / rows
xs = q0s_valid[0].astype(int)
ys = q0s_valid[1].astype(int)

# Sample RGB colors at the matched image locations
colors = im0_color_rect[ys, xs]

# Create a colored point cloud
pcd_colored = o3d.geometry.PointCloud()
pcd_colored.points = o3d.utility.Vector3dVector(Q.T)
pcd_colored.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd_colored])