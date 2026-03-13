from scipy.optimize import least_squares
import glob
import math

from utils import *; importlib.reload(sys.modules['utils']); from utils import *
np.set_printoptions(suppress=True)

R1 = np.eye(3)
R2 = np.eye(3)
t1 = np.array([[0.], [0.], [1.]])
t2 = np.array([[0.], [0.], [20.]])
K1 = np.array([
    [700, 0, 600],
    [0, 700, 400],
    [0, 0, 1],
], dtype=float)
K2 = K1
Q = np.array([[1.], [1.], [0.]])

# E5.1

P1 = K1 @ np.column_stack((R1, t1))
P2 = K2 @ np.column_stack((R2, t2))

q1 = Pi(P1 @ PiInv(Q))
q2 = Pi(P2 @ PiInv(Q))

print(f'q1:\n{q1}')
print(f'q2:\n{q2}')

# E5.2
q1_tilde = q1 + np.array([[1], [-1]])
q2_tilde = q2 + np.array([[1], [-1]])
Q_tilde = Pi(triangulate([q1_tilde, q2_tilde], [P1, P2]))

print(f'Q_tilde: {Q_tilde}')

# E5.3
def triangulate_nonlin(q_list, P_list):
    def compute_residuals(Q):
        f = []
        for q, P in zip(q_list, P_list):
            q = ensure_inhom(q, dim=2).flatten()
            f.append(Pi(P @ PiInv(Q)) - q)
        
        f = np.concatenate(f, axis=0)
        return f
    
    assert len(q_list) == len(P_list)
    for P in P_list:
        assert P.shape == (3, 4)
    
    x0 = Pi(triangulate(q_list, P_list))
    res = least_squares(compute_residuals, x0)
    if not res.success:
        raise RuntimeError
    Q = res.x
    return Q

# E5.4
Q_hat = triangulate_nonlin([q1_tilde, q2_tilde], [P1, P2])

print(f'Q_hat: {Q_hat}')

print(f'Linear error: {np.linalg.norm(Q_tilde - Q.ravel())}')
print(f'Nonlinear error: {np.linalg.norm(Q_hat - Q.ravel())}')

# E5.5-5.7
im = cv2.imread('data/checkerboard/straight_on.jpg')[:, :, ::-1]
im_small = cv2.resize(im, None, fx=0.25, fy=0.25)
patternSize = (7, 10)
ret, corners = cv2.findChessboardCorners(im_small, patternSize, None)

plt.imshow(im_small)
plt.plot(*corners.reshape(-1, 2).T, '.r')
plt.show()

files = glob.glob('data/checkerboard/*.jpg')
cols = 4
rows = math.ceil(len(files) / cols)

fig, ax = plt.subplots(rows, cols, figsize=(16, 4 * rows))
ax = ax.flatten()

plot_idx = 0
images = []

objp = checkerboard_points(*patternSize).T.astype(np.float32)
imgpoints = []
objpoints = []

for f in files:

    im = cv2.imread(f)[:, :, ::-1]
    # print(im.shape)
    im_small = cv2.resize(im, None, fx=0.25, fy=0.25)

    ret, corners = cv2.findChessboardCorners(im_small, patternSize, None)

    if ret:
        images.append(im_small)
        imgpoints.append(corners)
        objpoints.append(objp)

        ax[plot_idx].imshow(im_small)
        ax[plot_idx].plot(*corners.reshape(-1, 2).T, ".r")
        ax[plot_idx].plot(*corners.reshape(-1, 2).T, "-b")
        ax[plot_idx].plot(corners[0,0,0], corners[0,0,1], marker='D', color='lime', markersize=3)
        ax[plot_idx].axis("off")

        plot_idx += 1


# hide remaining unused axes
for j in range(plot_idx, len(ax)):
    ax[j].axis("off")

plt.tight_layout()
plt.show()

# E5.8
# CHECK THAT ORDER OF checkerboard_points MATCHES THE ORDER RETURNED BY cv2.findChessboardCorners (VERY IMPORTANT)
fig, ax = plt.subplots(1, 2)
ax[0].plot(*objp.T[:2])
ax[0].plot(*objp.T[:2, 0], '.r')
imgp = imgpoints[0].squeeze().T
ax[1].plot(*imgp)
ax[1].plot(*imgp[:, 0], '.r')
plt.show()

flags = cv2.CALIB_FIX_K1+cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6+cv2.CALIB_ZERO_TANGENT_DIST
# distCoeffs forced to 0 with these flags
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, im_small.shape[:2][::-1], None, None, flags=flags)
R_list = [cv2.Rodrigues(r)[0] for r in rvecs]
print(cameraMatrix)

# E5.9
reproj =[projectpoints(cameraMatrix, r, t, o.T) for r, t, o in zip(R_list, tvecs, objpoints)]
reproj = [Pi(p) for p in reproj]
errors = [np.linalg.norm(q_pred - q.squeeze().T, axis=0).mean().item() for q_pred, q in zip(reproj, imgpoints)]

print(f'reproj errors (no distortion):\n{errors}')
print(f'total: {sum(errors)}')

cols = 4
rows = math.ceil(len(images) / cols)

fig, ax = plt.subplots(rows, cols, figsize=(16, 4 * rows))
ax = ax.flatten()

for i, (im, q_pred, q) in enumerate(zip(images, reproj, imgpoints)):
    ax[i].imshow(im)
    ax[i].plot(*q.squeeze().T, '.r')
    ax[i].plot(*q_pred, '.g')

plt.tight_layout()
plt.show()

# E5.10

Q = 2*box3d() + 1

idx = 11
Q_proj = projectpoints(cameraMatrix, R_list[idx], tvecs[idx], Q)
Q_proj = Pi(Q_proj)

plt.imshow(images[idx])
plt.plot(*Q_proj, '.r')
plt.show()

# E5.11

########
# removed cv2.CALIB_FIX_K1
flags = cv2.CALIB_FIX_K2+cv2.CALIB_FIX_K3+cv2.CALIB_FIX_K4+cv2.CALIB_FIX_K5+cv2.CALIB_FIX_K6+cv2.CALIB_ZERO_TANGENT_DIST
# distCoeffs forced to 0 with these flags
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, im_small.shape[:2][::-1], None, None, flags=flags)
R_list = [cv2.Rodrigues(r)[0] for r in rvecs]
print(cameraMatrix)

reproj =[projectpoints(cameraMatrix, r, t, o.T, distCoeffs[0]) for r, t, o in zip(R_list, tvecs, objpoints)]
reproj = [Pi(p) for p in reproj]
errors = [np.linalg.norm(q_pred - q.squeeze().T, axis=0).mean().item() for q_pred, q in zip(reproj, imgpoints)]
print(f'reproj errors (distortion, k1):\n{errors}')
print(f'total: {sum(errors)}')

# only bit lower error