from utils import *
import importlib, sys, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
K = np.loadtxt('data/Glyp/K.txt')

image_paths = sorted(glob.glob('data/Glyp/sequence/*.png'))
ims = [cv2.imread(p) for p in image_paths]
N = len(ims)

# ----------------------------
# Feature extraction
# ----------------------------
sift = cv2.SIFT_create(nfeatures=2000)

kps = []
dess = []
kp_arrs = []

for im in ims:
    kp, des = sift.detectAndCompute(im, None)
    kps.append(kp)
    dess.append(des)
    kp_arrs.append(np.array([k.pt for k in kp]))


# ----------------------------
# Matching
# ----------------------------
def knnMatch(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append([m])

    good = sorted(good, key=lambda x: x[0].distance)
    arr = np.array([(m[0].queryIdx, m[0].trainIdx) for m in good], dtype=int)
    return good, arr


matches_arr = []
for i in range(N - 1):
    _, m_arr = knnMatch(dess[i], dess[i + 1])
    matches_arr.append(m_arr)


# ----------------------------
# Geometry helpers
# ----------------------------
def essentialMat(kp1_arr, kp2_arr, m_arr, K):
    p1 = kp1_arr[m_arr[:, 0]]
    p2 = kp2_arr[m_arr[:, 1]]
    E, mask = cv2.findEssentialMat(p1, p2, K)
    mask = (mask == 1).ravel()
    return E, mask


def recoverPose(kp1_arr, kp2_arr, m_arr, K, E):
    p1 = kp1_arr[m_arr[:, 0]]
    p2 = kp2_arr[m_arr[:, 1]]
    _, R, t, mask = cv2.recoverPose(E, p1, p2, K)
    mask = (mask == 255).ravel()
    return R, t, mask


def projection_matrix(R, t, K):
    return K @ np.column_stack((R, t))


# ----------------------------
# Initialize first two cameras
# ----------------------------
E01, E01_mask = essentialMat(kp_arrs[0], kp_arrs[1], matches_arr[0], K)
R01, t01, pose_mask = recoverPose(kp_arrs[0], kp_arrs[1], matches_arr[0], K, E01)

inlier_matches_01 = matches_arr[0][E01_mask & pose_mask]

Rs = [np.eye(3), R01]
ts = [np.zeros((3, 1)), t01]

all_Q = []
all_Q_inliers = []

# ----------------------------
# Main loop (Exercise 11.5)
# ----------------------------
for i in range(2, N):

    # Use inliers only for first step, then full matches
    if i == 2:
        m_prev = inlier_matches_01
    else:
        m_prev = matches_arr[i - 2]

    m_curr = matches_arr[i - 1]

    # ---- Exercise 11.3: track across 3 frames
    _, idx_prev, idx_curr = np.intersect1d(
        m_prev[:, 1],
        m_curr[:, 0],
        return_indices=True
    )

    if len(idx_prev) < 6:
        print(f"Skipping frame {i}: too few 3-frame tracks ({len(idx_prev)})")
        Rs.append(Rs[-1])
        ts.append(ts[-1])
        continue

    # Build correspondences
    pts_a = np.empty((len(idx_prev), 2))
    pts_b = np.empty((len(idx_prev), 2))
    pts_c = np.empty((len(idx_prev), 2))

    for k, (j_prev, j_curr) in enumerate(zip(idx_prev, idx_curr)):
        i_a, i_b = m_prev[j_prev]
        i_c = m_curr[j_curr, 1]

        pts_a[k] = kp_arrs[i - 2][i_a]
        pts_b[k] = kp_arrs[i - 1][i_b]
        pts_c[k] = kp_arrs[i][i_c]

    # ---- Exercise 11.4: triangulation
    P_a = projection_matrix(Rs[i - 2], ts[i - 2], K)
    P_b = projection_matrix(Rs[i - 1], ts[i - 1], K)

    Q = np.vstack([
        Pi(triangulate([pts_a[j], pts_b[j]], [P_a, P_b]))
        for j in range(len(pts_a))
    ])

    # ---- Estimate pose of frame i
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        Q.astype(np.float32),
        pts_c.astype(np.float32),
        K,
        np.zeros(5)
    )

    if not ok or inliers is None or len(inliers) < 6:
        print(f"Skipping frame {i}: PnP failed")
        Rs.append(Rs[-1])
        ts.append(ts[-1])
        continue

    R_i, _ = cv2.Rodrigues(rvec)
    t_i = tvec.reshape(3, 1)

    Rs.append(R_i)
    ts.append(t_i)

    inliers = inliers.flatten()
    all_Q.append(Q)
    all_Q_inliers.append(Q[inliers])


# ----------------------------
# Visualization in camera 1 frame
# ----------------------------
def world_to_cam1_points(X, R1, t1):
    return (R1 @ X.T + t1).T


def camera_center(R, t):
    return -R.T @ t


def world_to_cam1_camera(R, t, R1, t1):
    C = camera_center(R, t)
    return (R1 @ C + t1).ravel()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Transform points
for Q_in in all_Q_inliers:
    Q_cam1 = world_to_cam1_points(Q_in, Rs[1], ts[1])
    ax.scatter(Q_cam1[:, 0], Q_cam1[:, 1], Q_cam1[:, 2], s=2)

# Plot cameras
for i, (R, t) in enumerate(zip(Rs, ts)):
    C = world_to_cam1_camera(R, t, Rs[1], ts[1])
    ax.scatter(*C)
    ax.text(*C, f"cam_{i}")

ax.set_xlabel("X (cam1)")
ax.set_ylabel("Y (cam1)")
ax.set_zlabel("Z (cam1)")

# Optional visual rotation (not geometric)
ax.view_init(elev=45, azim=45)

plt.show()