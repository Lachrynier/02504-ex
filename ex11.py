from utils import *; importlib.reload(sys.modules['utils']); from utils import *

import cv2
import numpy as np

K = np.loadtxt('data/Glyp/K.txt')

im0, im1, im2 = [cv2.imread(f'data/Glyp/sequence/00000{i+1}.png') for i in range(3)]

# E11.1
sift = cv2.SIFT_create(nfeatures=2000)
kp0, des0 = sift.detectAndCompute(im0, None)
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

kp0_arr, kp1_arr, kp2_arr = [np.array([k.pt for k in kp]) for kp in [kp0, kp1, kp2]]

def knnMatch(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    
    matches = sorted(good, key=lambda x: x[0].distance)
    matches_arr = np.array([(m[0].queryIdx, m[0].trainIdx) for m in matches])
    return matches, matches_arr

def matchesImg(im1, kp1, im2, kp2, matches, num=100, show=True):
    im_matches = cv2.drawMatchesKnn(
        im1, kp1, im2, kp2, matches[:num], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    fig, ax = plt.subplots(figsize=(14,7))
    plt.imshow(im_matches[:, :, ::-1])
    plt.show()
    return im_matches

m01, m01_arr = knnMatch(des0, des1)
m12, m12_arr = knnMatch(des1, des2)

matchesImg(im0, kp0, im1, kp1, m01)
matchesImg(im1, kp1, im2, kp2, m12)

# E11.2
def essentialMat(kp1_arr, kp2_arr, m_arr, K):
    p1 = kp1_arr[m_arr[:, 0]]
    p2 = kp2_arr[m_arr[:, 1]]
    E, mask = cv2.findEssentialMat(p1, p2, K)
    mask = (mask == 1).ravel()
    return E, mask

def recoverPose(kp1_arr, kp2_arr, m_arr, K, E):
    p1 = kp1_arr[m_arr[:, 0]]
    p2 = kp2_arr[m_arr[:, 1]]
    inliers, R, t, mask = cv2.recoverPose(E, p1, p2, K)
    mask = (mask == 255).ravel()
    return inliers, R, t, mask

E01, E01_mask = essentialMat(kp0_arr, kp1_arr, m01_arr, K)
inliers, R1, t1, pose01_mask = recoverPose(kp0_arr, kp1_arr, m01_arr, K, E01)

inlier_m01_arr = m01_arr[E01_mask * pose01_mask]
_, idx01, idx12 = np.intersect1d(inlier_m01_arr[:,1], m12_arr[:,0], return_indices=True)

points0 = np.empty((len(idx01), 2))
points1 = np.empty((len(idx01), 2))
points2 = np.empty((len(idx01), 2))

for k, (i01, i12) in enumerate(zip(idx01, idx12)):
    i0, i1 = inlier_m01_arr[i01]
    i2 = m12_arr[i12, 1]
    points0[k] = kp0_arr[i0]
    points1[k] = kp1_arr[i1]
    points2[k] = kp2_arr[i2]

P0 = K @ np.column_stack((np.eye(3), np.zeros((3, 1))))
P1 = K @ np.column_stack((R1, t1))


Q = np.vstack([
    Pi(triangulate([points0[i], points1[i]], [P0, P1])) for i in range(len(points0))
])

_, R2, t2, inliers = cv2.solvePnPRansac(Q, points2, K, np.zeros(5))
R2 = cv2.Rodrigues(R2)[0]

def plot_cameras(ax, R_list, t_list, colors=None):
    for i, (R, t) in enumerate(zip(R_list, t_list)):
        camera_pos = (-R.T @ t).ravel()
        ax.scatter(*camera_pos)

        text = f"cam_{i}"
        if colors:
            ax.text(*camera_pos, text, colors[i])
        else:
            ax.text(*camera_pos, text)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(*Q[inliers.flatten()].T)

plot_cameras(ax, [np.eye(3), R1, R2], [np.zeros((3, 1)), t1, t2])

plt.show()

