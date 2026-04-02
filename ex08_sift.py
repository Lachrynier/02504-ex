from utils import *; importlib.reload(sys.modules['utils']); from utils import *
np.set_printoptions(suppress=True)


im = cv2.imread('data/sunflowers.jpg')

# E8.4

def transformIm(im, theta, s):
    c = (im.shape[1] // 2, im.shape[0] // 2)
    M = cv2.getRotationMatrix2D(c, theta, s)
    r_im = cv2.warpAffine(im, M, im.shape[:2][::-1])
    return r_im

r_im = transformIm(im, theta=30, s=1.5)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(im[:, :, ::-1])
ax[1].imshow(r_im[:, :, ::-1])
plt.show()

# E8.5
im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(r_im, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

ratio_test = 0.75
good = []
for m, n in matches:
    if m.distance < ratio_test*n.distance:
        good.append([m])

good = sorted(good, key=lambda x: x[0].distance)

im_matches = cv2.drawMatchesKnn(
    im1,kp1,im2,kp2,good[:30],None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)# + cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, ax = plt.subplots(figsize=(14,7))
plt.imshow(im_matches[:, :, ::-1])
plt.show()


im_kp1 = cv2.drawKeypoints(im1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(im_kp1[:, :, ::-1])
plt.show()