from utils import *; importlib.reload(sys.modules['utils']); from utils import *

im2 = cv2.imread('data/im1.jpg')
im1 = cv2.imread('data/im2.jpg')

# E10.1
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

bf = cv2.BFMatcher(crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

pairs = []
for m in matches:
    p1 = kp1[m.queryIdx].pt
    p2 = kp2[m.trainIdx].pt
    pairs.append((p1, p2))

im_matches = cv2.drawMatches(
    im1,kp1,im2,kp2,matches[:50],None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig, ax = plt.subplots(figsize=(14,7))
plt.imshow(im_matches[:, :, ::-1])
plt.show()

im_matches = cv2.drawMatches(
    im1,kp1,im2,kp2,matches,None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig, ax = plt.subplots(figsize=(14,7))
plt.imshow(im_matches[:, :, ::-1])
plt.show()

# E10.2
def homography_RANSAC(pairs, N, sigma=3):
    def dist_approx(H, Hinv, p1, p2):
        term1 = np.linalg.norm(Pi(p1) - Pi(H @ p2))**2
        term2 = np.linalg.norm(Pi(p2) - Pi(Hinv @ p1))**2
        return np.sqrt(term1 + term2)
    
    pairs = np.array(pairs)
    q1, q2 = np.transpose(pairs, (1, 2, 0))
    q1 = ensure_hom(q1, dim=2)
    q2 = ensure_hom(q2, dim=2)

    tau = 5.99 * sigma**2
    best = {'H': None, 'consensus': -1}
    for _ in range(N):
        idx = np.random.choice(len(pairs), 8, replace=False)
        H = hest(q1[:, idx], q2[:, idx])
        Hinv = np.linalg.inv(H)
        d = np.array([dist_approx(H, Hinv, q1[:, i], q2[:, i]) for i in range(len(pairs))])
        inlier_mask = d < tau
        consensus = np.sum(inlier_mask)
        if consensus > best['consensus']:
            best['H'] = H
            best['consensus'] = consensus
    
    d = np.array([
        dist_approx(best['H'], np.linalg.inv(best['H']), q1[:, i], q2[:, i])
        for i in range(len(pairs))
    ])
    inlier_mask = d < tau
    H = hest(q1[:, inlier_mask], q2[:, inlier_mask])
    Hinv = np.linalg.inv(H)

    d = np.array([dist_approx(H, Hinv, q1[:, i], q2[:, i]) for i in range(len(pairs))])
    inlier_mask = d < tau
    consensus = np.sum(inlier_mask)

    return H, consensus, inlier_mask

H, consensus, inlier_mask = homography_RANSAC(pairs, N=500, sigma=3)

plt.imshow(cv2.drawMatches(im1, kp1, im2, kp2, np.array(matches)[inlier_mask], None)[:, :, ::-1])

print(consensus)

# E10.3

def estHomographyRANSAC(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pairs = []
    for m in matches:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        pairs.append((p1, p2))
    
    H, consensus, inlier_mask = homography_RANSAC(pairs, N=500, sigma=3)
    plt.imshow(cv2.drawMatches(im1, kp1, im2, kp2, np.array(matches)[inlier_mask], None)[:, :, ::-1])
    return H

# E10.4

def warpImage(im, H, xRange, yRange):
    T = np.eye(3)
    T[:2, 2] = [-xRange[0], -yRange[0]]
    H = T@H
    outSize = (xRange[1]-xRange[0], yRange[1]-yRange[0])
    mask = np.ones(im.shape[:2], dtype=np.uint8)*255
    imWarp = cv2.warpPerspective(im, H, outSize)
    maskWarp = cv2.warpPerspective(mask, H, outSize)
    return imWarp, maskWarp

H = estHomographyRANSAC(kp1, des1, kp2, des2)

### E10.6
def autoRanges(im1, im2, H):
    corners = []
    for im in [im1, im2]:
        r, c = im.shape[:2]
        c = np.array([[0,0],[0,r-1],[c-1,0],[c-1,r-1]]).T
        corners.append(c)
    
    corners = np.hstack((corners[0], Pi(H @ PiInv(corners[1]))))
    xRange = [
        int(np.floor(np.min(corners[0]))),
        int(np.ceil(np.max(corners[0])))
    ]
    yRange = [
        int(np.floor(np.min(corners[1]))),
        int(np.ceil(np.max(corners[1])))
    ]
    return xRange, yRange
###
    

# xRange = [0, im1.shape[1]+800]
# yRange = [-300, im1.shape[0]+400]
xRange, yRange = autoRanges(im1, im2, H)

im1Warp, mask1Warp = warpImage(im1, np.eye(3), xRange, yRange)
im2Warp, mask2Warp = warpImage(im2, H, xRange, yRange)

### E10.5
im_stitch = im1Warp.copy()
only2 = ~(mask1Warp > 0) & mask2Warp
im_stitch += only2[:,:,None] * im2Warp
###

fig, ax = plt.subplots(4, 2, figsize=(6,12))
ax[0,0].imshow(im1[:,:,::-1])
ax[0,1].imshow(im2[:,:,::-1])
ax[1,0].imshow(im1Warp[:,:,::-1])
ax[1,1].imshow(mask1Warp)
ax[2,0].imshow(im2Warp[:,:,::-1])
ax[2,1].imshow(mask2Warp)
ax[3,0].imshow(im_stitch[:,:,::-1])
ax[3,1].imshow(only2)
plt.show()
