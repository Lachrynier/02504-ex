from utils import *; importlib.reload(sys.modules['utils']); from utils import *

# E9.1

def Fest_8point(q1, q2):
    q1 = ensure_hom(q1, dim=2)
    q2 = ensure_hom(q2, dim=2)
    assert q1.shape == q2.shape
    N = q1.shape[1]
    assert N >= 8

    B = []
    for i in range(N):
        Bi = (q2[:, [i]] @ q1[:, [i]].T).flatten()
        B.append(Bi)
    
    B = np.stack(B, axis=0)
    U, S, VT = np.linalg.svd(B)
    F = VT[-1, :].reshape((3, 3))
    return F

data = np.load('data/Fest_test.npy', allow_pickle=True).item()
q1 = data['q1']
q2 = data['q2']
Ftrue = data['Ftrue']

Fest = Fest_8point(q1, q2)

err = Fest / Fest[-1, -1] - Ftrue / Ftrue[-1, -1]
print('E9.1:\n', repr(err), sep="")

# E9.2
data = np.load('data/TwoImageData.npy', allow_pickle=True).item()
im1 = data['im1']
R1 = data['R1']
t1 = data['t1']
im2 = data['im2']
R2 = data['R2']
t2 = data['t2']
K1 = data['K']
K2 = K1

fig, ax = plt.subplots(1, 2)
ax[0].imshow(im1)
ax[1].imshow(im2)
plt.show()

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

bf = cv2.BFMatcher(crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

im_matches = cv2.drawMatches(
    im1,kp1,im2,kp2,matches[:50],None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig, ax = plt.subplots(figsize=(14,7))
plt.imshow(im_matches[:, :, ::-1])
plt.show()

pairs = []
for m in matches:
    p1 = kp1[m.queryIdx].pt
    p2 = kp2[m.trainIdx].pt
    pairs.append((p1, p2))

def plot_matches(im1, im2, pairs, line_color='r', linewidth=0.5, show_points=True):
    h1, w1 = im1.shape
    h2, w2 = im2.shape

    combined = np.zeros((max(h1, h2), w1 + w2), dtype=im1.dtype)
    combined[:h1, :w1] = im1
    combined[:h2, w1:w1 + w2] = im2

    plt.figure(figsize=(10, 5))
    plt.imshow(combined)

    for (x1, y1), (x2, y2) in pairs:
        x2 += w1
        plt.plot([x1, x2], [y1, y2], color=line_color, linewidth=linewidth)
        if show_points:
            plt.scatter([x1, x2], [y1, y2], c=line_color, s=5)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_matches(im1, im2, pairs[:50])
plot_matches(im1, im2, pairs[-50:])

# E9.3
def SampsonsDistance(F, p1, p2):
    numer = (p2.T @ F @ p1)**2
    denom1 = np.sum((p2.T @ F)[:2]**2)
    denom2 = np.sum((F @ p1)[:2]**2)
    d = numer / (denom1 + denom2)
    return d

def fundamental_matrix_RANSAC(pairs, N, sigma=3):
    pairs = np.array(pairs)
    q1, q2 = np.transpose(pairs, (1, 2, 0))
    q1 = ensure_hom(q1, dim=2)
    q2 = ensure_hom(q2, dim=2)

    tau = 3.84 * sigma**2
    best = {'F': None, 'consensus': -1}
    for _ in range(N):
        idx = np.random.choice(len(pairs), 8, replace=False)
        F = Fest_8point(q1[:, idx], q2[:, idx])
        d = np.array([SampsonsDistance(F, q1[:, i], q2[:, i]) for i in range(len(pairs))])
        inlier_mask = d < tau
        consensus = np.sum(inlier_mask)
        if consensus > best['consensus']:
            best['F'] = F
            best['consensus'] = consensus
    
    d = np.array([SampsonsDistance(best['F'], q1[:, i], q2[:, i]) for i in range(len(pairs))])
    inlier_mask = d < tau
    F = Fest_8point(q1[:, inlier_mask], q2[:, inlier_mask])

    d = np.array([SampsonsDistance(F, q1[:, i], q2[:, i]) for i in range(len(pairs))])
    inlier_mask = d < tau
    consensus = np.sum(inlier_mask)

    return F, consensus, inlier_mask

Fest, consensus, inlier_mask = fundamental_matrix_RANSAC(pairs, N=200, sigma=3)

R = R2 @ R1.T
t = -R2 @ R1.T @ t1 + t2
E = CrossOp(t) @ R # essential matrix
Ftrue = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1) # fundamental matrix

# err = Fest / Fest[-1, -1] - Ftrue / Ftrue[-1, -1]
# print('E9.3:\n', repr(err), sep="")

print('E9.3:')
print((Fest*Ftrue).sum() / (np.linalg.norm(Fest)*np.linalg.norm(Ftrue)))

# E9.4

pairs_np = np.array(pairs, dtype=object)

inlier_pairs = pairs_np[inlier_mask].tolist()
outlier_pairs = pairs_np[~inlier_mask].tolist()

print('Total matches:', len(pairs))
print('Inliers:', len(inlier_pairs))
print('Outliers:', len(outlier_pairs))
print('Inlier ratio:', len(inlier_pairs) / len(pairs))

print("pairs")
plot_matches(im1, im2, pairs[:50])
print("inlier_pairs")
plot_matches(im1, im2, inlier_pairs)
print("outlier_pairs")
plot_matches(im1, im2, outlier_pairs)