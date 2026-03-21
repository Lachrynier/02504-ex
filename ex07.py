from skimage.transform import hough_line, hough_line_peaks

from utils import *; importlib.reload(sys.modules['utils']); from utils import *
np.set_printoptions(suppress=True)

DATA_ROOT = Path('data/week06_data')

# E7.1
im = cv2.imread(DATA_ROOT / 'Box3.bmp')
im_canny = cv2.Canny(im, 150, 200) # thresh2=strong edges, thresh1=weak edges

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(im[:, :, ::-1])
ax[1].imshow(im_canny)
plt.show()

# E7.2
hspace, angles, dists = hough_line(im_canny)
plt.imshow(hspace.T); plt.show()

# E7.3
fig, axd = plt.subplot_mosaic([
    ['A', 'B'],
    ['C', 'D']
], figsize=(12, 8))

extent = [angles[0], angles[-1], dists[-1], dists[0]]
axd['A'].imshow(hspace, extent=extent, aspect='auto')

# E7.4
extH, extAngles, extDists = hough_line_peaks(hspace, angles, dists, num_peaks=np.inf)
axd['B'].imshow(hspace, extent=extent, aspect='auto')
axd['B'].plot(extAngles, extDists, '.r')

# E7.5
axd['C'].imshow(im[:, :, ::-1])
plt.sca(axd['C'])
for th, r in zip(extAngles, extDists):
    l = np.array([np.cos(th), np.sin(th), -r])
    DrawLine(l, im.shape[:2])

axd['D'].imshow(im_canny)
plt.sca(axd['D'])
for th, r in zip(extAngles, extDists):
    l = np.array([np.cos(th), np.sin(th), -r])
    DrawLine(l, im.shape[:2])

plt.show()

# E7.6
def test_points(n_in, n_out):
    a = (np.random.rand(n_in)-.5)*10
    b = np.vstack((a, a*.5+np.random.randn(n_in)*.25))
    points = np.hstack((b, 2*np.random.randn(2, n_out)))
    return np.random.permutation(points.T).T

n_in = 40
n_out = 10
points = test_points(n_in, n_out)

plt.plot(*points, '.'); plt.show()

def line_from_two_points(p1, p2):
    p1 = ensure_hom(p1, 2)
    p2 = ensure_hom(p2, 2)
    l = np.cross(p1, p2)
    return l

# E7.7
def compute_inlier_mask(points, l, tau):
    # normalized line
    l_norm = l / np.linalg.norm(l[:2])
    inlier_mask = np.abs(l_norm @ PiInv(ensure_inhom(points, 2))) < tau
    return inlier_mask

# E7.8
def compute_consensus(points, l, tau):
    inlier_mask = compute_inlier_mask(points, l, tau)
    return np.sum(inlier_mask)

# E7.9
def sample_two_points(points):
    idx = np.random.choice(points.shape[1], size=2, replace=False)
    p1, p2 = points[:, idx].T
    return p1, p2

# E7.10-13
def pca_line(x): # assumes x is a (2 x n) array of points
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l@x.mean(1)))
    return l

def line_RANSAC(points, tau, p, epsilon):
    N = np.ceil(np.log(1 - p) / np.log(1 - (1 - epsilon)**2)).astype(int)
    best = {'l': None, 'consensus': -1}
    for _ in range(N):
        p1, p2 = sample_two_points(points)
        l = line_from_two_points(p1, p2)
        consensus = compute_consensus(points, l, tau)
        if consensus > best['consensus']:
            best['l'] = l
            best['consensus'] = consensus
    
    inlier_mask = compute_inlier_mask(points, best['l'], tau)
    l = pca_line(points[:, inlier_mask])
    consensus = compute_consensus(points, l, tau)

    return l, consensus

tau = 0.75
l, consensus = line_RANSAC(points, tau=tau, p=0.99, epsilon=n_out/(n_in+n_out))

def plot_ransac_line(l, tau=None, ax=None, draw_band=True):
    if ax is None:
        ax = plt.gca()

    l = np.asarray(l, dtype=float)
    l = l / np.linalg.norm(l[:2])
    a, b, c = l

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # --- helper: get two endpoints for a line ax + by + c = 0 ---
    def line_segment(a, b, c):
        if abs(b) > 1e-8:
            xs = np.array([x_min, x_max])
            ys = -(a * xs + c) / b
        else:
            ys = np.array([y_min, y_max])
            xs = np.full_like(ys, -c / a)
        return xs, ys

    # --- main line ---
    xs, ys = line_segment(a, b, c)
    ax.plot(xs, ys, color='green', linewidth=1.2)

    if tau is not None:
        # offset lines: shift c → c ± tau (since line is normalized)
        xs_p, ys_p = line_segment(a, b, c - tau)
        xs_m, ys_m = line_segment(a, b, c + tau)

        # boundary lines
        ax.plot(xs_p, ys_p, '-', color='grey', linewidth=1)
        ax.plot(xs_m, ys_m, '-', color='grey', linewidth=1)

        if draw_band:
            if abs(b) > 1e-8:
                ax.fill_between(xs_p, ys_m, ys_p, color='green', alpha=0.2)
            else:
                ax.fill_betweenx(ys_p, xs_m, xs_p, color='green', alpha=0.2)

plt.plot(*points, '.')
plot_ransac_line(l, tau=tau)
plt.show()
