from scipy.signal import convolve2d
from scipy.ndimage import convolve
from scipy.ndimage import maximum_filter
from skimage.morphology import disk

from utils import *; importlib.reload(sys.modules['utils']); from utils import *
np.set_printoptions(suppress=True)

# E6.1
def gaussian1DKernel(sigma):
    size = np.round(4*sigma).astype(int)
    x = np.arange(-size, size + 1)
    g = np.exp(-x**2 / (2 * sigma**2))
    g = g / g.sum()
    gd = (-x / sigma**2) * g
    return g, gd

# E6.2
def gaussianSmoothing(im, sigma):
    g, gd = gaussian1DKernel(sigma)
    
    # I = convolve2d(convolve2d(im, g[None, :]), g[:, None])
    # Ix = convolve2d(convolve2d(im, gd[None, :]), g[:, None])
    # Iy = convolve2d(convolve2d(im, g[None, :]), gd[:, None])
    
    I = cv2.sepFilter2D(im, -1, kernelX=g, kernelY=g)
    Ix = cv2.sepFilter2D(im, -1, kernelX=gd, kernelY=g)
    Iy = cv2.sepFilter2D(im, -1, kernelX=g, kernelY=gd)
    
    return I, Ix, Iy

DATA_ROOT = Path('data/week06_data')
im = cv2.imread(DATA_ROOT / 'TestIm1.png')[:, :, ::-1].astype(np.float32) / 255
im_grey = cv2.cvtColor(im[:, :, ::-1], cv2.COLOR_BGR2GRAY)

sigma = 5
I, Ix, Iy = gaussianSmoothing(im_grey, sigma)

fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax = ax.flatten()
ax[0].imshow(im)
ax[1].imshow(I)
c2 = ax[2].imshow(Ix)
fig.colorbar(c2, ax=ax[2])
ax[3].imshow(Iy)
plt.show()

# sepFilter2D does cross-correlation, while convolve2d uses convolution. Thus they have flipped signs in results.

# E6.3
def structureTensor(im, sigma, epsilon):
    g_eps, _ = gaussian1DKernel(epsilon)
    I, Ix, Iy = gaussianSmoothing(im, sigma)
    
    Cxx = cv2.sepFilter2D(Ix**2, -1, g_eps, g_eps)
    Cxy = cv2.sepFilter2D(Ix*Iy, -1, g_eps, g_eps)
    Cyy = cv2.sepFilter2D(Iy**2, -1, g_eps, g_eps)
    C = np.zeros((2, 2, *Cxx.shape))
    C[0, 0] = Cxx
    C[0, 1] = Cxy
    C[1, 0] = Cxy
    C[1, 1] = Cyy
    return C

# E6.4
def harrisMeasure(im, sigma, epsilon, k=0.06):
    C = structureTensor(im, sigma, epsilon)
    a = C[0, 0]
    b = C[1, 1]
    c = C[0, 1]
    r = a*b - c**2 - k*(a + b)**2
    return r

# if epsilon is 0 (that is, no filtering is done), then the determinant is always 0
# hence, the response function is always non-negative
epsilon = 5
k = 0.06
r = harrisMeasure(im_grey, sigma, epsilon, k)

plt.imshow(r)
plt.show()

# E6.5
def cornerDetector(im, sigma, epsilon, k, tau_frac=None, tau=None, disk_rad=1):
    r = harrisMeasure(im, sigma, epsilon, k=k)
    if tau is None:
        if tau_frac is None:
            return ValueError()
        tau = tau_frac * r.max()
    M = r > tau
    # footprint = np.array([
    #     [0, 1, 0],
    #     [1, 0, 1],
    #     [0, 1, 0]
    # ], dtype=bool)
    footprint = disk(disk_rad, dtype=bool)
    footprint[disk_rad, disk_rad] = 0
    Nmax = maximum_filter(r, footprint=footprint, mode='constant', cval=-np.inf)
    c = np.where(M & (r > Nmax))
    return np.array(c)
    
c = cornerDetector(im_grey, sigma=sigma, epsilon=epsilon, k=k, tau_frac=0.1)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(r)
ax[1].imshow(im)
ax[1].plot(*c[::-1], '.r')
plt.show()

def corner_pipeline(fname, sigma, epsilon, tau_frac, disk_rad=1, k=0.006):
    im = cv2.imread(DATA_ROOT / fname)[:, :, ::-1].astype(np.float32) / 255
    im_grey = cv2.cvtColor(im[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    r = harrisMeasure(im_grey, sigma, epsilon, k)
    c = cornerDetector(im_grey, sigma=sigma, epsilon=epsilon, k=k, tau_frac=tau_frac, disk_rad=disk_rad)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(r)
    ax[1].imshow(im)
    ax[1].plot(*c[::-1], '.r')
    plt.show()

# try to match that in VisSolutions
corner_pipeline('TestIm1.png', sigma=1, epsilon=1, tau_frac=0.01, disk_rad=5)

# try on other images
corner_pipeline('Box3.bmp', sigma=5, epsilon=5, tau_frac=0.01, disk_rad=5)

# E6.6
im = cv2.imread(DATA_ROOT / 'TestIm1.png')
im_edges = cv2.Canny(im, 20, 10)
plt.imshow(im_edges)
plt.show()

# E6.7
im = cv2.imread(DATA_ROOT / 'TestIm2.png')
im_edges = cv2.Canny(im, 100, 20)
plt.imshow(im_edges)
plt.show()
# cant go lower than 20 on threshold2, otherwise it connects across on the color gradients