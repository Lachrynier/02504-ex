from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from utils import *; importlib.reload(sys.modules['utils']); from utils import *
np.set_printoptions(suppress=True)


im = cv2.imread('data/sunflowers.jpg')
im = im.astype(float).mean(2)/255

# E8.1

def scaleSpaced(im, sigma, n):
    """
    im: HxW

    returns: im_scales, where
      im_scales[i] = gaussian_filter(im, 2**i * sigma), i=0,1,...,n-1
    
    --- Calculations ---
    2s = sqrt(s^2+x^2)
    x^2 = 4s^2 - s^2
    x = sqrt(3)s

    2^(k+1) s = sqrt((2^k s)^2 + x^2)
    x^2 = 2^{2(k+1)} s^2 - 2^{2k} s^2
    x = s sqrt(2^{2k+2*1} - 2^{2k}) = s 2^k sqrt(3)
    """
    im_scales = []
    im_scales.append(gaussian_filter(im, sigma))
    scales = [sigma]
    for i in range(n - 1):
        scales.append(2**(i+1) * sigma)
        sig_inc = np.sqrt(3) * 2**i * sigma
        im_scales.append(gaussian_filter(im_scales[i], sig_inc))

        # # validate
        # im_val = gaussian_filter(im, 2**(i+1) * sigma)
        # print(np.mean(np.abs(im_scales[i+1] - im_val)))
    
    return np.array(im_scales), np.array(scales, dtype=float)

sigma = 2
n = 7
im_scales, scales = scaleSpaced(im, sigma=sigma, n=n)

# E8.2
def differenceOfGaussians(im, sigma, n):
    im_scales, scales = scaleSpaced(im, sigma, n)
    DoG = []
    for i in range(n - 1):
        DoG.append(im_scales[i+1] - im_scales[i])
    
    return np.array(DoG), scales

DoG, scales = differenceOfGaussians(im, sigma=sigma, n=n)

fig, ax = plt.subplots(n, 2, figsize=(6, 2.3*n))
for i in range(n):
    ax[i, 0].imshow(im_scales[i], cmap='grey')
    ax[i, 0].title.set_text('scale space')
    if i == 0:
        ax[i, 1].imshow(im, cmap='grey')
        ax[i, 1].title.set_text('original')
    else:
        ax[i, 1].imshow(DoG[i-1], cmap='grey')
        ax[i, 1].title.set_text('DoG')

# E8.3
def detectBlobs(im, sigma, n, tau):
    DoG, scales = differenceOfGaussians(im, sigma=sigma, n=n)
    is_high_magnitude = np.abs(DoG) > tau
    footprint = np.ones(3*(3,), dtype=bool)
    footprint[1, 1, 1] = 0
    is_local_min = DoG < minimum_filter(DoG, footprint=footprint, mode='reflect')
    is_local_max = DoG > maximum_filter(DoG, footprint=footprint, mode='reflect')
    blobs = (is_local_min | is_local_max) & is_high_magnitude
    return blobs, DoG, scales

def vizBlobs(im, blobs, scales, circle_cmap='autumn'):
    blob_locations = np.argwhere(blobs)
    patches = []
    edgecolors = []

    cmap = plt.get_cmap(circle_cmap)
    n_scales = blobs.shape[0]
    for s, y, x in blob_locations:
        r = np.sqrt(2 * scales[s] * scales[s+1]) # effective DoG radius

        color = cmap(s / max(n_scales - 1, 1))
        edgecolors.append(color)
        patches.append(Circle((x, y), r))

    fig, ax = plt.subplots()
    ax.imshow(im, cmap='grey')
    coll = PatchCollection(patches, edgecolors=edgecolors, facecolors='None')
    ax.add_collection(coll)
    plt.show()


tau = 0.1
blobs, DoG, scales = detectBlobs(im, sigma, n, tau)
vizBlobs(im, blobs, scales)