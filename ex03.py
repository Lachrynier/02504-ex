import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from utils import *

### Epipolar geometry
K = np.array([
    [1000, 0, 300],
    [0, 1000, 200],
    [0, 0, 1],
], dtype=float)

R1 = np.eye(3)
t1 = np.zeros(3)

R2 = Rotation.from_euler('xyz', [0.7, -0.5, 0.8]).as_matrix()
t2 = np.array([0.2, 2, 1])


# E2.1
Q = np.array([1, 0.5, 4, 1])

# q1 = 