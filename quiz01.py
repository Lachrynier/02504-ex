import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

### Q1
# If a2 + b2 = 1 and the scale of the homogeneous point is 1 then,
q = np.array([2,4,3])
q_hat = q / q[-1]
l = np.array([1,2,2])
l_hat = l / np.linalg.norm(l[:2])

d = q_hat @ l_hat
print(f'd: {d}')

### Q1 verification
x = np.linspace(-5, 5, 500)
y = (-l[0]*x - l[2]) / l[1]
p = np.column_stack((x,y))
q_inhom = Pi(q)
d_empirical = np.min(np.linalg.norm(p - q_inhom, axis=1))
print(f'd_empirical: {d_empirical}')

plt.plot(x,y)
plt.plot(*q_inhom, '.r')
plt.grid(True)
[f(-2,3) for f in (plt.xlim, plt.ylim)]
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

### Q2
f = 1720
delta_x = 680
delta_y = 610.0
K = np.array([
   [f,0,  delta_x],
   [0, f, delta_y],
   [0, 0, 1]
])

R = cv2.Rodrigues(np.array([-0.1, 0.1, -0.2]))[0]
t = np.array([[0.09], [0.05], [0.05]])
q_world = np.array([-0.03, 0.01, 0.59])

q_cam = np.column_stack((R, t)) @ PiInv(q_world)
q_im = Pi(K @ q_cam)
print(f'q_im: {q_im}')