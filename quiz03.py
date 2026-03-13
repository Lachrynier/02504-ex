from utils import *; importlib.reload(sys.modules['utils']); from utils import *
import numpy as np

K = np.array([[900, 0, 1070], [0, 900, 610.0], [0, 0, 1]], float)
R1 = cv2.Rodrigues(np.array([-1.6, 0.3, -2.1]))[0]
t1 = np.array([[0.0], [1.0], [3.0]], float)
R2 = cv2.Rodrigues(np.array([-0.4, -1.3, -1.6]))[0]
t2 = np.array([[0.0], [1.0], [6.0]], float)
R3 = cv2.Rodrigues(np.array([2.5, 1.7, -0.4]))[0]
t3 = np.array([[2.0], [-7.0], [25.0]], float)

p1 = np.array([[1046.0], [453.0]])
p2 = np.array([[1126.0], [671.0]])
p3 = np.array([[1165.0], [453.0]])

### Q1
def im2cam(K, q):
    # im coords to cam coords at the image plane
    Q_norm = np.linalg.inv(K) @ PiInv(q)
    Q = K[0, 0] * Q_norm
    return Q

def cam2cam(K, R1, t1, R2, t2, Q2):
    """
    Convert coordinates from camera 2 to camera 1.
    Q2 (3D inhom.) described in camera 2 is turned into Q1 (3D inhom.)
    """
    assert Q2.ndim == 2
    # Q2 = R2 @ Qw + t2 <=> Qw = R2.T @ (Q2 - t2)
    Qw = R2.T @ (Q2 - t2)
    # Q1 = R1 @ Qw + t1
    Q1 = R1 @ Qw + t1
    return Q1

def dist_to_line(A, B, C):
    """Distance from C to the line between A and B"""
    assert A.ndim == 2
    assert A.shape == B.shape == C.shape
    AB = B - A
    AC = C - A
    # projection of C onto AB
    proj = (AB.T @ AC / np.linalg.norm(AB)**2) * AB + A
    dist = np.linalg.norm(C - proj)
    return dist

Q1_C2 = cam2cam(K, R2, t2, R1, t1, im2cam(K, p1))
Qc1_C2 = cam2cam(K, R2, t2, R1, t1, np.zeros((3, 1)))

to_im_plane = lambda K, Q: Q * K[0, 0] / Q[-1]
Q1_C2_epiline = to_im_plane(K, Q1_C2)
Qc1_C2_epiline = to_im_plane(K, Qc1_C2)

Q2_epiline = im2cam(K, p2)

answer_q1 = dist_to_line(Q1_C2_epiline, Qc1_C2_epiline, Q2_epiline)

print(f'answer_q1 (analytical geometry): {answer_q1}')

# try to do it the way on slides
# find R,t that maps from reference frame of camera 1 to reference frame of camera 2
# p1 = R1 @ Q + t1 <=> Q = R1.T @ (p1 - t1)
# p2 = R2 @ Q + t2 = R2 @ R1.T @ p1 + t2 - R2 @ R1.T @ t1

R, t = ref2ref(R1, t1, R2, t2)
F = fundamental_matrix(R, t, K, K)
l = F @ PiInv(p1)
# l = (a,b,c)
# ax + by + cz = 0 <=> (a,b).(x,y) = -cz
# proj = (a,b).(x,y)/||(a,b)||^2 * (a,b) = ((a,b)/||(a,b)||).(x,y) * ((a,b)/||(a,b)||)
# is distance along the unit direction of (a,b)
# ((a,b)/||(a,b)||).(x/z,y/z) = -c/||(a,b)||
# distance to line is  | ((a,b)/||(a,b)||).(x/z,y/z) - (-c) |
l_norm = l / np.linalg.norm(l[:2])
dist_to_l = np.abs(PiInv(p2).T @ l_norm)
print(f'answer_q1 (fundamental matrix): {dist_to_l}')


### Q2
P1 = K @ np.column_stack((R1, t1))
P2 = K @ np.column_stack((R2, t2))
P3 = K @ np.column_stack((R3, t3))
Q_est = triangulate((p1, p2, p3), (P1, P2, P3))
print(f'answer_q2: {Pi(Q_est)}')
print(p1.ravel(), Pi(P1 @ Q_est))
print(p2.ravel(), Pi(P2 @ Q_est))
print(p3.ravel(), Pi(P3 @ Q_est))