from utils import *; importlib.reload(sys.modules['utils']); from utils import *
from itertools import product
from scipy.spatial.transform import Rotation


Q = np.array(list(product((0, 1), repeat=3)), dtype=float).T

f = 1000
dx = 1920 // 2
dy = 1080 // 2
K = np.array([
    [f, 0, dx],
    [0, f, dy],
    [0, 0, 1]
])

s = np.sqrt(0.5); R = np.array([
    [s, -s, 0],
    [s,  s, 0],
    [0,  0, 1]
])
t = np.array([0., 0., 10.])[:,None]

# E4.1
P = K @ np.column_stack((R, t))
q = P @ PiInv(Q)

# E4.2
P_est = pest(q, PiInv(Q))
q_est = P_est @ PiInv(Q)
e_proj = np.mean(np.linalg.norm(Pi(q_est) - Pi(q), axis=0))
print(f'reprojection error (no normalization): {e_proj}')

P_est = pest(q, PiInv(Q), normalize=True)
q_est = P_est @ PiInv(Q)
e_proj = np.mean(np.linalg.norm(Pi(q_est) - Pi(q), axis=0))
print(f'reprojection error (normalization): {e_proj}')

# E4.3

print(checkerboard_points(2, 3))

# E4.4
Q_omega = checkerboard_points(10, 20)

Ra = Rotation.from_euler('xyz', [np.pi/10, 0, 0]).as_matrix()
Rb = Rotation.from_euler('xyz', [0, 0, 0]).as_matrix()
Rc = Rotation.from_euler('xyz', [-np.pi/10, 0, 0]).as_matrix()
Qa = Ra @ Q_omega
Qb = Rb @ Q_omega
Qc = Rc @ Q_omega
Qs = [Qa, Qb, Qc]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(*Qa)
ax.scatter(*Qb)
ax.scatter(*Qc)

qa = P @ PiInv(Qa)
qb = P @ PiInv(Qb)
qc = P @ PiInv(Qc)

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
for i, qi in enumerate([qa, qb, qc]):
    ax[i].scatter(*Pi(qi))
plt.show()

# E4.5
def estimateHomographies(Q_omega, qs):
    """
    Q_omega: an array original un-transformed checkerboard points in 3D, for example Q_omega.
    qs: a list of arrays, each element in the list containing Q_omega projected to the image plane from 
        different views, for example qs could be [qa, qb, qc].
    Returns homographies that map from Q_omega to each of the entries in qs
    """
    Q_omega = ensure_inhom(Q_omega, dim=3)
    return [hest(q, Q_omega[:2]) for q in qs]

qs = [qa, qb, qc]
Hs = estimateHomographies(Q_omega, qs)
tilde_Q_omega = PiInv(Q_omega[:2])
for q, H in zip(qs, Hs):
    q_est = H @ tilde_Q_omega
    err = np.mean(np.linalg.norm(Pi(q_est) - Pi(q), axis=0))
    print(err)

# E4.6
def estimate_b(Hs):
    def v(H, alpha, beta):
        return np.array([
            H[0, alpha] * H[0, beta],
            H[0, alpha] * H[1, beta] + H[1, alpha] * H[0, beta],
            H[1, alpha] * H[1, beta],
            H[2, alpha] * H[0, beta] + H[0, alpha] * H[2, beta],
            H[2, alpha] * H[1, beta] + H[1, alpha] * H[2, beta],
            H[2, alpha] * H[2, beta],
        ])
    
    V = []
    for H in Hs:
        constr = np.array([
            v(H, 0, 1),
            v(H, 0, 0) - v(H, 1, 1)
        ])
        V.append(constr)
    
    V = np.concatenate(V, axis=0)
    U, S, VT = np.linalg.svd(V)
    b = VT[-1]
    return b

b = estimate_b(Hs)

B_true = np.linalg.inv(K).T @ np.linalg.inv(K)
b_true = np.array([
    B_true[0,0],
    B_true[0,1],
    B_true[1,1],
    B_true[0,2],
    B_true[1,2],
    B_true[2,2]
])
print(f'||b-b_true||: {np.linalg.norm(b/b[-1] - b_true/b_true[-1])}')

# E4.7
def estimateIntrinsics(Hs):
    b = estimate_b(Hs)
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha**2 / lam
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
    K = A
    return K

K_est = estimateIntrinsics(Hs)

# E4.8
def estimateExtrinsics(K, Hs):
    Kinv = np.linalg.inv(K)
    Rs = []
    ts = []
    for H in Hs:
        lam = 1.0 / np.linalg.norm(Kinv @ H[:, 0])
        t_tmp = lam * (Kinv @ H[:, 2])
        if t_tmp[2] < 0:
            H = -H  # flip scale if "behind camera"
        
        lam = 1 / np.linalg.norm(Kinv @ H[:, 0])
        r1 = lam * Kinv @ H[:, 0]
        r2 = lam * Kinv @ H[:, 1]
        r3 = np.cross(r1, r2)
        t = lam * Kinv @ H[:, 2]
        Rs.append(np.column_stack((r1, r2, r3)))
        ts.append(t)

    return Rs, ts

def calibrateCamera(qs, Q):
    Hs = estimateHomographies(Q, qs)
    K = estimateIntrinsics(Hs)
    Rs, ts = estimateExtrinsics(K, Hs)
    return K, Rs, ts

Rs, ts = estimateExtrinsics(K_est, Hs)

for i in range(len(Rs)):
    print('\n================')
    print(f'Est. i: i={i}')
    print(f'Rs[i]:\n{Rs[i]}')
    print(f't[i]:\n{ts[i].ravel()}')

    # R @ Ri @ Q + t
    # [R @ Ri, t @ Ri]
    Ri = [Ra, Rb, Rc][i]
    print(f'R:\n{R @ Ri}')
    print(f't:\n{t.ravel()}')
    print('================')

# E4.9
K_est, Rs, ts = calibrateCamera(qs, Q_omega)

for i, qi in enumerate(qs):
    print('\n================')
    print(f'E4.9: i={i}')
    P_est = K_est @ np.column_stack((Rs[i], ts[i]))
    q_est = P_est @ PiInv(Q_omega)
    err = np.mean(np.linalg.norm(Pi(q_est) - Pi(qi), axis=0))
    print(f'err: {err}')

# E4.10
noisy_qs = [qi + np.random.normal(size=qi.shape) for qi in qs]
K_est, Rs, ts = calibrateCamera(noisy_qs, Q_omega)

for i, qi in enumerate(noisy_qs):
    print('\n================')
    print(f'E4.10 (noisy): i={i}')
    P_est = K_est @ np.column_stack((Rs[i], ts[i]))
    q_est = P_est @ PiInv(Q_omega)
    err = np.mean(np.linalg.norm(Pi(q_est) - Pi(qi), axis=0))
    print(f'err: {err}')

print(f'Noisy K_est:\n{K_est}')