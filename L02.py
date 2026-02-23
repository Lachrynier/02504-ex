import sympy as sp
from sympy.matrices import Matrix
from sympy.matrices.expressions import kronecker_product

# Scalars with LaTeX-style subscripts
x_1i, y_1i, x_2i, y_2i = sp.symbols('x_1i y_1i x_2i y_2i')

# Homogeneous points q_{1i}, q_{2i}
q_1i = Matrix([x_1i, y_1i, 1])
q_2i = Matrix([x_2i, y_2i, 1])

# Cross-product (skew-symmetric) matrix [q_{1i}]_x
q_1i_cross = Matrix([
    [0,      -q_1i[2],  q_1i[1]],
    [q_1i[2],  0,      -q_1i[0]],
    [-q_1i[1], q_1i[0],  0]
])

# Equivalent explicit form (matches your screenshot):
# q_1i_cross = Matrix([
#     [0,   -1,   y_1i],
#     [1,    0,  -x_1i],
#     [-y_1i, x_1i, 0]
# ])

# B^{(i)} = q_{2i}^T ⊗ [q_{1i}]_x
B_i = kronecker_product(q_2i.T, q_1i_cross)

B_i