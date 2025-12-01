import numpy as np
from cvxopt import matrix, solvers


P = matrix([[1.0, 0.0], [0.0, 4.0]])
q = matrix([-8.0, -16.0])
G = matrix([[-1.0, 0.0, 1.0, -1.0], [0.0, -1.0, 1.0, 0.0]])
h = matrix([0.0, 0.0, 5.0, 3.0])

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h)
print(sol['x'])

C = 1
X = X1
y = y1 * 2 - 1
d = X.shape[1]
N = X.shape[0]

P = np.block([[np.eye(d),     np.zeros((d, N+1))],
              [np.zeros((N+1, d)), np.zeros((N+1, N+1))]
             ])

q = np.vstack((np.zeros(d+1)[:, None], C*np.ones(N)[:, None]))

G = np.block([[-y[:, None] * X,         -y[:, None], -np.eye(N)],
              [np.zeros((N, d+1)), -np.eye(N)]
             ])

h = np.vstack((-np.ones(N)[:, None], np.zeros(N)[:, None]))

A = matrix(np.hstack((y, np.zeros((3))))[:, None].T) # A = y^T
b = matrix(0.0)

P.shape, q.shape, G.shape, h.shape

solvers.options['show_progress'] = False
sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
w = np.array(sol['x'])[:2]
# w = w/np.linalg.norm(w)
b = np.array(sol['x'])[2:3][0][0]
ksi = np.array(sol['x'])[3:]
w.shape, b.shape, ksi.shape

plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap='autumn', edgecolor='black')

# Создаем точки для построения линий
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, 100)

# Разделяющая гиперплоскость: w·x + b = 0
yy = (-w[0][0] * xx + b) / w[1][0]
plt.plot(xx, yy, 'k-', label='Decision boundary')

# Верхняя граница: w·x + b = 1
yy_upper = (-w[0][0] * xx + b - 1) / w[1][0]
plt.plot(xx, yy_upper, 'k--', label='Margin')

# Нижняя граница: w·x + b = -1
yy_lower = (-w[0][0] * xx + b + 1) / w[1][0]
plt.plot(xx, yy_lower, 'k--')

plt.ylim(0, 2)
plt.legend()
plt.show()