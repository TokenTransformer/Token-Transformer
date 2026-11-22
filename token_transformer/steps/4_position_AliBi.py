import numpy as np

X0 = np.random.randn(3, 4)
I = np.eye(3)
M = np.tril(np.ones((3, 3)))

# Строим матрицу относительных позиционных смещений (AliBi)
# Штрафуем нижний треугольник за удаленность от диагонали
gamma = 0.5
B = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        if i > j:
            B[i, j] = -(i - j) * gamma

def get_N_matrix(D):
    return np.diag(1.0 / np.sqrt(np.diag(D) + 1e-9))

# --- Расчет геометрии с внедрением позиций в пространстве N x N ---
D0 = X0 @ X0.T

# Слой 1: Внимание + Маска + Позиционное смещение B
A0 = (D0 * M) + B  # Позиции вошли в чат!
D1_tilde = (I + A0) @ D0 @ (I + A0).T
N1 = get_N_matrix(D1_tilde)
D1 = N1 @ D1_tilde @ N1

# Слой 2
A1 = (D1 * M) + B
D2_tilde = (I + A1) @ D1 @ (I + A1).T
N2 = get_N_matrix(D2_tilde)

# Сквозные веса W теперь содержат в себе информацию о порядке токенов
W = N2 @ (I + A1) @ N1 @ (I + A0)
X2 = W @ X0

print("Матрица сквозных весов W с учетом позиционной информации:\n", np.round(W, 4))
