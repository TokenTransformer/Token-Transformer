import numpy as np

X0 = np.random.randn(3, 4)
I = np.eye(3)

def l2_norm_rows(M):
    # Стандартная нормализация строк матрицы векторов
    return M / np.linalg.norm(M, axis=1, keepdims=True)

def get_N_matrix(D):
    # Вычисление матрицы N(D) через диагональ матрицы Грама
    diag = np.diag(D)
    # Защита от деления на 0
    return np.diag(1.0 / np.sqrt(diag + 1e-9))

# --- Стандартный послойный расчет векторов с L2-нормализацией ---
# Слой 1
X1_unnorm = X0 + X0 @ X0.T @ X0
X1_std = l2_norm_rows(X1_unnorm)
# Слой 2
X2_unnorm = X1_std + X1_std @ X1_std.T @ X1_std
X2_std = l2_norm_rows(X2_unnorm)

# --- Расчет только через геометрию (матрицу Грама) ---
D0 = X0 @ X0.T

# Слой 1
G1_tilde = (I + D0) @ D0 @ (I + D0).T
N1 = get_N_matrix(G1_tilde)
D1 = N1 @ G1_tilde @ N1  # Новая матрица Грама

# Слой 2
G2_tilde = (I + D1) @ D1 @ (I + D1).T
N2 = get_N_matrix(G2_tilde)

# Сборка общей матрицы переходов W
W = N2 @ (I + D1) @ N1 @ (I + D0)

# Финальный расчет
X2_alt = W @ X0

assert np.allclose(X2_std, X2_alt)
print("Тест пройден! Матрицы совпадают.")
