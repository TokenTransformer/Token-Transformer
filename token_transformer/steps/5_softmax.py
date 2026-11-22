import numpy as np

np.random.seed(42)

N, d = 3, 4
X0 = np.random.randn(N, d)
I = np.eye(N)

def causal_softmax(D):
    # Причинный Softmax: заполняем строго верхний треугольник -inf
    D_masked = D.copy()
    D_masked[np.triu_indices(N, k=1)] = -np.inf

    # Стабильный Softmax построчно
    shift_D = D_masked - np.max(D_masked, axis=1, keepdims=True)
    exp_D = np.exp(shift_D)
    # Для верхнего треугольника exp(-inf) = 0, деление корректно
    return exp_D / np.sum(exp_D, axis=1, keepdims=True)

def get_N_matrix(D):
    return np.diag(1.0 / np.sqrt(np.diag(D) + 1e-9))

# --- Стандартный послойный расчет векторов с Softmax ---
# Слой 1
A0_std = causal_softmax((X0 @ X0.T) / np.sqrt(d))
X1_unnorm = X0 + A0_std @ X0
X1_std = X1_unnorm / np.linalg.norm(X1_unnorm, axis=1, keepdims=True)

# Слой 2
A1_std = causal_softmax((X1_std @ X1_std.T) / np.sqrt(d))
X2_unnorm = X1_std + A1_std @ X1_std
X2_std = X2_unnorm / np.linalg.norm(X2_unnorm, axis=1, keepdims=True)


# --- Геометрический алгоритм (строго в пространстве N x N) ---
D0 = X0 @ X0.T

# Слой 1
A0 = causal_softmax(D0 / np.sqrt(d))
D1_tilde = (I + A0) @ D0 @ (I + A0).T
N1 = get_N_matrix(D1_tilde)
D1 = N1 @ D1_tilde @ N1

# Слой 2
A1 = causal_softmax(D1 / np.sqrt(d))
D2_tilde = (I + A1) @ D1 @ (I + A1).T
N2 = get_N_matrix(D2_tilde)

# Сквозная матрица весов W
W = N2 @ (I + A1) @ N1 @ (I + A0)
X2_alt = W @ X0

assert np.allclose(X2_std, X2_alt)
print("Тест пройден! Softmax успешно интегрирован в геометрию.")
