import numpy as np

np.random.seed(42)

# 3 токена, размерность модели d, размерность головы h
N, d, h = 3, 8, 4
X0 = np.random.randn(N, d)
I = np.eye(N)

# Генерируем случайные матрицы проекций для 2 слоев
WQ1, WK1 = np.random.randn(d, h), np.random.randn(d, h)
WQ2, WK2 = np.random.randn(d, h), np.random.randn(d, h)

def causal_softmax(A):
    A_masked = A.copy()
    A_masked[np.triu_indices(N, k=1)] = -np.inf
    shift_A = A_masked - np.max(A_masked, axis=1, keepdims=True)
    exp_A = np.exp(shift_A)
    return exp_A / np.sum(exp_A, axis=1, keepdims=True)

def get_N_matrix(D):
    return np.diag(1.0 / np.sqrt(np.diag(D) + 1e-9))

# =====================================================================
# ШАГ 1: ПРЕДВЫЧИСЛЕНИЕ (Делаем на самом старте до прохода по слоям)
# =====================================================================
D0 = X0 @ X0.T
W0 = I  # W0 — это просто единичная матрица на старте

# Вычисляем семантические ядра для каждого слоя
D1_static = X0 @ (WQ1 @ WK1.T) @ X0.T
D2_static = X0 @ (WQ2 @ WK2.T) @ X0.T

# =====================================================================
# ШАГ 2: ДИНАМИЧЕСКИЙ ПРОХОД ПО СЛОЯМ (Работаем только в пространстве N x N)
# =====================================================================
# Слой 1: Используем предвычисленную D1_static и W0
A1_raw = W0 @ D1_static @ W0.T
A1 = causal_softmax(A1_raw / np.sqrt(h))

G1_tilde = (I + A1) @ D0 @ (I + A1).T  # метрика для нормы
N1 = get_N_matrix(G1_tilde)

# Обновляем динамическую матрицу переходов W после 1-го слоя
W1 = N1 @ (I + A1) @ W0

# Слой 2: Используем предвычисленную D2_static и динамическую W1
A2_raw = W1 @ D2_static @ W1.T
A2 = causal_softmax(A2_raw / np.sqrt(h))

D1 = W1 @ D0 @ W1.T  # Расчёт D1 из D0 с помощью матрицы переходов W1

G2_tilde = (I + A2) @ D1 @ (I + A2).T
N2 = get_N_matrix(G2_tilde)

# Финальное обновление W
W2 = N2 @ (I + A2) @ W1

# Итоговый выход системы
X2_alt = W2 @ X0

# =====================================================================
# ПРОВЕРКА: Стандартный послойный расчет векторов (для верификации)
# =====================================================================
# Слой 1
X1_u = X0 + causal_softmax((X0 @ WQ1 @ WK1.T @ X0.T) / np.sqrt(h)) @ X0
X1_std = X1_u / np.linalg.norm(X1_u, axis=1, keepdims=True)
# Слой 2
X2_u = X1_std + causal_softmax((X1_std @ WQ2 @ WK2.T @ X1_std.T) / np.sqrt(h)) @ X1_std
X2_std = X2_u / np.linalg.norm(X2_u, axis=1, keepdims=True)

assert np.allclose(X2_std, X2_alt)
print("Тест пройден! Матрицы совпадают.")
