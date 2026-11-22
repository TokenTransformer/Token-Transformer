import numpy as np

np.random.seed(42)

# Параметры словаря и модели
V, d, h = 50, 8, 4  # Словарь 50 слов, размерность 8, голова 4
E = np.random.randn(V, d) * 0.5   # Матрица входных эмбеддингов
W_U = np.random.randn(V, d) * 0.5 # Матрица выходных эмбеддингов (Unembedding)

# Входная последовательность из N=10 токенов (N >= d для чистоты базиса)
input_ids = np.array([3, 17, 42, 5, 19, 8, 33, 1, 27, 11])
N = len(input_ids)
M = np.tril(np.ones((N, N)))

# Веса одного слоя внимания
WQ, WK = np.random.randn(d, h) * 0.5, np.random.randn(d, h) * 0.5
WV, WO = np.random.randn(d, h) * 0.3, np.random.randn(h, d) * 0.3
W_QK, W_OV = WQ @ WK.T, WV @ WO

def softmax_rows(S, mask=None):
    S = S.copy()
    if mask is not None: S[mask == 0] = -np.inf
    E_exp = np.exp(S - np.max(S, axis=1, keepdims=True))
    return E_exp / np.sum(E_exp, axis=1, keepdims=True)

def l2_norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# ============================================================================
# 1. КЛАССИЧЕСКИЙ ПРОХОД LLM (IDs -> Векторы -> Логиты)
# ============================================================================
X0 = E[input_ids]  # Embedding lookup
A_std = softmax_rows((X0 @ W_QK @ X0.T) / np.sqrt(h), M)
X1_std = l2_norm_rows(X0 + A_std @ X0 @ W_OV)
logits_std = X1_std @ W_U.T  # Unembedding

# ============================================================================
# 2. ТОКЕННЫЙ ПРОХОД (Только через N x N матрицы и базовые логиты L0)
# ============================================================================
# На старте получаем только геометрию и прямые словарные проекции для input_ids
D0 = X0 @ X0.T
D0_pinv = np.linalg.pinv(D0, hermitian=True)
Q_core = X0 @ W_QK @ X0.T
C_ov = (X0 @ W_OV @ X0.T) @ D0_pinv
L0 = X0 @ W_U.T  # Базовые предсказания словаря (N x V)
W = np.eye(N)

# Считаем сквозную матрицу W (N x N)
A = softmax_rows((Q_core) / np.sqrt(h), M)
W_tilde = W + A @ C_ov
N_mat = np.diag(1.0 / np.sqrt(np.diag(W_tilde @ D0 @ W_tilde.T) + 1e-12))
W = N_mat @ W_tilde

# Финальные логиты всей сети — просто взвешивание базовых логитов L0
logits_alt = W @ L0

assert np.allclose(logits_std, logits_alt)
print("Полный цикл LLM (от ID токенов до логитов словаря) совпал идеально!")
print(f"Предсказанный следующий токен для последней позиции: ID = {np.argmax(logits_alt[-1])}")
