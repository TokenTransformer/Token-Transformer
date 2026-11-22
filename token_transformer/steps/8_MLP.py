import numpy as np

np.random.seed(42)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax_rows(S, mask=None):
    S = S.copy()
    if mask is not None:
        S[mask == 0] = -np.inf
    alive = np.isfinite(S).any(axis=1)
    A = np.zeros_like(S)
    Sa = S[alive]
    E = np.exp(Sa - Sa.max(axis=1, keepdims=True))
    A[alive] = E / E.sum(axis=1, keepdims=True)
    return A

def get_N_matrix(D_gram):
    diag = np.diag(D_gram)
    inv_sqrt = np.where(diag > 1e-12, 1.0 / np.sqrt(diag), 0.0)
    return np.diag(inv_sqrt)

def l2_norm_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.where(norms > 1e-12, X / norms, 0.0)

# Параметры: N >= d для полноты базиса
N, d, h, d_mlp = 12, 8, 4, 16
X0 = np.random.randn(N, d)
M = np.tril(np.ones((N, N)))

# Веса одного полного блока Трансформера (Attention + MLP)
WQ, WK = np.random.randn(d, h) * 0.5, np.random.randn(d, h) * 0.5
WV, WO = np.random.randn(d, h) * 0.3, np.random.randn(h, d) * 0.3
W_QK = WQ @ WK.T
W_OV = WV @ WO

W_in = np.random.randn(d, d_mlp) * 0.3
W_out = np.random.randn(d_mlp, d) * 0.3

# ============================================================================
# 1. ЭТАЛОННЫЙ ПРОХОД (В пространстве признаков d)
# ============================================================================
# Подблок Attention + L2Norm
A_std = softmax_rows((X0 @ W_QK @ X0.T) / np.sqrt(h), M)
X_mid = l2_norm_rows(X0 + A_std @ X0 @ W_OV)

# Подблок MLP + L2Norm
mlp_out_std = gelu(X_mid @ W_in) @ W_out
X_final_std = l2_norm_rows(X_mid + mlp_out_std)

# ============================================================================
# 2. ТОКЕННЫЙ ПРОХОД (Через эволюцию сквозной матрицы W)
# ============================================================================
D0 = X0 @ X0.T
D0_pinv = np.linalg.pinv(D0, hermitian=True)

# Предвычисляем формы внимания
Q_core = X0 @ W_QK @ X0.T
C_ov = (X0 @ W_OV @ X0.T) @ D0_pinv

W = np.eye(N)

# --- Шаг А: Attention ---
A = softmax_rows((W @ Q_core @ W.T) / np.sqrt(h), M)
W_att = W + A @ W @ C_ov
N_att = get_N_matrix(W_att @ D0 @ W_att.T)
W = N_att @ W_att

# --- Шаг Б: MLP ---
# Материализуем текущее состояние для нелинейности GELU
X_curr = W @ X0
delta_X = gelu(X_curr @ W_in) @ W_out

# Переносим нелинейный прирост в токенное пространство (координаты в базисе X0)
C_mlp = (delta_X @ X0.T) @ D0_pinv
W_mlp = W + C_mlp

# Финальная L2-нормализация после MLP
N_mlp = get_N_matrix(W_mlp @ D0 @ W_mlp.T)
W = N_mlp @ W_mlp

# ============================================================================
# ПРОВЕРКА
# ============================================================================
X_final_alt = W @ X0
rel_err = np.linalg.norm(X_final_alt - X_final_std) / np.linalg.norm(X_final_std)
print(f"Полный блок (Attention + MLP + L2Norm): отн. ошибка = {rel_err:.2e}")
assert np.allclose(X_final_alt, X_final_std)
print("Успешно!")
