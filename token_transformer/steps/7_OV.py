import numpy as np

np.random.seed(42)

def softmax_rows(S, mask=None):
    """Построчный Softmax; замаскированные элементы (mask == 0) обнуляются."""
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
    """Вычисляет диагональную матрицу L2-нормализации из матрицы Грама."""
    diag = np.diag(D_gram)
    # Защита от деления на ноль для виртуальных (замаскированных) токенов
    inv_sqrt = np.where(diag > 1e-12, 1.0 / np.sqrt(diag), 0.0)
    return np.diag(inv_sqrt)

def precompute_forms(X0, W_QKs, W_OVs):
    """
    Единственное место касания с пространством признаков d.
    Предвычисляет на старте базовую геометрию и семантические ядра слоёв (N x N).
    """
    D0 = X0 @ X0.T
    Qs = [X0 @ W_QK @ X0.T for W_QK in W_QKs]
    Gs = [X0 @ W_OV @ X0.T for W_OV in W_OVs]
    return D0, Qs, Gs

def token_space_transformer(D0, Qs, Gs, h, mask=None):
    """
    Полный проход Трансформера исключительно через матрицы N x N.
    Включает: QK-внимание, причинный Softmax, OV-контур и L2-нормализацию.
    Возвращает сквозную матрицу переходов W, историю W по слоям и карты внимания.
    """
    N = D0.shape[0]
    D0_pinv = np.linalg.pinv(D0, hermitian=True)

    # Межтокенные операторы контента: X0 @ W_OVl = C_l @ X0
    Cs = [G @ D0_pinv for G in Gs]

    W = np.eye(N)
    Ws, As = [W.copy()], []

    for Q_core, C in zip(Qs, Cs):
        # 1. Логиты внимания через предвычисленное QK-ядро и текущую матрицу W
        logits = (W @ Q_core @ W.T) / np.sqrt(h)
        A = softmax_rows(logits, mask)

        # 2. Остаточное соединение с учётом переноса W_OV налево
        W_tilde = W + A @ W @ C

        # 3. L2-нормализация через обновлённую матрицу Грама
        D_tilde = W_tilde @ D0 @ W_tilde.T
        N_mat = get_N_matrix(D_tilde)
        W = N_mat @ W_tilde

        As.append(A)
        Ws.append(W.copy())

    return W, Ws, As

def reference_forward(X0, W_QKs, W_OVs, h, mask=None):
    """Классический послойный проход для проверки (в пространстве признаков d)."""
    Xs = [X0]
    for W_QK, W_OV in zip(W_QKs, W_OVs):
        X = Xs[-1]
        A = softmax_rows((X @ W_QK @ X.T) / np.sqrt(h), mask)
        X_tilde = X + A @ X @ W_OV

        norms = np.linalg.norm(X_tilde, axis=1, keepdims=True)
        X_norm = np.where(norms > 1e-12, X_tilde / norms, 0.0)
        Xs.append(X_norm)
    return Xs

# ============================================================================
# ТЕСТ 1: Длинный контекст (N >= d) — точное описание без дополнения базиса
# ============================================================================
N, d, h, L = 12, 8, 4, 3
X0 = np.random.randn(N, d)
M = np.tril(np.ones((N, N)))

W_QKs, W_OVs = [], []
for _ in range(L):
    WQ, WK = np.random.randn(d, h) * 0.5, np.random.randn(d, h) * 0.5
    WV, WO = np.random.randn(d, h) * 0.3, np.random.randn(h, d) * 0.3
    W_QKs.append(WQ @ WK.T)
    W_OVs.append(WV @ WO)

Xs_std = reference_forward(X0, W_QKs, W_OVs, h, M)
D0, Qs, Gs = precompute_forms(X0, W_QKs, W_OVs)
W, Ws, As = token_space_transformer(D0, Qs, Gs, h, M)

X_alt = W @ X0
rel_err = np.linalg.norm(X_alt - Xs_std[-1]) / np.linalg.norm(Xs_std[-1])
print(f"Тест 1 (N={N} >= d={d}, {L} слоя): отн. ошибка = {rel_err:.2e}")
assert np.allclose(X_alt, Xs_std[-1])

# ============================================================================
# ТЕСТ 2: Короткий контекст (N < d) — с виртуальными токенами для полноты базиса
# ============================================================================
Ns, ds = 3, 8
Xs = np.random.randn(Ns, ds)
Xs_ref = reference_forward(Xs, W_QKs, W_OVs, h, np.tril(np.ones((Ns, Ns))))[-1]

# Дополняем строки Xs до полного ортонормированного базиса R^d
Q_basis, _ = np.linalg.qr(Xs.T, mode='complete')
X_aug = np.vstack([Xs, Q_basis[:, Ns:].T])  # размер d x d

# Маска: внимание разрешено только внутри реальных токенов (верхний левый блок Ns x Ns)
mask_aug = np.zeros((ds, ds))
mask_aug[:Ns, :Ns] = np.tril(np.ones((Ns, Ns)))

D0_aug, Qs_aug, Gs_aug = precompute_forms(X_aug, W_QKs, W_OVs)
W_aug, _, _ = token_space_transformer(D0_aug, Qs_aug, Gs_aug, h, mask_aug)

X_aug_out = (W_aug @ X_aug)[:Ns]
rel_err_s = np.linalg.norm(X_aug_out - Xs_ref) / np.linalg.norm(Xs_ref)
print(f"Тест 2 (N={Ns} < d={ds} + виртуальные токены): отн. ошибка = {rel_err_s:.2e}")
assert np.allclose(X_aug_out, Xs_ref)
print("Все проверки пройдены идеально!")
