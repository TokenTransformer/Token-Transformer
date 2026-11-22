import numpy as np

np.random.seed(42)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def l2_norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def get_N_matrix(D_gram):
    return np.diag(1.0 / (np.sqrt(np.diag(D_gram)) + 1e-12))

# Параметры: N >= d (режим полного базиса слов)
N, d, d_mlp = 12, 8, 32
X0 = np.random.randn(N, d)

# Случайная текущая смесь токенов (например, после слоя внимания)
W_in_state = np.random.randn(N, N) * 0.3

# Параметрические ключи и значения MLP
W_in = np.random.randn(d, d_mlp) * 0.4
W_out = np.random.randn(d_mlp, d) * 0.4

# ============================================================================
# 1. КЛАССИЧЕСКИЙ MLP (Через перевод смеси в плотные векторы X и обратно)
# ============================================================================
X_curr = W_in_state @ X0                  # Материализуем плотные векторы
X_after_mlp = X_curr + gelu(X_curr @ W_in) @ W_out
X_std = l2_norm_rows(X_after_mlp)

# ============================================================================
# 2. ТОКЕННЫЙ MLP (Работа напрямую со смесями W без плотных векторов)
# ============================================================================
# --- Предвычисление на старте (перевод параметров MLP на язык токенов) ---
D0 = X0 @ X0.T
D0_pinv = np.linalg.pinv(D0, hermitian=True)

K_tok = X0 @ W_in                         # Ключи MLP в координатах слов (N x d_mlp)
V_tok = (W_out @ X0.T) @ D0_pinv          # Значения MLP как рецепты смесей (d_mlp x N)

# --- Прямой проход (только матрицы смесей N x N и нейроны d_mlp) ---
neuron_activations = gelu(W_in_state @ K_tok)   # Ассоциирование смеси с ключами
C_mlp = neuron_activations @ V_tok              # Взвешенное суммирование рецептов значений

W_tilde = W_in_state + C_mlp                    # Обновление концентрации смеси
N_mat = get_N_matrix(W_tilde @ D0 @ W_tilde.T)  # Нормализация через матрицу Грама
W_out_state = N_mat @ W_tilde

# ============================================================================
# ПРОВЕРКА
# ============================================================================
X_alt = W_out_state @ X0
rel_err = np.linalg.norm(X_alt - X_std) / np.linalg.norm(X_std)
print(f"Ошибка токенного MLP: {rel_err:.2e}")
assert np.allclose(X_alt, X_std)
print("Идеально! MLP отработал напрямую со смесями слов без плотных векторов.")
