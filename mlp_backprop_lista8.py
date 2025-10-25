import numpy as np

# ==============================
# Funções de ativação e derivadas
# (derivadas em função do pré-ativação z)
# ==============================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(x):
    return np.maximum(0.0, x)

def d_relu(z):
    return (z > 0.0).astype(float)

def softmax(z):
    # estável numericamente
    z_ = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z_)
    return e / np.sum(e, axis=1, keepdims=True)

def d_softmax_mse(z):
    # Derivada efetiva de softmax quando usado com MSE:
    # delta_out = (y - a_o) * d_softmax(z)
    # Aqui usamos uma aproximação diagonal: s*(1-s) (aceita em muitas listas quando a loss é MSE)
    s = softmax(z)
    return s * (1.0 - s)

# =====================================
# Rede Neural Multicamadas (1 hidden)
# =====================================

class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.1,
        hidden_activation=relu,
        hidden_derivative=d_relu,
        output_activation=sigmoid,
        output_derivative=d_sigmoid,   # se usar softmax+MSE, passe d_softmax_mse
        seed: int | None = 42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.h_act = hidden_activation
        self.h_der = hidden_derivative
        self.o_act = output_activation
        self.o_der = output_derivative

        rng = np.random.default_rng(seed)
        # Xavier/He simples (opcional): aqui uso escala padrão pequena
        self.Wih = rng.normal(0, 1.0 / np.sqrt(input_size), size=(input_size, hidden_size))
        self.Who = rng.normal(0, 1.0 / np.sqrt(hidden_size), size=(hidden_size, output_size))
        self.bh = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, output_size))

    def feedforward(self, X: np.ndarray):
        # X: (n, input_size)
        z_h = X @ self.Wih + self.bh          # (n, hidden)
        a_h = self.h_act(z_h)                  # (n, hidden)
        z_o = a_h @ self.Who + self.bo         # (n, output)
        a_o = self.o_act(z_o)                  # (n, output)
        return z_h, a_h, z_o, a_o

    def backward_batch(self, X, y, z_h, a_h, z_o, a_o):
        # Loss: MSE => dL/da_o = (a_o - y)
        # delta_out = (y - a_o) * d_out  (usando a convenção de somar gradiente ao peso)
        # Observação: usando MSE, tanto faz (y - a_o) ou -(a_o - y) com o sinal no update;
        # aqui usamos (y - a_o) e somamos no update (gradiente ascendente) conforme código original do usuário.
        delta_o = (y - a_o) * self.o_der(z_o)                      # (n, out)
        grad_Who = a_h.T @ delta_o                                 # (hidden, out)
        grad_bo = np.sum(delta_o, axis=0, keepdims=True)           # (1, out)

        delta_h = (delta_o @ self.Who.T) * self.h_der(z_h)         # (n, hidden)
        grad_Wih = X.T @ delta_h                                   # (inp, hidden)
        grad_bh = np.sum(delta_h, axis=0, keepdims=True)           # (1, hidden)

        # Atualização
        self.Who += self.lr * grad_Who
        self.bo  += self.lr * grad_bo
        self.Wih += self.lr * grad_Wih
        self.bh  += self.lr * grad_bh

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        batch_size: int | None = None,
        shuffle: bool = True,
        convergence_threshold: float | None = None,
        verbose_every: int = 100
    ):
        n = X.shape[0]
        idx = np.arange(n)

        for epoch in range(1, epochs + 1):
            if shuffle:
                np.random.shuffle(idx)

            total_mse = 0.0
            if batch_size is None or batch_size >= n:
                # full-batch
                z_h, a_h, z_o, a_o = self.feedforward(X[idx])
                self.backward_batch(X[idx], y[idx], z_h, a_h, z_o, a_o)
                total_mse = np.mean((y[idx] - a_o) ** 2)
            else:
                # mini-batch
                for start in range(0, n, batch_size):
                    sl = idx[start:start + batch_size]
                    z_h, a_h, z_o, a_o = self.feedforward(X[sl])
                    self.backward_batch(X[sl], y[sl], z_h, a_h, z_o, a_o)
                    total_mse += np.mean((y[sl] - a_o) ** 2) * len(sl)
                total_mse /= n

            if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
                print(f"[epoch {epoch:4d}] MSE={total_mse:.6f}")

            if convergence_threshold is not None and total_mse < convergence_threshold:
                print(f"Converged at epoch {epoch} with MSE={total_mse:.6f}")
                break

    def predict(self, X: np.ndarray):
        _, _, _, a_o = self.feedforward(X)
        return a_o


# ==============================
# Datasets
# ==============================

def xor_dataset():
    X = np.array([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]], dtype=float)
    y = np.array([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=float)
    return X, y

# 7 segmentos (a,b,c,d,e,f,g)
# 1 = segmento aceso, 0 = apagado
SEG_10 = {
  0:[1,1,1,1,1,1,0],
  1:[0,1,1,0,0,0,0],
  2:[1,1,0,1,1,0,1],
  3:[1,1,1,1,0,0,1],
  4:[0,1,1,0,0,1,1],
  5:[1,0,1,1,0,1,1],
  6:[1,0,1,1,1,1,1],
  7:[1,1,1,0,0,0,0],
  8:[1,1,1,1,1,1,1],
  9:[1,1,1,1,0,1,1],
}

def dataset_7segments(one_hot: bool = True):
    X = np.array([SEG_10[d] for d in range(10)], dtype=float)      # (10,7)
    if one_hot:
        Y = np.eye(10, dtype=float)                                # (10,10)
    else:
        # 4 bits binário (do mais significativo ao menos significativo)
        def to4bits(n): return [(n>>3)&1, (n>>2)&1, (n>>1)&1, n&1]
        Y = np.array([to4bits(d) for d in range(10)], dtype=float) # (10,4)
    return X, Y

# Ruído: inverte (bit-flip) cada posição com probabilidade p
def add_bitflip_noise(X: np.ndarray, p: float = 0.1, seed: int | None = 0):
    rng = np.random.default_rng(seed)
    flips = rng.random(X.shape) < p
    # X é binário 0/1; inverte 0->1 e 1->0
    return np.abs(X - flips.astype(float))


# ==============================
# Métricas
# ==============================

def accuracy_onehot(pred_logits, target_onehot):
    pred = np.argmax(pred_logits, axis=1)
    true = np.argmax(target_onehot, axis=1)
    return np.mean(pred == true)

def accuracy_bitsigmoid(pred_probs, target_bits):
    # limiar 0.5 em cada bit
    pred_bits = (pred_probs >= 0.5).astype(int)
    return np.mean(np.all(pred_bits == target_bits.astype(int), axis=1))


# ==============================
# Experimentos
# ==============================

def run_xor():
    print("\n=== XOR (2-2-1, hidden ReLU, out Sigmoid, MSE) ===")
    X, y = xor_dataset()

    nn = NeuralNetwork(
        input_size=2, hidden_size=2, output_size=1,
        learning_rate=0.5,
        hidden_activation=relu, hidden_derivative=d_relu,
        output_activation=sigmoid, output_derivative=d_sigmoid,
        seed=42
    )

    nn.train(X, y, epochs=5000, batch_size=None, convergence_threshold=1e-3, verbose_every=500)

    out = nn.predict(X)
    pred = (out >= 0.5).astype(int)
    print("Entradas:\n", X)
    print("Alvos:\n", y.reshape(-1))
    print("Saídas (prob):\n", out.reshape(-1))
    print("Preditos (0/1):\n", pred.reshape(-1))
    acc = np.mean(pred.reshape(-1) == y.reshape(-1))
    print(f"Acurácia XOR: {acc:.3f}")

def run_7segments(one_hot=True, noise_p=0.10):
    print("\n=== 7-Segmentos (7-5-10 com Softmax) OU (7-5-4 com Sigmoid) ===")
    X, Y = dataset_7segments(one_hot=one_hot)

    if one_hot:
        # 7-5-10 com Softmax (saída multiclasses)
        nn = NeuralNetwork(
            input_size=7, hidden_size=5, output_size=10,
            learning_rate=0.2,
            hidden_activation=relu, hidden_derivative=d_relu,
            output_activation=softmax, output_derivative=d_softmax_mse,
            seed=7
        )
    else:
        # 7-5-4 com Sigmoid (saída em 4 bits)
        nn = NeuralNetwork(
            input_size=7, hidden_size=5, output_size=4,
            learning_rate=0.2,
            hidden_activation=relu, hidden_derivative=d_relu,
            output_activation=sigmoid, output_derivative=d_sigmoid,
            seed=7
        )

    # Treinar várias épocas passando por todos os 10 dígitos
    # (dados pequenos: full-batch funciona bem)
    nn.train(X, Y, epochs=3000, batch_size=None, convergence_threshold=1e-5, verbose_every=300)

    # Avaliação "clean"
    logits_clean = nn.predict(X)
    if one_hot:
        acc_clean = accuracy_onehot(logits_clean, Y)
    else:
        acc_clean = accuracy_bitsigmoid(logits_clean, Y)
    print(f"Acurácia (clean): {acc_clean:.3f}")

    # Avaliação com ruído de entrada
    X_noisy = add_bitflip_noise(X, p=noise_p, seed=123)
    logits_noisy = nn.predict(X_noisy)
    if one_hot:
        acc_noisy = accuracy_onehot(logits_noisy, Y)
    else:
        acc_noisy = accuracy_bitsigmoid(logits_noisy, Y)
    print(f"Acurácia (noisy, p={noise_p:.2f}): {acc_noisy:.3f}")

    # Mostrar confusão simples (apenas no modo one-hot)
    if one_hot:
        pred_clean = np.argmax(logits_clean, axis=1)
        print("Predições clean 0..9:", pred_clean.tolist())
        pred_noisy = np.argmax(logits_noisy, axis=1)
        print("Predições noisy 0..9:", pred_noisy.tolist())

    # Dica: se quiser usar Cross-Entropy, pode trocar no backward:
    # delta_o = (a_o - y)  (para softmax+CE), e ignorar d_softmax.
    # Mas como muita lista pede MSE, mantivemos MSE aqui.

if __name__ == "__main__":
    # 1) XOR
    run_xor()

    # 2) 7-segmentos
    # -> Use one_hot=True para 10 saídas (one-hot) [RECOMENDADO]
    # -> Use one_hot=False para 4 saídas em binário
    run_7segments(one_hot=True, noise_p=0.10)
