import requests
import numpy as np
import time
import os
import sys

# Adiciona o diretório scripts ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils.datasetConf import preparar_mnist_non_iid

SERVER_URL    = "http://localhost:5000"
NUM_CLIENTES  = 50
RODADAS       = 160
HIDDEN_DIM    = 128
OUTPUT_DIM    = 10
INPUT_DIM     = 784
LEARNING_RATE = 0.05  # LR inicial
LR_DECAY      = 0.96  # fator multiplicativo por rodada
LR_MIN        = 0.01  # floor — LR nunca cai abaixo disso
MU            = 0.001  # coeficiente proximal FedProx — ancora clientes ao modelo global

# W1 congelada — simula o firmware real (W1 fixa no ESP32) por conta do curto espaço no ESP32
np.random.seed(42)
W1_GLOBAL = np.random.randn(HIDDEN_DIM, INPUT_DIM) * 0.05

# ─────────────────────────────────────────────
# Funções utilitárias
# ─────────────────────────────────────────────

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def calcular_lr(rodada):
    """LR decay exponencial: começa em LEARNING_RATE e decai até LR_MIN."""
    return max(LEARNING_RATE * (LR_DECAY ** (rodada - 1)), LR_MIN)

# ─────────────────────────────────────────────
# Treinamento e avaliação local (FedProx)
# ─────────────────────────────────────────────

def simular_treinamento_real(pesos_globais, indices, dataset, lr=0.05):
    W2        = pesos_globais.copy().reshape((OUTPUT_DIM, HIDDEN_DIM))
    W2_global = pesos_globais.copy().reshape((OUTPUT_DIM, HIDDEN_DIM))  # referência fixa

    # Embaralha e separa treino/validação — conjuntos disjuntos
    indices_copy = indices.copy()
    np.random.shuffle(indices_copy)
    indices_treino = indices_copy[:250]    # 250 amostras para treinar
    indices_val    = indices_copy[250:500]  # 250 amostras nunca vistas no treino

    # ── Treinamento — 5 épocas locais com FedProx ──
    for _ in range(5):
        for idx in indices_treino:
            img, label = dataset[idx]
            x = img.numpy().flatten()[:INPUT_DIM]

            A1    = np.maximum(0, np.dot(W1_GLOBAL, x))
            probs = softmax(np.dot(W2, A1))

            # Gradiente Cross-Entropy padrão
            dZ2 = probs.copy()
            dZ2[label] -= 1
            grad_ce = np.outer(dZ2, A1)

            # Termo proximal FedProx: puxa W2 de volta ao modelo global
            # Penaliza desvios grandes de W2_global — reduz client drift
            grad_prox = MU * (W2 - W2_global)

            # Atualização combinada: CE + proximal
            W2 -= lr * (grad_ce + grad_prox)

    # ── Avaliação em dados NUNCA vistos ──
    loss_total, hits, confidence_total = 0.0, 0, 0.0
    for idx in indices_val:
        img, label = dataset[idx]
        x = img.numpy().flatten()[:INPUT_DIM]
        A1    = np.maximum(0, np.dot(W1_GLOBAL, x))
        probs = softmax(np.dot(W2, A1))

        loss_total       += -np.log(probs[label] + 1e-7)
        confidence_total += probs[label] * 100.0
        if np.argmax(probs) == label:
            hits += 1

    n_val = len(indices_val)
    tempo_simulado_ms      = int(2200 + np.random.randint(-150, 150))
    memoria_simulada_bytes = int(204800 + np.random.randint(-10240, 10240))

    return (
        W2.flatten(),
        loss_total / n_val,
        (hits / n_val) * 100.0,
        confidence_total / n_val,
        tempo_simulado_ms,
        memoria_simulada_bytes
    )

# ─────────────────────────────────────────────
# Execução de cada rodada federada
# ─────────────────────────────────────────────

def executar_rodada(rodada, dataset, dict_clientes):
    lr_atual = calcular_lr(rodada)
    print(f"\n--- INICIANDO RODADA {rodada} | LR: {lr_atual:.5f} | MU: {MU} ---")

    try:
        response = requests.get(f"{SERVER_URL}/baixar_modelo")
        modelo_global = (
            np.frombuffer(response.content, dtype=np.float32)
            if response.status_code == 200
            else np.random.randn(HIDDEN_DIM * OUTPUT_DIM).astype(np.float32) * 0.01
        )
    except:
        modelo_global = np.random.randn(HIDDEN_DIM * OUTPUT_DIM).astype(np.float32) * 0.01

    for i in range(NUM_CLIENTES):
        indices = dict_clientes[i]
        pesos_up, loss, acc, conf, t_time, mem = simular_treinamento_real(
            modelo_global, indices, dataset, lr=lr_atual
        )

        headers = {
            'Content-Type':    'application/octet-stream',
            'X-Loss':          str(loss),
            'X-Accuracy':      str(acc),
            'X-Confidence':    str(conf),
            'X-Training-Time': str(t_time),
            'X-Memory-Usage':  str(mem),
            'X-Learning-Rate': str(lr_atual)
        }
        requests.post(f"{SERVER_URL}/atualizar_pesos", data=pesos_up.tobytes(), headers=headers)

        if i % 25 == 0 or i == NUM_CLIENTES - 1:
            print(f"[CLIENT {i}] Acc (val): {acc:.2f}% | Certeza: {conf:.2f}% | LR: {lr_atual:.5f}")

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    dataset, dict_clientes = preparar_mnist_non_iid(
        num_clientes=NUM_CLIENTES, classes_por_cliente=2
    )
    for r in range(1, RODADAS + 1):
        executar_rodada(r, dataset, dict_clientes)
        time.sleep(1)