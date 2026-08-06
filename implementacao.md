# 🛠️ Plano de Implementação — Melhorias FedCycle

> Baseado na análise das métricas das 15 rodadas federadas com 50 clientes (CIFAR-10 Non-IID).  
> Cada seção é independente e pode ser implementada incrementalmente.

---

## 📌 Índice

1. [W1 com Features Pré-Treinadas (MobileNetV1)](#1-w1-com-features-pré-treinadas-mobilenetv1)
2. [Aumento de Amostras Locais](#2-aumento-de-amostras-locais)
3. [Alinhamento de Épocas Locais](#3-alinhamento-de-épocas-locais)
4. [Label Smoothing na Loss](#4-label-smoothing-na-loss)
5. [Learning Rate com Decaimento por Rodada](#5-learning-rate-com-decaimento-por-rodada)
6. [FedProx em vez de FedAvg Puro](#6-fedprox-em-vez-de-fedavg-puro)

---

## 1. W1 com Features Pré-Treinadas (MobileNetV1)

**Problema:** A W1 atual é inicializada aleatoriamente com seed fixa. Isso limita estruturalmente o teto de acurácia e é a principal causa do gap acurácia/certeza (~27pp).

**Objetivo:** Substituir a W1 aleatória por uma camada congelada de MobileNetV1 exportada via `export_tinyML.py`, usando o pipeline já existente em `extract_weight.py`.

### 1.1 Exportar features do MobileNetV1 via `export_tinyML.py`

```python
# export_tinyML.py — adicionar exportação de W1 pré-treinada
import torch
import torchvision.models as models
import numpy as np

def exportar_w1_mobilenet(output_path="pesos_w1_mobilenet.npy"):
    """
    Extrai a primeira camada convolucional do MobileNetV1 pré-treinado
    e a converte para uma matriz densa compatível com o firmware (INPUT_DIM x HIDDEN_DIM).
    """
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.eval()

    # Usar a saída do avg_pool como extrator fixo (feature vector de 1280-d)
    # Reduzimos para 128-d via projeção linear treinada
    extrator = torch.nn.Sequential(*list(mobilenet.features.children())[:5])

    # Entrada dummy: imagem CIFAR-10 (3x32x32)
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        features = extrator(dummy)

    print(f"Shape das features extraídas: {features.shape}")
    # Achata e exporta como matriz W1 compatível
    w1_np = features.squeeze().detach().numpy().flatten()
    np.save(output_path, w1_np)
    print(f"W1 pré-treinada salva em: {output_path}")

if __name__ == "__main__":
    exportar_w1_mobilenet()
```

### 1.2 Atualizar `simulate_50_clients.py` para carregar W1 do arquivo

```python
# simulate_50_clients.py — substituir W1_GLOBAL aleatória

import numpy as np

# ANTES (remover):
# np.random.seed(42)
# W1_GLOBAL = np.random.randn(HIDDEN_DIM, INPUT_DIM) * 0.05

# DEPOIS (adicionar):
W1_PATH = "pesos_w1_mobilenet.npy"

def carregar_w1_pretrained(path, hidden_dim, input_dim):
    """Carrega W1 pré-treinada ou cai de volta para aleatória com aviso."""
    try:
        w1_flat = np.load(path).astype(np.float32)
        W1 = w1_flat[:hidden_dim * input_dim].reshape(hidden_dim, input_dim)
        print(f"[W1] ✅ Features pré-treinadas carregadas de '{path}'")
        return W1
    except FileNotFoundError:
        print(f"[W1] ⚠️ '{path}' não encontrado. Usando W1 aleatória como fallback.")
        np.random.seed(42)
        return np.random.randn(hidden_dim, input_dim) * 0.05

W1_GLOBAL = carregar_w1_pretrained(W1_PATH, HIDDEN_DIM, INPUT_DIM)
```

### 1.3 Atualizar o firmware `main.cpp` para receber W1 via downlink

```cpp
// main.cpp — Etapa 1 (DOWNLINK): baixar também W1 do servidor
// Adicionar nova rota no servidor: GET /baixar_w1

HTTPClient http_w1;
http_w1.begin("http://192.168.100.22:5000/baixar_w1");
int codeW1 = http_w1.GET();

if (codeW1 == 200) {
    WiFiClient* stream = http_w1.getStreamPtr();
    stream->readBytes((uint8_t*)train_W1, INPUT_DIM * HIDDEN_DIM * sizeof(float));
    Serial.println("✅ W1 pré-treinada baixada com sucesso!");
} else {
    Serial.println("⚠️ W1 não disponível. Usando inicialização determinística.");
    inicializar_pesos_deterministicos(train_W1, INPUT_DIM * HIDDEN_DIM, 42);
}
http_w1.end();
```

### 1.4 Adicionar rota `/baixar_w1` no `server.py`

```python
# server.py — nova rota para servir W1 pré-treinada
import numpy as np

W1_PRETRAINED_PATH = "../assets/models/pesos_w1_mobilenet.npy"
w1_pretrained = None

def carregar_w1():
    global w1_pretrained
    try:
        w1_pretrained = np.load(W1_PRETRAINED_PATH).astype(np.float32)
        print(f"[SERVER] W1 pré-treinada carregada: {w1_pretrained.shape}")
    except FileNotFoundError:
        print("[SERVER] ⚠️ W1 pré-treinada não encontrada.")

@app.route('/baixar_w1', methods=['GET'])
def enviar_w1():
    if w1_pretrained is None:
        return "W1 não disponível", 404
    return w1_pretrained.tobytes(), 200, {'Content-Type': 'application/octet-stream'}

# Chamar no início:
# carregar_w1()
```

**Resultado esperado:** Certeza Média deve subir de ~27% para acima de 40%, e Acurácia Real pode chegar a 60–70% nas mesmas 15 rodadas.

---

## 2. Aumento de Amostras Locais

**Problema:** Cada cliente treina em apenas 20 amostras por rodada (`amostras_indices = indices[:20]`), o que é insuficiente para o CIFAR-10 produzir gradientes representativos.

**Objetivo:** Aumentar para 50–100 amostras por cliente por rodada.

### 2.1 Alterar `simulate_50_clients.py`

```python
# simulate_50_clients.py — função simular_treinamento_real

def simular_treinamento_real(pesos_globais, indices, dataset, num_amostras=80):
    # ANTES:
    # amostras_indices = indices[:20]

    # DEPOIS: amostragem aleatória sem reposição para maior diversidade
    n = min(num_amostras, len(indices))
    amostras_indices = np.random.choice(indices, size=n, replace=False)

    # ... restante da função sem alteração
```

### 2.2 Ajustar o firmware `main.cpp`

No firmware, o número de amostras é determinado pelo arquivo `dados_cliente0.h`. Para aumentar, atualizar a chamada em `datasetConf.py`:

```python
# datasetConf.py — exportar_amostra_para_c
# ANTES:
# exportar_amostra_para_c(dataset, indices_cliente_0, num_amostras=5)

# DEPOIS:
exportar_amostra_para_c(
    dataset,
    indices_cliente_0,
    num_amostras=50,          # Aumentado de 5 para 50
    nome_arquivo="dados_cliente0.h",
    max_pixels=2048           # Mantém compatibilidade com INPUT_DIM=784 + padding
)
```

> ⚠️ **Atenção à memória do ESP32:** 50 amostras × 2048 bytes = ~100 KB em Flash. Verifique o espaço disponível antes de aumentar além de 50.

**Resultado esperado:** Gradientes mais estáveis, Loss convergindo mais rápido por rodada.

---

## 3. Alinhamento de Épocas Locais

**Problema:** O firmware faz **10 épocas** locais, mas o simulador faz apenas **1 passagem** pelos dados (loop único em `simular_treinamento_real`). Isso causa heterogeneidade entre clientes reais e simulados.

**Objetivo:** Alinhar o simulador para também executar múltiplas épocas.

### 3.1 Refatorar `simulate_50_clients.py`

```python
# simulate_50_clients.py

LOCAL_EPOCHS = 10  # Alinhado com o firmware (main.cpp faz 10 épocas)

def simular_treinamento_real(pesos_globais, indices, dataset, num_amostras=80):
    W2 = pesos_globais.copy().reshape((OUTPUT_DIM, HIDDEN_DIM))

    amostras_indices = np.random.choice(indices, size=min(num_amostras, len(indices)), replace=False)

    loss_final = 0
    hits_final = 0
    confidence_final = 0.0

    for epoca in range(LOCAL_EPOCHS):
        loss_epoca = 0
        hits_epoca = 0
        conf_epoca = 0.0

        # Shuffle a cada época para evitar overfitting na ordem
        np.random.shuffle(amostras_indices)

        for idx in amostras_indices:
            img, label = dataset[idx]
            x = img.numpy().flatten()[:INPUT_DIM]

            A1 = np.maximum(0, np.dot(W1_GLOBAL, x))
            logits = np.dot(W2, A1)
            probs = softmax(logits)

            loss_epoca += -np.log(probs[label] + 1e-7)
            conf_epoca += probs[label] * 100.0
            if np.argmax(probs) == label:
                hits_epoca += 1

            dZ2 = probs.copy()
            dZ2[label] -= 1
            grad_W2 = np.outer(dZ2, A1)
            W2 -= LEARNING_RATE * grad_W2

        # Guarda métricas da última época (igual ao firmware)
        loss_final = loss_epoca / len(amostras_indices)
        hits_final = hits_epoca
        confidence_final = conf_epoca / len(amostras_indices)

    # Tempo simulado proporcional às épocas
    tempo_simulado_ms = int((2200 * LOCAL_EPOCHS / 10) + np.random.randint(-150, 150))
    memoria_simulada_bytes = int(204800 + np.random.randint(-10240, 10240))

    return (
        W2.flatten(),
        loss_final,
        (hits_final / len(amostras_indices)) * 100.0,
        confidence_final,
        tempo_simulado_ms,
        memoria_simulada_bytes
    )
```

**Resultado esperado:** Simulador e firmware produzindo distribuições de pesos equivalentes, reduzindo o viés no FedAvg.

---

## 4. Label Smoothing na Loss

**Problema:** A certeza média (~27%) indica Softmax com distribuição muito achatada. O modelo "sabe" a resposta certa, mas não concentra probabilidade nela. Label Smoothing ajuda a calibrar melhor as probabilidades, paradoxalmente aumentando a certeza ao regularizar a confiança excessiva em exemplos errados.

**Objetivo:** Substituir a Cross-Entropy pura por Cross-Entropy com Label Smoothing (ε = 0.1).

### 4.1 Atualizar `simulate_50_clients.py`

```python
# simulate_50_clients.py — nova função de loss com smoothing

LABEL_SMOOTHING = 0.1  # ε — valores entre 0.05 e 0.15 são típicos

def cross_entropy_smoothed(probs, label, num_classes=10, epsilon=LABEL_SMOOTHING):
    """
    Cross-Entropy com Label Smoothing.
    Distribui epsilon uniformemente entre todas as classes,
    e (1 - epsilon) fica na classe correta.
    """
    # Alvo suavizado: (epsilon / K) para todas, mais (1 - epsilon) na correta
    smooth_target = np.full(num_classes, epsilon / num_classes)
    smooth_target[label] += (1.0 - epsilon)

    # Loss: -sum(target * log(prob))
    loss = -np.sum(smooth_target * np.log(probs + 1e-7))
    return loss, smooth_target

# Dentro do loop de treinamento, substituir:
# loss_total += -np.log(probs[label] + 1e-7)
# Por:
loss_val, smooth_target = cross_entropy_smoothed(probs, label)
loss_total += loss_val

# E o gradiente dZ2 passa a usar o alvo suavizado:
# dZ2 = probs - smooth_target  (em vez de dZ2[label] -= 1)
dZ2 = probs - smooth_target
```

### 4.2 Atualizar o firmware `main.cpp`

```cpp
// main.cpp — backward_pass com Label Smoothing

const float LABEL_SMOOTHING = 0.1f;
const float smooth_floor = LABEL_SMOOTHING / OUTPUT_DIM; // ε/K

void backward_pass_smoothed(const float* X, int true_label) {
    for (int i = 0; i < OUTPUT_DIM; i++) {
        // Alvo suavizado
        float target = smooth_floor;
        if (i == true_label) target += (1.0f - LABEL_SMOOTHING);

        dZ2[i] = A2[i] - target;  // Gradiente com smoothing
    }

    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            train_W2[i * HIDDEN_DIM + j] -= learning_rate * dZ2[i] * A1[j];
        }
        train_b2[i] -= learning_rate * dZ2[i];
    }
}

// No loop de treinamento, substituir backward_pass() por backward_pass_smoothed()
```

**Resultado esperado:** Certeza Média deve subir 3–8 pontos percentuais, com melhor calibração entre acurácia e confiança.

---

## 5. Learning Rate com Decaimento por Rodada

**Problema:** O learning rate fixo de 0,01 não se adapta ao estágio do treinamento. Nas rodadas iniciais, um LR maior acelera a convergência. Nas finais, um LR menor estabiliza e evita overshooting.

**Objetivo:** Implementar decaimento exponencial do LR a cada rodada federada.

### 5.1 Atualizar `simulate_50_clients.py`

```python
# simulate_50_clients.py — LR dinâmico por rodada

BASE_LR = 0.05       # LR inicial mais alto
LR_DECAY = 0.92      # Fator de decaimento por rodada
MIN_LR = 0.005       # LR mínimo (floor)

def calcular_lr(rodada):
    lr = BASE_LR * (LR_DECAY ** (rodada - 1))
    return max(lr, MIN_LR)

# Na função executar_rodada, passar o LR calculado:
def executar_rodada(rodada, dataset, dict_clientes):
    lr_atual = calcular_lr(rodada)
    print(f"[LR] Rodada {rodada} | Learning Rate: {lr_atual:.5f}")

    # Passar lr_atual para simular_treinamento_real
    for i in range(NUM_CLIENTES):
        indices = dict_clientes[i]
        pesos_up, loss, acc, conf, t_time, mem = simular_treinamento_real(
            modelo_global, indices, dataset, learning_rate=lr_atual
        )
        # ...

# simular_treinamento_real recebe learning_rate como parâmetro:
def simular_treinamento_real(pesos_globais, indices, dataset, num_amostras=80, learning_rate=0.01):
    # ... usar learning_rate no lugar de LEARNING_RATE fixo
    W2 -= learning_rate * grad_W2
```

### 5.2 Enviar LR atual como cabeçalho HTTP

```python
# simulate_50_clients.py — adicionar X-Learning-Rate nos headers
headers = {
    'Content-Type': 'application/octet-stream',
    'X-Loss': str(loss),
    'X-Accuracy': str(acc),
    'X-Confidence': str(conf),
    'X-Training-Time': str(t_time),
    'X-Memory-Usage': str(mem),
    'X-Learning-Rate': str(lr_atual)   # Novo cabeçalho
}
```

### 5.3 Registrar LR no `server.py` e incluir no gráfico

```python
# server.py — registrar histórico de LR e adicionar ao painel

historico_lr = []

# Em receber_pesos():
lr_cliente = float(request.headers.get('X-Learning-Rate', 0.01))
# ... após FedAvg:
historico_lr.append(np.mean([item.get('lr', 0.01) for item in pesos_recebidos_buffer]))

# Em plotar_graficos() — adicionar subplot ou linha secundária no gráfico de Loss:
ax_lr = axs[0, 0].twinx()
ax_lr.plot(x_rodadas, historico_lr, color='orange', linestyle=':', linewidth=1.5, label='LR')
ax_lr.set_ylabel('Learning Rate', color='orange')
ax_lr.legend(loc='upper right')
```

**Resultado esperado:** Convergência 2–3× mais rápida nas primeiras rodadas, com estabilização nas rodadas finais.

---

## 6. FedProx em vez de FedAvg Puro

**Problema:** Com distribuição Non-IID severa (S=2 classes por cliente), o FedAvg puro sofre com *client drift* — cada cliente otimiza para sua distribuição local, e a média global pode divergir ou convergir lentamente.

**Objetivo:** Adicionar o termo proximal do FedProx (μ · ‖w − w_global‖²) ao gradiente local, ancorando os clientes ao modelo global e reduzindo o drift.

### 6.1 Implementar FedProx no `simulate_50_clients.py`

```python
# simulate_50_clients.py — adicionar termo proximal

MU = 0.01  # Coeficiente proximal (valores típicos: 0.001 a 0.1)

def simular_treinamento_real(pesos_globais, indices, dataset, num_amostras=80, learning_rate=0.01):
    W2 = pesos_globais.copy().reshape((OUTPUT_DIM, HIDDEN_DIM))
    W2_global = pesos_globais.copy().reshape((OUTPUT_DIM, HIDDEN_DIM))  # Referência fixa

    # ... forward pass e backprop normais, depois adicionar o termo proximal:

    for idx in amostras_indices:
        # ... forward pass ...
        # ... gradiente padrão ...
        grad_W2 = np.outer(dZ2, A1)

        # Termo proximal FedProx: puxa W2 de volta ao global
        grad_proximal = MU * (W2 - W2_global)

        # Atualização com FedProx
        W2 -= learning_rate * (grad_W2 + grad_proximal)
```

### 6.2 Implementar FedProx no firmware `main.cpp`

```cpp
// main.cpp — backward_pass_fedprox

const float MU = 0.01f; // Coeficiente proximal

// Salvar W2 global antes do treino (após o downlink):
float* W2_global_ref = (float*) ps_malloc(HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
memcpy(W2_global_ref, train_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));

void backward_pass_fedprox(const float* X, int true_label) {
    // Gradiente Cross-Entropy padrão
    for (int i = 0; i < OUTPUT_DIM; i++) {
        dZ2[i] = A2[i];
        if (i == true_label) dZ2[i] -= 1.0f;
    }

    // Atualiza W2 com termo proximal
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float grad_ce = dZ2[i] * A1[j];
            float grad_prox = MU * (train_W2[i * HIDDEN_DIM + j] - W2_global_ref[i * HIDDEN_DIM + j]);
            train_W2[i * HIDDEN_DIM + j] -= learning_rate * (grad_ce + grad_prox);
        }
        train_b2[i] -= learning_rate * dZ2[i];
    }
}

// Liberar ao final:
// free(W2_global_ref);
```

### 6.3 Monitorar o impacto no `server.py`

```python
# server.py — adicionar métrica de divergência de pesos (client drift)

def calcular_divergencia(lista_pesos, modelo_global):
    """Calcula a divergência média entre os pesos locais e o modelo global anterior."""
    divergencias = [np.linalg.norm(p - modelo_global) for p in lista_pesos]
    return np.mean(divergencias)

# Em receber_pesos(), após FedAvg:
if modelo_global_w2 is not None:
    drift = calcular_divergencia(lista_pesos, modelo_global_w2)
    print(f"[SERVER] 📐 Client Drift Médio: {drift:.4f}")
    historico_drift.append(drift)

# Adicionar ao painel gráfico como 5º subplot ou anotação
```

**Resultado esperado:** Redução do client drift em distribuições Non-IID severas, com ganho de 3–7% de acurácia nas rodadas 5–15.

---

## 🗺️ Ordem de Implementação Recomendada

```
Prioridade Alta (maior impacto na acurácia/certeza):
  1 → W1 Pré-Treinada
  2 → Aumento de Amostras Locais

Prioridade Média (estabilidade e convergência):
  3 → Alinhamento de Épocas
  5 → Learning Rate com Decaimento

Prioridade Baixa (refinamento fino):
  4 → Label Smoothing
  6 → FedProx
```

> **Dica:** Implemente e valide uma melhoria por vez, rodando as 15 rodadas completas e comparando os painéis gerados em `assets/results/`. Isso facilita isolar o impacto de cada mudança.