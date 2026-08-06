# 📊 Novas Métricas de Desempenho e Monitoramento TinyML no FedCycle

Para aprimorar o monitoramento do treinamento federado e entender o comportamento de dispositivos de borda com restrições de hardware, reestruturamos as métricas enviadas pelos clientes (ESP32 / simulador) e implementamos um painel multi-métricas completo no servidor global.

---

## 📈 Resumo das Métricas Implementadas

Abaixo está o detalhamento das métricas agora rastreadas pelo **FedCycle**:

| Métrica | Cabeçalho HTTP | Descrição | Importância no TinyML / FL |
| :--- | :--- | :--- | :--- |
| **Perda Média (Loss)** | `X-Loss` | Média da função de perda Cross-Entropy calculada localmente no dispositivo. | Avaliar a convergência do aprendizado em cada rodada. |
| **Acurácia Real** | `X-Accuracy` | Porcentagem real de predições corretas (onde a classe de maior probabilidade é a classe alvo). | Medir a taxa de acerto real do classificador. |
| **Certeza Média (Confiança)** | `X-Confidence` | A média das probabilidades associadas à classe correta pelo Softmax. | Mensurar o nível de certeza do modelo, mesmo quando ele erra ou acerta. |
| **Tempo de Treinamento** | `X-Training-Time` | O tempo total gasto para rodar o treinamento local (em milissegundos). | Monitorar a eficiência de processamento nos microcontroladores. |
| **Memória Heap Livre** | `X-Memory-Usage` | A quantidade de memória RAM (Heap) livre no ESP32 após o treino (em bytes). | Garantir a segurança operacional e evitar estouros de memória. |

---

## 🛠️ Arquivos Modificados

As alterações foram integradas de ponta a ponta no ecossistema do projeto:

1. **Firmware C++ (`main.cpp`)**:
   - Atualizado para medir o tempo exato de treino usando `millis()`.
   - Modificado para coletar a memória heap livre operacional por meio de `ESP.getFreeHeap()`.
   - Correção na lógica de acurácia: a taxa de acerto real (argmax do Softmax) e a certeza média agora são calculadas separadamente e enviadas de forma distinta.
   - Link do arquivo: [main.cpp](file:///home/emmanoel/pesquisa/firmware/src/main.cpp#L139-L184)

2. **Simulador de Clientes (`simulate_50_clients.py`)**:
   - Atualizado para simular e enviar todos os novos cabeçalhos (`X-Confidence`, `X-Training-Time` e `X-Memory-Usage`) correspondentes a parâmetros realistas de dispositivos ESP32 com PSRAM.
   - Link do arquivo: [simulate_50_clients.py](file:///home/emmanoel/pesquisa/scripts/simulate_50_clients.py#L32-L82)

3. **Servidor Global (`server.py`)**:
   - Rota `/atualizar_pesos` estendida para capturar e calcular a média federada de todas as novas métricas da rodada.
   - A função de plotagem foi redesenhada de um layout simples de 2 colunas para uma **grade 2x2 altamente profissional**, exibindo simultaneamente o comportamento do modelo e a eficiência dos recursos.
   - Link do arquivo: [server.py](file:///home/emmanoel/pesquisa/server/server.py#L16-L122)

---

## 🎨 O Novo Painel Gráfico (Grade 2x2)

A cada rodada federada concluída, o servidor gera automaticamente um gráfico consolidado com resolução otimizada (DPI = 150) em `assets/results/resultado_fedavg_rodada_X.png`.

Abaixo está o gráfico gerado com sucesso após a execução completa de **30 rodadas federadas de 50 clientes**:

![Painel de Métricas da Rodada 30](../assets/results/resultado_fedavg_rodada_30.png)

### O painel contém:
* **Top-Left (Perda Médio - Loss):** Curva de redução de erro global em vermelho.
* **Top-Right (Acurácia Real vs Certeza Média):** Gráfico comparativo que contrasta a taxa de acerto real com a probabilidade (confiança) média do modelo na classe correta.
* **Bottom-Left (Tempo de Treinamento):** Gráfico de acompanhamento de velocidade operacional (em segundos), vital para rastrear gargalos ou otimizações.
* **Bottom-Right (Heap Livre):** Gráfico de telemetria de RAM livre (em KB), garantindo visibilidade sobre o consumo em dispositivos restritos.
