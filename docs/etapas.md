

### 🏛️ Dossiê do Projeto FedCycle: Do Zero ao Aprendizado Distribuído

#### 1. O Desafio Inicial: O Gargalo de Memória e o Boot Loop
* **O Problema:** Tentamos carregar os pesos pré-treinados em PyTorch diretamente para a memória Flash do ESP32 usando um arquivo `.h` de 1 MB (com a matriz de características `W1`).
* **O Conflito:** Ao tentarmos ligar a pesada biblioteca `WiFi.h` do ESP32 ao mesmo tempo, o simulador Wokwi sofria um *DROM Cache Window Overflow* (colisão no mapa de endereçamento de memória virtual), resultando em um *Boot Loop* fatal antes mesmo da inicialização do código.
* **A Solução:** Adotamos uma estratégia de contorno em tempo de execução. Removemos a matriz gigante do arquivo estático e criamos um gerador **Xorshift** determinístico para popular a matriz `W1` dinamicamente na **PSRAM** (Memória RAM externa) durante o *setup*, permitindo que o Wi-Fi iniciasse sem colisões.

#### 2. O Motor TinyML: Treinamento On-Device (C++)
* Escrevemos a matemática de uma Rede Neural de duas camadas **do zero, em C++ puro**, sem depender de frameworks pesados como o TensorFlow Lite.
* **Forward Pass:** O microcontrolador faz a inferência calculando a saída através de funções ReLU e Softmax.
* **Backward Pass (Backpropagation):** Implementamos a retropropagação do erro (Cross-Entropy Loss), permitindo que o ESP32 recalcule os gradientes e atualize a própria matriz de pesos usando a Taxa de Aprendizado (Learning Rate).
* **Gerenciamento de Memória:** Implementamos um ciclo de alocação inteligente (`ps_malloc` e `free`), limpando as matrizes pesadas da RAM antes de acionar os drivers de rádio (Wi-Fi), evitando estouros de memória.

#### 3. A Infraestrutura Global: O Servidor Python
* Criamos um servidor **Flask** leve e focado em comunicação binária pura (`application/octet-stream`).
* **A Agregação (FedAvg):** Programamos a escola central para aguardar 3 "alunos" (clientes simulados). O servidor usa o **NumPy** para empilhar as matrizes recebidas (`np.stack`) e calcular a média exata de cada parâmetro (`np.mean`), gerando o Novo Modelo Global.
* **Visualização de Dados:** Integramos a biblioteca **Matplotlib** para ler os cabeçalhos customizados (`X-Loss` e `X-Accuracy`) enviados pelo ESP32 e desenhar, salvar e exportar gráficos PNG automáticos a cada rodada de comunicação.

#### 4. O Ciclo Federado Bidirecional (O Fechamento)
* **Downlink (GET):** O ESP32 foi ensinado a iniciar sua rotina perguntando à nuvem: *"Existe um modelo global novo?"*. Se o servidor retorna `200 OK`, ele baixa os 5 KB da matriz `W2` e substitui sua própria mente, herdando o conhecimento dos outros dispositivos.
* **Uplink (POST):** Após desligar o Wi-Fi, treinar localmente por 10 épocas, e religar a rede, o ESP32 devolve sua matriz `W2` atualizada para o servidor, contribuindo para a próxima média global.

#### 5. A Sacada de Mestre: Transfer Learning Extremo
* **O Diagnóstico:** Percebemos que na Rodada 2, o gráfico mostrou uma piora no aprendizado (a perda subiu). Identificamos o fenômeno da "Amnésia de Camada", pois o Xorshift resetava a matriz `W1` a cada reinicialização do Wokwi, dessincronizando-a do modelo global `W2`.
* **A Otimização Final:** Aplicamos o conceito de Congelamento de Camada (*Layer Freezing*). Retiramos a matriz `W1` do cálculo de *Backward Pass*, transformando-a em uma Lente de Extração de Características fixa. Passamos a treinar e agregar **apenas** a camada de decisão final (`W2`).
* **O Resultado:** Uma curva de aprendizado perfeita. O Loss médio caiu e a Acurácia subiu a cada rodada, provando matematicamente e visualmente o sucesso da arquitetura.

---

#### 6. Modelagem de Referência em PyTorch e Extração de Pesos
* **O que foi feito:** Desenvolvemos a arquitetura de referência `ModeloFedCycle` em PyTorch (`fedcycle_model.py`), dividindo formalmente o fluxo de processamento entre a extração convolucional de características (camada de representação com Conv2d, ReLU e MaxPool2d, totalizando 2048 saídas) e a cabeça classificadora linear (dois blocos lineares e ReLU).
* **A Ponte para C++:** Implementamos o script `extract_weight.py` para automatizar a extração das matrizes de pesos (`W1`, `b1`, `W2`, `b2`) do PyTorch e convertê-las em constantes formatadas em C++ (`pesos_classificacao.h`), prontas para serem embarcadas diretamente no ESP32.

#### 7. Geração de Dados Sintéticos para Destilação (GAN Data-Free)
* **O Conceito:** Para viabilizar o alinhamento de representações sem tráfego de dados sensíveis ou imagens reais dos clientes, implementamos o `GeradorDataFree` em PyTorch (`gan_generator.py`), capaz de sintetizar mapas de características de 2048 dimensões a partir de ruído aleatório.
* **Métricas de Perda:** Adotamos as funções de perda propostas pelo ecossistema FedBKD:
  * **Perda de Confiança ($L_{oh}$):** Treina o gerador para criar features sintéticas que maximizam a certeza do discriminador (classificador).
  * **Perda de Diversidade ($L_{ms}$):** Pune o gerador caso ele sofra colapso de modo, garantindo variabilidade nos dados sintéticos em resposta à alteração do ruído de entrada.

#### 8. Ciclo de Destilação Bidirecional de Conhecimento (FedBKD)
* **O que foi feito:** Codificamos em `bidirectional_distillation.py` o ciclo completo de transferência mútua de conhecimento em nível de features usando a Divergência de Kullback-Leibler (KL) com suavização por temperatura.
* **O Fluxo:**
  * **Global $\to$ Local (Fase 1):** O modelo global central atua como professor, ensinando e refinando o extrator local de características (representação) ao longo de 4 épocas (camada classificadora congelada).
  * **Local $\to$ Global (Fase 2):** O modelo local atualizado ensina o modelo global durante 1 época, permitindo que a nuvem incorpore as particularidades das representações locais aprendidas pelos clientes.

#### 9. Particionamento Não-IID de Dados e Exportação Wokwi
* **O Desafio Realístico:** Cenários reais de aprendizado federado lidam com dados que não são identicamente distribuídos (Não-IID).
* **A Solução:** Implementamos a lógica de partição Não-IID parametrizável em `datasetConf.py` para CIFAR-10 e MNIST (configurado por padrão para 100 clientes, com apenas 2 classes por cliente).
* **Simulação Local:** Desenvolvemos o script `generate_client_headers.py` para gerar arquivos individuais `.h` (`dados_cliente{i}.h`) contendo matrizes de pixels reais. Estes cabeçalhos são carregados diretamente no firmware, simulando o conjunto de dados proprietário e isolado de cada microcontrolador.

#### 10. Pipeline de Exportação ONNX para TinyML
* **O que foi feito:** Desenvolvemos o script `export_tinyML.py` para exportar a representação congelada e a cabeça classificadora do `ModeloFedCycle` como arquivos ONNX separados (`representacao_congelada.onnx` e `classificacao_treinavel.onnx`).
* **Utilidade:** Permite o mapeamento do modelo para frameworks industriais de inferência em microcontroladores (como Apache TVM, TensorFlow Lite Micro ou STM32Cube.AI) para posterior otimização específica de hardware.

#### 11. Escalabilidade de Rede e Simulação com 50 Clientes
* **O que foi feito:** Escalamos a infraestrutura para suportar e simular concorrentemente **50 clientes** ao longo de 15 rodadas federadas.
* **Firmware do ESP32:** O código `main.cpp` no firmware realiza conexões Wi-Fi dinâmicas, faz download (GET) do modelo global do servidor, executa o treinamento local com matrizes alocadas na PSRAM e pesos congelados de $W_1$, e devolve (POST) a atualização da matriz $W_2$ junto a cabeçalhos HTTP com métricas (`X-Loss` e `X-Accuracy`).
* **Agregação e Visualização:** O servidor Flask (`server/server.py`) centraliza e faz a média dos pesos (FedAvg) dos 50 clientes usando NumPy, salvando os checkpoints em `.npy` e gerando relatórios gráficos automáticos (`resultado_fedavg_rodada_{rodada}.png`) da perda e acurácia médias em `assets/results/`. O script `simulate_50_clients.py` simula as interações simultâneas de todos os clientes com o servidor.

---

### 🎯 Próximos Passos Sugeridos

Com a infraestrutura de simulação, modelagem, exportação e firmware totalmente construída, os próximos passos do projeto englobam:

1. **Validação em Placa Física:** Executar o firmware compilado em um lote de placas ESP32-WROVER físicas conectadas em uma rede Wi-Fi local para analisar o consumo energético real e o tempo de treinamento.
2. **Ativação do FedBKD no Servidor:** Integrar o script de destilação bidirecional com GAN (`bidirectional_distillation.py`) diretamente nas rotas do servidor Flask, permitindo destilação real de características em cada rodada em vez de apenas agregação simples via FedAvg.
3. **Otimização de Modelo (Quantização INT8):** Aplicar técnicas de quantização nos pesos para avaliar o impacto na precisão versus a redução no tempo de treinamento e uso de RAM no microcontrolador.
4. **Redação do Artigo Científico:** Reunir os resultados de desempenho coletados na simulação de 50 clientes (gráficos gerados em `assets/results/`) para a elaboração formal do relatório/artigo científico sobre a viabilidade de treinamento on-device com FedCycle.