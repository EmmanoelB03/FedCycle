#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Importando apenas W2 e b2 do professor (5KB)
#include "pesos_classificacao.h"
#include "dados_cliente0.h"

const int INPUT_DIM = 784;
const int HIDDEN_DIM = 128;
const int OUTPUT_DIM = 10;


float* train_W1;
float* train_b1;
float* train_W2;
float* train_b2;

float A1[HIDDEN_DIM];
float Z2[OUTPUT_DIM];
float A2[OUTPUT_DIM];

// Só precisamos do erro da última camada agora!
float dZ2[OUTPUT_DIM];

const float learning_rate = 0.01;

// --- GERADOR DETERMINÍSTICO PARA W1 e b1 (Evita o Boot Loop do Wokwi) ---
void inicializar_pesos_deterministicos(float* W, int tamanho, uint32_t semente) {
    uint32_t state = semente;
    for (int i = 0; i < tamanho; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        W[i] = ((float)(state & 0xFFFF) / 65535.0f - 0.5f) * 0.1f;
    }
}

void forward_pass(const float* X) {
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float sum = train_b1[i];
        for (int j = 0; j < INPUT_DIM; j++) {
            sum += train_W1[i * INPUT_DIM + j] * X[j];
        }
        A1[i] = (sum > 0) ? sum : 0;
    }

    float max_val = -1e9;
    for (int i = 0; i < OUTPUT_DIM; i++) {
        float sum = train_b2[i];
        for (int j = 0; j < HIDDEN_DIM; j++) {
            sum += train_W2[i * HIDDEN_DIM + j] * A1[j];
        }
        Z2[i] = sum;
        if (sum > max_val) max_val = sum;
    }

    float sum_exp = 0.0;
    for (int i = 0; i < OUTPUT_DIM; i++) {
        A2[i] = exp(Z2[i] - max_val);
        sum_exp += A2[i];
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
        A2[i] /= sum_exp;
    }
}

float compute_loss(int true_label) {
    return -log(A2[true_label] + 1e-7);
}

// 🔥 A MÁGICA DO TRANSFER LEARNING: W1 AGORA É FIXA! 🔥
void backward_pass(const float* X, int true_label) {
    // 1. Calcula o erro na saída
    for (int i = 0; i < OUTPUT_DIM; i++) {
        dZ2[i] = A2[i];
        if (i == true_label) dZ2[i] -= 1.0; 
    }

    // 2. Atualiza APENAS os pesos da última camada (W2 e b2)
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float grad_w = dZ2[i] * A1[j];
            train_W2[i * HIDDEN_DIM + j] -= learning_rate * grad_w;
        }
        train_b2[i] -= learning_rate * dZ2[i];
    }
}

void setup() {
    Serial.begin(115200);
    delay(2000); 

    if (!psramFound()) {
        Serial.println("ERRO: PSRAM não encontrada!");
        return;
    }

    // 1. Aloca os pequenos e copia a base da Flash
    train_W2 = (float*) ps_malloc(HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    train_b2 = (float*) ps_malloc(OUTPUT_DIM * sizeof(float));
    memcpy(train_W2, W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    memcpy(train_b2, b2, OUTPUT_DIM * sizeof(float));

    // =========================================================
    // ETAPA 1: O DOWNLINK (Baixar o Cérebro Global)
    // =========================================================
    Serial.println("\n--- Conectando ao Wi-Fi para buscar o Modelo Global ---");
    WiFi.begin("Wokwi-GUEST", "", 6);
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
    
    HTTPClient http;
    http.begin("http://192.168.100.22:5000/baixar_modelo"); 
    int httpCode = http.GET();

    if (httpCode == 200) {
        Serial.println("✅ Modelo Global encontrado! Baixando e substituindo W2...");
        WiFiClient* stream = http.getStreamPtr();
        stream->readBytes((uint8_t*)train_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    } else {
        Serial.println("⚠️ Servidor retornou 404 (Primeira Rodada). Usando os pesos de fábrica.");
    }
    http.end();
    
    // Desliga o Wi-Fi para limpar o buffer de rede e não travar o Wokwi!
    WiFi.disconnect(true);
    Serial.println("Wi-Fi Desligado temporariamente para poupar RAM.");

    // =========================================================
    // ETAPA 2: TREINAMENTO LOCAL (On-Device)
    // =========================================================
    train_W1 = (float*) ps_malloc(INPUT_DIM * HIDDEN_DIM * sizeof(float));
    train_b1 = (float*) ps_malloc(HIDDEN_DIM * sizeof(float));
    
    inicializar_pesos_deterministicos(train_W1, INPUT_DIM * HIDDEN_DIM, 42);
    inicializar_pesos_deterministicos(train_b1, HIDDEN_DIM, 43);

    Serial.println("\n--- Iniciando Treinamento Local ---");
    
    float perda_final = 0.0;
    float acuracia_final = 0.0;
    float certeza_final = 0.0;
    
    uint32_t tempo_inicio = millis();
    
    for (int epoca = 1; epoca <= 10; epoca++) {
        float perda_acumulada = 0.0;
        float certeza_acumulada = 0.0;
        int acertos = 0;

        for (int s = 0; s < num_amostras_locais; s++) {
            float input_features[INPUT_DIM];
            for (int p = 0; p < INPUT_DIM; p++) {
                input_features[p] = (float)dados_pixels[s][p] / 255.0f;
            }

            forward_pass(input_features);
            perda_acumulada += compute_loss(labels_locais[s]);
            backward_pass(input_features, labels_locais[s]);
            certeza_acumulada += A2[labels_locais[s]] * 100.0;

            // Calcula acerto real (argmax do softmax)
            int predicao = 0;
            float max_val = A2[0];
            for (int i = 1; i < OUTPUT_DIM; i++) {
                if (A2[i] > max_val) {
                    max_val = A2[i];
                    predicao = i;
                }
            }
            if (predicao == labels_locais[s]) {
                acertos++;
            }
        }
        
        perda_final = perda_acumulada / num_amostras_locais;
        certeza_final = certeza_acumulada / num_amostras_locais;
        acuracia_final = ((float)acertos / num_amostras_locais) * 100.0f;
        
        Serial.printf("Época %d | Perda Média: %.4f | Acurácia Real: %.2f%% | Certeza Média: %.2f%%\n", epoca, perda_final, acuracia_final, certeza_final);
    }
    
    uint32_t tempo_total = millis() - tempo_inicio;
    uint32_t memoria_livre = ESP.getFreeHeap();
    
    Serial.println("Limpando a memória pesada (W1) para dar espaço ao Wi-Fi...");
    free(train_W1);
    free(train_b1);

    // =========================================================
    // ETAPA 3: O UPLINK (Enviar o novo aprendizado)
    // =========================================================
    Serial.println("\n--- Religa Wi-Fi para enviar Cliente Update ---");
    WiFi.begin("Wokwi-GUEST", "", 6);
    while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }

    http.begin("http://192.168.100.22:5000/atualizar_pesos"); 
    http.addHeader("Content-Type", "application/octet-stream");
    http.addHeader("X-Loss", String(perda_final));
    http.addHeader("X-Accuracy", String(acuracia_final));
    http.addHeader("X-Confidence", String(certeza_final));
    http.addHeader("X-Training-Time", String(tempo_total));
    http.addHeader("X-Memory-Usage", String(memoria_livre));
    
    int postCode = http.POST((uint8_t*)train_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    Serial.printf("Envio Concluído! Código HTTP: %d\n", postCode);
    http.end();
}

void loop() { }