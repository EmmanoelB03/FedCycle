from flask import Flask, request
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# --- Configurações do Aprendizado Federado ---
NUM_CLIENTES_ESPERADOS = 3   
pesos_recebidos_buffer = []  
rodada_atual = 1             
modelo_global_w2 = None 

# --- Histórico para os Gráficos ---
historico_loss = []
historico_acuracia = []

def plotar_graficos():
    """Gera e salva o gráfico de evolução do modelo global"""
    plt.figure(figsize=(10, 5))
    
    # Gráfico de Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(historico_loss) + 1), historico_loss, marker='o', color='red')
    plt.title('Evolução da Perda (Loss)')
    plt.xlabel('Rodadas de Comunicação')
    plt.ylabel('Loss Médio')
    plt.grid(True)
    
    # Gráfico de Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(historico_acuracia) + 1), historico_acuracia, marker='o', color='blue')
    plt.title('Evolução da Certeza (Classe Alvo)')
    plt.xlabel('Rodadas de Comunicação')
    plt.ylabel('Certeza (%)')
    plt.grid(True)
    
    plt.tight_layout()
    nome_grafico = f"resultado_fedavg_rodada_{rodada_atual}.png"
    plt.savefig(nome_grafico)
    plt.close()
    print(f"[SERVER] 📊 Gráfico de desempenho salvo: {nome_grafico}")

@app.route('/baixar_modelo', methods=['GET'])
def enviar_modelo():
    global modelo_global_w2
    if modelo_global_w2 is None:
        return "Nenhum modelo global disponivel ainda", 404
    
    print(f"\n[SERVER] Um cliente (ESP32) está baixando o Modelo Global da Rodada {rodada_atual - 1}...")
    return modelo_global_w2.tobytes(), 200, {'Content-Type': 'application/octet-stream'}

@app.route('/atualizar_pesos', methods=['POST'])
def receber_pesos():
    global pesos_recebidos_buffer, rodada_atual, modelo_global_w2
    global historico_loss, historico_acuracia

    # 1. Lê os pesos binários
    dados_brutos = request.data
    pesos_cliente = np.frombuffer(dados_brutos, dtype=np.float32)

    # 2. Lê as métricas enviadas nos cabeçalhos pelo ESP32
    # Se o ESP32 não mandar, usamos um valor padrão (0.0) para não quebrar
    loss_cliente = float(request.headers.get('X-Loss', 0.0))
    acc_cliente = float(request.headers.get('X-Accuracy', 0.0))

    print(f"\n[SERVER - Rodada {rodada_atual}] Recebendo ClientUpdate...")
    print(f"[SERVER] Métricas do Cliente -> Loss: {loss_cliente:.4f} | Acurácia: {acc_cliente:.2f}%")
    
    # Guardamos os pesos e as métricas para fazer a média depois
    pesos_recebidos_buffer.append({
        'pesos': pesos_cliente,
        'loss': loss_cliente,
        'acc': acc_cliente
    })
    
    clientes_recebidos = len(pesos_recebidos_buffer)
    print(f"[SERVER] Progresso da Agregação: {clientes_recebidos}/{NUM_CLIENTES_ESPERADOS} clientes.")

    # 3. FEDAVG E GERAÇÃO DE GRÁFICOS
    if clientes_recebidos >= NUM_CLIENTES_ESPERADOS:
        print("\n" + "="*60)
        print(f"🚀 INICIANDO FEDERATED AVERAGING (FEDAVG) - RODADA {rodada_atual} 🚀")
        
        # Extrai os dados do buffer
        lista_pesos = [item['pesos'] for item in pesos_recebidos_buffer]
        lista_loss = [item['loss'] for item in pesos_recebidos_buffer]
        lista_acc = [item['acc'] for item in pesos_recebidos_buffer]
        
        # Calcula as médias da rodada
        modelo_global_w2 = np.mean(np.stack(lista_pesos), axis=0)
        loss_medio_rodada = np.mean(lista_loss)
        acc_media_rodada = np.mean(lista_acc)
        
        # Salva no histórico para o gráfico
        historico_loss.append(loss_medio_rodada)
        historico_acuracia.append(acc_media_rodada)
        
        print(f"[SERVER] Média Federada calculada com sucesso!")
        
        # Salva o arquivo .npy
        np.save(f"modelo_global_w2_rodada_{rodada_atual}.npy", modelo_global_w2)
        
        # Desenha o gráfico!
        plotar_graficos()
        
        pesos_recebidos_buffer = [] 
        rodada_atual += 1
        print("="*60 + "\n")

    return "Pesos e métricas agregados com sucesso!", 200

if __name__ == '__main__':
    print("--- Servidor Global FedCycle (Com Gráficos) Iniciado ---")
    app.run(host='0.0.0.0', port=5000)