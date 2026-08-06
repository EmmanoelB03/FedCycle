from flask import Flask, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# --- Configurações do Aprendizado Federado ---
NUM_CLIENTES_ESPERADOS = 50   
pesos_recebidos_buffer = []  
rodada_atual = 1             
modelo_global_w2 = None 

# --- Histórico para os Gráficos ---
historico_loss = []
historico_acuracia = []
historico_certeza = []
historico_tempo = []
historico_memoria = []

def plotar_graficos():
    """Gera e salva o gráfico de evolução do modelo global com as novas métricas"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Painel de Controle e Métricas FedCycle - Rodada {rodada_atual}', fontsize=16, fontweight='bold', color='#2C3E50', y=0.98)
    
    x_rodadas = range(1, len(historico_loss) + 1)
    
    # 1. Gráfico de Perda (Loss)
    axs[0, 0].plot(x_rodadas, historico_loss, marker='o', color='#E74C3C', linewidth=2, label='Loss Médio')
    axs[0, 0].set_title('Evolução da Perda (Loss)', fontsize=12, fontweight='bold', color='#2C3E50')
    axs[0, 0].set_xlabel('Rodadas de Comunicação')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()
    
    # 2. Gráfico de Acurácia vs Certeza
    axs[0, 1].plot(x_rodadas, historico_acuracia, marker='s', color='#3498DB', linewidth=2, label='Acurácia Real (%)')
    axs[0, 1].plot(x_rodadas, historico_certeza, marker='^', color='#2ECC71', linewidth=1.5, linestyle='--', label='Certeza Média (%)')
    axs[0, 1].set_title('Acurácia Real vs Certeza Média', fontsize=12, fontweight='bold', color='#2C3E50')
    axs[0, 1].set_xlabel('Rodadas de Comunicação')
    axs[0, 1].set_ylabel('Porcentagem (%)')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()
    
    # 3. Gráfico de Tempo de Treinamento
    tempos_segundos = [t / 1000.0 for t in historico_tempo]
    axs[1, 0].plot(x_rodadas, tempos_segundos, marker='D', color='#8E44AD', linewidth=2, label='Tempo Médio (s)')
    axs[1, 0].set_title('Tempo Médio de Treinamento On-Device', fontsize=12, fontweight='bold', color='#2C3E50')
    axs[1, 0].set_xlabel('Rodadas de Comunicação')
    axs[1, 0].set_ylabel('Tempo (segundos)')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()
    
    # 4. Gráfico de Memória Heap Livre
    memoria_kb = [m / 1024.0 for m in historico_memoria]
    axs[1, 1].plot(x_rodadas, memoria_kb, marker='x', color='#16A085', linewidth=2, label='Heap Livre (KB)')
    axs[1, 1].set_title('Memória Heap Livre Média no Cliente', fontsize=12, fontweight='bold', color='#2C3E50')
    axs[1, 1].set_xlabel('Rodadas de Comunicação')
    axs[1, 1].set_ylabel('Memória (KB)')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'results'))
    if not os.path.exists(output_dir_results):
        os.makedirs(output_dir_results)
        
    nome_grafico = os.path.join(output_dir_results, f"resultado_fedavg_rodada_{rodada_atual}.png")
    plt.savefig(nome_grafico, dpi=150)
    plt.close()
    print(f"[SERVER] 📊 Gráfico avançado de desempenho salvo: {nome_grafico}")

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
    global historico_loss, historico_acuracia, historico_certeza, historico_tempo, historico_memoria

    # 1. Lê os pesos binários
    dados_brutos = request.data
    pesos_cliente = np.frombuffer(dados_brutos, dtype=np.float32)

    # 2. Lê as métricas enviadas nos cabeçalhos pelo ESP32 / simulador
    loss_cliente = float(request.headers.get('X-Loss', 0.0))
    acc_cliente = float(request.headers.get('X-Accuracy', 0.0))
    conf_cliente = float(request.headers.get('X-Confidence', acc_cliente)) # fallback para acc_cliente se não enviado
    time_cliente = float(request.headers.get('X-Training-Time', 0.0))
    mem_cliente = float(request.headers.get('X-Memory-Usage', 0.0))

    print(f"\n[SERVER - Rodada {rodada_atual}] Recebendo ClientUpdate...")
    print(f"[SERVER] Métricas do Cliente -> Loss: {loss_cliente:.4f} | Acurácia: {acc_cliente:.2f}% | Certeza: {conf_cliente:.2f}% | Tempo: {time_cliente/1000.0:.2f}s | Mem. Livre: {mem_cliente/1024.0:.2f}KB")
    
    # Guardamos os pesos e as métricas para fazer a média depois
    pesos_recebidos_buffer.append({
        'pesos': pesos_cliente,
        'loss': loss_cliente,
        'acc': acc_cliente,
        'conf': conf_cliente,
        'time': time_cliente,
        'mem': mem_cliente
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
        lista_conf = [item['conf'] for item in pesos_recebidos_buffer]
        lista_time = [item['time'] for item in pesos_recebidos_buffer]
        lista_mem = [item['mem'] for item in pesos_recebidos_buffer]
        
        # Calcula as médias da rodada
        modelo_global_w2 = np.mean(np.stack(lista_pesos), axis=0)
        loss_medio_rodada = np.mean(lista_loss)
        acc_media_rodada = np.mean(lista_acc)
        conf_media_rodada = np.mean(lista_conf)
        time_medio_rodada = np.mean(lista_time)
        mem_media_rodada = np.mean(lista_mem)
        
        # Salva no histórico para o gráfico
        historico_loss.append(loss_medio_rodada)
        historico_acuracia.append(acc_media_rodada)
        historico_certeza.append(conf_media_rodada)
        historico_tempo.append(time_medio_rodada)
        historico_memoria.append(mem_media_rodada)
        
        print(f"[SERVER] Média Federada calculada com sucesso!")
        print(f"[SERVER] Métricas Médias da Rodada {rodada_atual}:")
        print(f"  -> Loss Médio: {loss_medio_rodada:.4f}")
        print(f"  -> Acurácia Real Média: {acc_media_rodada:.2f}%")
        print(f"  -> Certeza Média: {conf_media_rodada:.2f}%")
        print(f"  -> Tempo Médio de Treinamento: {time_medio_rodada/1000.0:.2f} s")
        print(f"  -> Memória Heap Livre Média: {mem_media_rodada/1024.0:.2f} KB")
        
        # Caminho atualizado para a pasta de modelos
        output_dir_models = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'models'))
        if not os.path.exists(output_dir_models):
            os.makedirs(output_dir_models)
            
        # Salva o arquivo .npy
        caminho_modelo = os.path.join(output_dir_models, f"modelo_global_w2_rodada_{rodada_atual}.npy")
        np.save(caminho_modelo, modelo_global_w2)
        
        # Desenha o gráfico!
        plotar_graficos()
        
        pesos_recebidos_buffer = [] 
        rodada_atual += 1
        print("="*60 + "\n")

    return "Pesos e métricas agregados com sucesso!", 200

if __name__ == '__main__':
    print("--- Servidor Global FedCycle (Com Painel Multi-Métricas) Iniciado ---")
    app.run(host='0.0.0.0', port=5000)
