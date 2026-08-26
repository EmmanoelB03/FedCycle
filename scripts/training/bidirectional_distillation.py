import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Importando o que já construímos
from fedcycle_model import ModeloFedCycle

def divergencia_kl_features(features_aluno, features_professor, temperatura=2.0):
    """
    Calcula a Divergência Kullback-Leibler (KL) entre os mapas de características.
    A temperatura 'amolece' as distribuições para facilitar a transferência de conhecimento.
    """
    # O PyTorch exige que o aluno esteja em formato Log-Softmax e o professor em Softmax
    log_probs_aluno = F.log_softmax(features_aluno / temperatura, dim=1)
    probs_professor = F.softmax(features_professor / temperatura, dim=1)
    
    # KLDivLoss do PyTorch
    perda_kl = F.kl_div(log_probs_aluno, probs_professor, reduction='batchmean')
    
    # Multiplicamos pelo quadrado da temperatura para manter a escala dos gradientes
    return perda_kl * (temperatura ** 2)

def destilacao_global_para_local(modelo_global, modelo_local, dados_sinteticos, epocas=4, lr=0.01):
    """
    Fase 1: O Professor (Global) ensina o Aluno (Local).
    """
    print("\n--- Iniciando Destilação Global -> Local ---")
    
    # Congelamos a camada de classificação do modelo local 
    for param in modelo_local.classificacao.parameters():
        param.requires_grad = False
        
    # Otimizador apenas para a camada de REPRESENTAÇÃO do modelo local
    otimizador = optim.SGD(modelo_local.representacao.parameters(), lr=lr)
    
    # O artigo FedBKD sugere 4 épocas para essa via 
    for epoca in range(epocas):
        otimizador.zero_grad()
        
        # O professor extrai as características (sem calcular gradientes)
        with torch.no_grad():
            features_globais = modelo_global.representacao(dados_sinteticos)
            
        # O aluno tenta extrair as características
        features_locais = modelo_local.representacao(dados_sinteticos)
        
        # Equação 14 do FedBKD: Perda = KL[Theta_R_local, Theta_R_global] 
        perda_i_g = divergencia_kl_features(features_locais, features_globais)
        
        perda_i_g.backward()
        otimizador.step() # Atualiza o aluno (Equação 15) 
        
    print(f"Destilação G->L concluída. Perda final: {perda_i_g.item():.4f}")
    
    # Descongelamos a classificação para quando ele for treinar no ESP32
    for param in modelo_local.classificacao.parameters():
        param.requires_grad = True

def destilacao_local_para_global(modelo_local, modelo_global, dados_sinteticos, epocas=1, lr=0.01):
    """
    Fase 2: O Aluno (Local) ensina o Professor (Global).
    """
    print("\n--- Iniciando Destilação Local -> Global ---")
    
    # Congelamos a camada de classificação do modelo global para manter generalização 
    for param in modelo_global.classificacao.parameters():
        param.requires_grad = False
        
    # Otimizador apenas para a camada de REPRESENTAÇÃO do modelo global
    otimizador = optim.SGD(modelo_global.representacao.parameters(), lr=lr)
    
    # O artigo FedBKD sugere 1 época para essa via 
    for epoca in range(epocas):
        otimizador.zero_grad()
        
        # O aluno (agora como professor) extrai as características empíricas
        with torch.no_grad():
            features_locais = modelo_local.representacao(dados_sinteticos)
            
        # O professor global (agora como aluno) tenta imitar
        features_globais = modelo_global.representacao(dados_sinteticos)
        
        # Equação 17 do FedBKD: Perda = KL[Theta_R_global, Theta_R_local] 
        perda_g_i = divergencia_kl_features(features_globais, features_locais)
        
        perda_g_i.backward()
        otimizador.step() # Atualiza o global (Equação 18) 
        
    print(f"Destilação L->G concluída. Perda final: {perda_g_i.item():.4f}")

# --- Testando o Ciclo ---
if __name__ == "__main__":
    modelo_servidor = ModeloFedCycle()
    modelo_cliente = ModeloFedCycle()
    
    # Simulando 16 dados sintéticos gerados pela nossa GAN (lote de 3 canais, 32x32 pixels)
    # Na prática real, a GAN gera os features, mas aqui vamos simular a imagem passando pela rede
    imagens_sinteticas = torch.randn(16, 3, 32, 32)
    
    destilacao_global_para_local(modelo_servidor, modelo_cliente, imagens_sinteticas, epocas=4)
    destilacao_local_para_global(modelo_cliente, modelo_servidor, imagens_sinteticas, epocas=1)
    
    print("\nCiclo bidirecional finalizado com sucesso!")