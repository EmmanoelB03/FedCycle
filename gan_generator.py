import torch
import torch.nn as nn
import torch.nn.functional as F

# Importando o modelo que criamos no passo anterior
from fedcycle_model import ModeloFedCycle

class GeradorDataFree(nn.Module):
    def __init__(self, dimensao_ruido=100, dimensao_saida=2048):
        super(GeradorDataFree, self).__init__()
        # O Gerador recebe um ruído aleatório (ex: 100 números) e expande
        # até chegar no tamanho do mapa de características (2048)
        self.rede = nn.Sequential(
            nn.Linear(dimensao_ruido, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, dimensao_saida),
            # Usamos ReLU no final porque os feature maps originais 
            # (que saem da representação) também passam por um ReLU
            nn.ReLU() 
        )

    def forward(self, ruido):
        return self.rede(ruido)

# --- SISTEMA DE JULGAMENTO (Funções de Perda) ---

def perda_confianca(saida_discriminador):
    """
    Equação L_oh do artigo FedBKD.
    Recompensa o Gerador se o Cliente (Discriminador) tiver certeza do que está vendo.
    """
    # Descobre qual classe o Discriminador achou mais provável (o pseudo-gabarito)
    pseudo_labels = torch.argmax(saida_discriminador, dim=1)
    
    # Calcula a entropia cruzada entre a saída e o pseudo-gabarito
    # Quanto mais confiante o modelo for, menor será essa perda (melhor para o Gerador)
    perda = F.cross_entropy(saida_discriminador, pseudo_labels)
    return perda

def perda_diversidade(ruido1, ruido2, gerado1, gerado2):
    """
    Equação L_ms do artigo FedBKD.
    Pune o Gerador se ele tentar "trapacear" gerando sempre o mesmo dado.
    """
    # Calcula a distância entre os ruídos de entrada
    distancia_ruido = torch.mean(torch.abs(ruido1 - ruido2), dim=1)
    
    # Calcula a distância entre os dados sintéticos gerados
    distancia_gerados = torch.mean(torch.abs(gerado1 - gerado2), dim=1)
    
    # Queremos maximizar a relação (distancia_gerados / distancia_ruido)
    # Como o PyTorch sempre *minimiza* a perda, nós retornamos o valor negativo
    perda_ms = -torch.mean(distancia_gerados / (distancia_ruido + 1e-5))
    return perda_ms

# --- Testando a nossa GAN ---
if __name__ == "__main__":
    batch_size = 16
    dim_ruido = 100
    
    # 1. Instanciando os Atores
    gerador = GeradorDataFree(dimensao_ruido=dim_ruido, dimensao_saida=2048)
    
    # O Discriminador é o modelo local (a camada de classificação)
    modelo_cliente = ModeloFedCycle()
    discriminador = modelo_cliente.classificacao
    
    # Congelando o Discriminador (ele é só o juiz agora, não aprende nada)
    discriminador.eval() 
    for param in discriminador.parameters():
        param.requires_grad = False
        
    # 2. O Gerador cria os dados sintéticos a partir do nada (ruído R)
    ruido_a = torch.randn(batch_size, dim_ruido)
    ruido_b = torch.randn(batch_size, dim_ruido)
    
    x_gerado_a = gerador(ruido_a)
    x_gerado_b = gerador(ruido_b)
    
    print(f"Formato do dado sintético gerado: {x_gerado_a.shape} (Esperado: 16 lotes de 2048 atributos)")
    
    # 3. O Cliente julga os dados gerados
    palpites_cliente = discriminador(x_gerado_a)
    
    # 4. Calculando as Punições/Recompensas
    l_oh = perda_confianca(palpites_cliente)
    l_ms = perda_diversidade(ruido_a, ruido_b, x_gerado_a, x_gerado_b)
    
    peso_diversidade = 1.0 # O lambda da equação
    perda_total_gerador = l_oh + (peso_diversidade * l_ms)
    
    print(f"Perda de Confiança (L_oh): {l_oh.item():.4f}")
    print(f"Perda de Diversidade (L_ms): {l_ms.item():.4f}")
    print(f"Perda Total do Gerador: {perda_total_gerador.item():.4f}")