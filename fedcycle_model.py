import torch
import torch.nn as nn

class ModeloFedCycle(nn.Module):
    def __init__(self, num_classes=10):
        super(ModeloFedCycle, self).__init__()
        
        # 1. CAMADA DE REPRESENTAÇÃO (Theta R) - O Extrator de Características
        # Esta parte ficará "congelada" no microcontrolador durante o treinamento local.
        self.representacao = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten() # Achata os mapas de características para entregar à classificação
        )
        
        # 2. CAMADA DE CLASSIFICAÇÃO (Theta C) - O Decisor
        # É esta parte leve que o ESP32 vai treinar com os dados dele.
        # Imagens 32x32 passam por dois MaxPools (dividindo tamanho por 2 e depois por 2) -> 8x8.
        # Então temos 32 canais * 8 * 8 = 2048 características extraídas.
        self.classificacao = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # A informação flui da representação para a classificação
        features = self.representacao(x)
        saida = self.classificacao(features)
        return saida

# --- Testando a Arquitetura ---
if __name__ == "__main__":
    modelo_global = ModeloFedCycle()
    
    # Criando uma "imagem" falsa só para testar se os tamanhos das matrizes batem
    # (1 batch, 3 canais RGB, 32 altura, 32 largura)
    imagem_teste = torch.randn(1, 3, 32, 32)
    
    saida_teste = modelo_global(imagem_teste)
    
    print("Arquitetura carregada com sucesso!")
    print(f"Formato da saída final (deve ser 1 matriz com 10 probabilidades): {saida_teste.shape}")