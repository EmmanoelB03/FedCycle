import numpy as np
import torch
from torchvision import datasets, transforms
import os
import sys

# Parâmetros da arquitetura (devem bater com o simulador)
INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def main():
    diretorio_modelos = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'models'))
    
    # Encontra todos os arquivos de modelo global salvos nas rodadas e pega o mais recentemente gravado
    arquivos = [os.path.join(diretorio_modelos, f) for f in os.listdir(diretorio_modelos) if f.startswith("modelo_global_w2_rodada_") and f.endswith(".npy")]
    
    if not arquivos:
        print(f"[ERRO] Nenhum arquivo modelo_global_w2_rodada_*.npy encontrado em {diretorio_modelos}")
        return
        
    caminho_modelo = max(arquivos, key=os.path.getmtime)
    nome_arquivo = os.path.basename(caminho_modelo)
    rodada_recente = int(nome_arquivo.replace("modelo_global_w2_rodada_", "").replace(".npy", ""))
        
    print(f"Carregando pesos W2 da Rodada {rodada_recente} (arquivo mais recente) de: {caminho_modelo}")
    W2 = np.load(caminho_modelo).reshape((OUTPUT_DIM, HIDDEN_DIM))
    
    # Reconstrói a W1 idêntica usando a mesma semente aleatória
    np.random.seed(42)
    W1_GLOBAL = np.random.randn(HIDDEN_DIM, INPUT_DIM) * 0.05
    
    # Carrega o conjunto de TESTE do MNIST (dados totalmente inéditos com as 10 classes)
    print("\nCarregando dataset de teste do MNIST (10 classes)...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    hits = 0
    total = len(test_dataset)
    
    # Para métricas por classe
    hits_por_classe = {i: 0 for i in range(10)}
    total_por_classe = {i: 0 for i in range(10)}
    
    print(f"Avaliando o modelo global em {total} amostras de teste...")
    
    for img, label in test_dataset:
        x = img.numpy().flatten()[:INPUT_DIM]
        
        # Forward pass (idêntico ao cliente)
        A1 = np.maximum(0, np.dot(W1_GLOBAL, x))  # ReLU
        logits = np.dot(W2, A1)
        probs = softmax(logits)
        
        pred = np.argmax(probs)
        
        total_por_classe[label] += 1
        if pred == label:
            hits += 1
            hits_por_classe[label] += 1
            
    acuracia_global = (hits / total) * 100.0
    
    print("\n==============================================")
    print("📊 RESULTADO DA AVALIAÇÃO DO MODELO GLOBAL 📊")
    print("==============================================")
    print(f"Acurácia Global (10 classes): {acuracia_global:.2f}%")
    print("----------------------------------------------")
    print("Acurácia detalhada por classe:")
    for i in range(10):
        acc_classe = (hits_por_classe[i] / total_por_classe[i]) * 100.0 if total_por_classe[i] > 0 else 0
        print(f"  -> Classe {i}: {acc_classe:.2f}% ({hits_por_classe[i]}/{total_por_classe[i]})")
    print("==============================================\n")

if __name__ == "__main__":
    main()
