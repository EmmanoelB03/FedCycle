import torch
import numpy as np
from fedcycle_model import ModeloFedCycle

def exportar_pesos_para_cpp():
    print("Carregando o modelo destilado...")
    # Instanciamos o modelo. Em um cenário real de execução, você carregaria 
    # os pesos que acabaram de sair da Destilação Bidirecional aqui.
    modelo_cliente = ModeloFedCycle()
    
    # Vamos acessar especificamente a camada de classificação:
    # classificacao[0] = Camada Linear (2048 -> 128)
    # classificacao[1] = ReLU
    # classificacao[2] = Camada Linear (128 -> 10)
    
    camada1 = modelo_cliente.classificacao[0]
    camada2 = modelo_cliente.classificacao[2]
    
    nome_arquivo = "pesos_classificacao.h"
    print(f"\nExtraindo as matrizes matemáticas para '{nome_arquivo}'...")
    
    with open(nome_arquivo, 'w') as f:
        f.write("#ifndef PESOS_CLASSIFICACAO_H\n")
        f.write("#define PESOS_CLASSIFICACAO_H\n\n")
        
        f.write("// --- PESOS DA CAMADA OCULTA (2048 -> 128) ---\n")
        # Achata a matriz de pesos (128x2048) em uma linha reta
        w1 = camada1.weight.detach().numpy().flatten()
        b1 = camada1.bias.detach().numpy().flatten()
        
        f.write(f"const float W1[{len(w1)}] = {{\n    ")
        f.write(", ".join(f"{val:.6f}" for val in w1))
        f.write("\n};\n\n")
        
        f.write(f"const float b1[{len(b1)}] = {{\n    ")
        f.write(", ".join(f"{val:.6f}" for val in b1))
        f.write("\n};\n\n")
        
        f.write("// --- PESOS DA CAMADA DE SAÍDA (128 -> 10) ---\n")
        w2 = camada2.weight.detach().numpy().flatten()
        b2 = camada2.bias.detach().numpy().flatten()
        
        f.write(f"const float W2[{len(w2)}] = {{\n    ")
        f.write(", ".join(f"{val:.6f}" for val in w2))
        f.write("\n};\n\n")
        
        f.write(f"const float b2[{len(b2)}] = {{\n    ")
        f.write(", ".join(f"{val:.6f}" for val in b2))
        f.write("\n};\n\n")
        
        f.write("#endif // PESOS_CLASSIFICACAO_H\n")
        
    print(f"\nSucesso! Foram extraídos {len(w1) + len(b1) + len(w2) + len(b2)} parâmetros treináveis.")
    print("O cérebro do aluno está pronto para ser injetado no Wokwi!")

if __name__ == "__main__":
    exportar_pesos_para_cpp()