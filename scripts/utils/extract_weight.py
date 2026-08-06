import torch
import numpy as np
import os
import sys

# Adiciona a pasta de treinamento ao path para encontrar o modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))
from fedcycle_model import ModeloFedCycle

def exportar_pesos_para_cpp():
    print("Carregando o modelo destilado...")
    modelo_cliente = ModeloFedCycle()
    
    camada1 = modelo_cliente.classificacao[0]
    camada2 = modelo_cliente.classificacao[2]
    
    # Caminho atualizado para a pasta do firmware
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'firmware', 'include'))
    nome_arquivo = os.path.join(output_dir, "pesos_classificacao.h")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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