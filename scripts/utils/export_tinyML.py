import torch
import torch.nn as nn
import os
import sys

# Adiciona a pasta de treinamento ao path para encontrar o modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))
from fedcycle_model import ModeloFedCycle

def exportar_para_onnx():
    print("Carregando o modelo local destilado...")
    modelo_cliente = ModeloFedCycle()
    modelo_cliente.eval() # Modo de inferência para exportação
    
    # Caminho atualizado para a pasta de modelos
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'models'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Exportando a Camada de Representação (O Extrator Fixo)
    # Entrada: 1 imagem RGB 32x32
    entrada_representacao = torch.randn(1, 3, 32, 32)
    caminho_rep = os.path.join(output_dir, "representacao_congelada.onnx")
    
    print(f"\nExportando a Camada de Representação para {caminho_rep}...")
    torch.onnx.export(
        modelo_cliente.representacao, 
        entrada_representacao, 
        caminho_rep,
        export_params=True,
        input_names=['entrada_imagem'],
        output_names=['saida_features']
    )
    
    # 2. Exportando a Camada de Classificação (A que será treinada no ESP32)
    # Entrada: O feature map de 2048 valores
    entrada_classificacao = torch.randn(1, 2048)
    caminho_class = os.path.join(output_dir, "classificacao_treinavel.onnx")
    
    print(f"Exportando a Camada de Classificação para {caminho_class}...")
    torch.onnx.export(
        modelo_cliente.classificacao, 
        entrada_classificacao, 
        caminho_class,
        export_params=True,
        input_names=['entrada_features'],
        output_names=['saida_probabilidades']
    )
    
    print("\nExportação ONNX concluída com sucesso! Modelos prontos para a pipeline do TinyML.")

if __name__ == "__main__":
    exportar_para_onnx()
