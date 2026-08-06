import numpy as np
import os

def main():
    # Caminho do modelo global treinado na rodada 160
    caminho_modelo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'models', 'modelo_global_w2_rodada_160.npy'))
    
    # Caminho de saída dos pesos de classificação
    caminho_saida = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'firmware', 'include', 'pesos_classificacao.h'))
    
    if not os.path.exists(caminho_modelo):
        print(f"[ERRO] O arquivo do modelo não foi encontrado em: {caminho_modelo}")
        return
        
    print(f"Carregando os pesos do modelo a partir de: {caminho_modelo}")
    w2 = np.load(caminho_modelo)
    
    # Certifica-se de que possui o tamanho esperado (128 * 10 = 1280)
    w2 = w2.flatten()
    if len(w2) != 1280:
        print(f"[AVISO] O tamanho de W2 é {len(w2)} em vez de 1280. Verifique as dimensões do modelo.")
        
    print(f"Exportando {len(w2)} pesos W2 e inicializando b2 zerado para {caminho_saida}...")
    
    with open(caminho_saida, 'w') as f:
        f.write("#ifndef PESOS_CLASSIFICACAO_H\n")
        f.write("#define PESOS_CLASSIFICACAO_H\n\n")
        
        f.write("// --- PESOS DA CAMADA DE SAÍDA AGREGADOS (128 -> 10) ---\n")
        f.write(f"const float W2[{len(w2)}] = {{\n    ")
        # Escreve os pesos com formatação float de 6 casas decimais
        f.write(",\n    ".join(", ".join(f"{val:.6f}" for val in w2[i:i+8]) for i in range(0, len(w2), 8)))
        f.write("\n};\n\n")
        
        # Inicializa o bias b2 zerado (10 classes) para o fine-tuning on-device começar limpo
        b2 = [0.0] * 10
        f.write(f"const float b2[{len(b2)}] = {{\n    ")
        f.write(", ".join(f"{val:.6f}" for val in b2))
        f.write("\n};\n\n")
        
        f.write("#endif // PESOS_CLASSIFICACAO_H\n")
        
    print(f"✅ Sucesso! O arquivo pesos_classificacao.h foi atualizado e reduzido para ~10KB.")

if __name__ == "__main__":
    main()
