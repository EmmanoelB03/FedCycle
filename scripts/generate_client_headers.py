import os
import sys

# Adiciona o diretório scripts ao path para importar as utilidades
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils.datasetConf import preparar_mnist_non_iid, exportar_amostra_para_c

def main():
    num_clientes = 50
    # Define o diretório de saída dentro de firmware/include
    output_dir = os.path.abspath(os.path.join(current_dir, '..', 'firmware', 'include', 'clientes'))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Diretório criado: {output_dir}")

    # Prepara os dados Non-IID para 50 clientes usando MNIST
    dataset, dict_clientes = preparar_mnist_non_iid(num_clientes=num_clientes, classes_por_cliente=2)
    
    print(f"\n[INFO] Gerando {num_clientes} arquivos de cabeçalho em {output_dir}...")
    
    for i in range(num_clientes):
        nome_arquivo = os.path.join(output_dir, f"dados_cliente{i}.h")
        # Exporta 5 amostras reais do MNIST para cada cliente (784 pixels)
        exportar_amostra_para_c(dataset, dict_clientes[i], num_amostras=5, nome_arquivo=nome_arquivo, max_pixels=784)
        if (i + 1) % 10 == 0:
            print(f"[PROGRESSO] {i + 1}/{num_clientes} arquivos gerados.")

    
    print("\n✅ Sucesso! Todos os 50 arquivos foram gerados.")
    print(f"📍 Localização: firmware/include/clientes/")

if __name__ == "__main__":
    main()
