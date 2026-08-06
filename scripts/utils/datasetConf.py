import numpy as np
from torchvision import datasets, transforms

def preparar_dataset_non_iid(train_dataset, num_clientes, classes_por_cliente, nome_dataset):
    """
    Lógica genérica para dividir um dataset de forma Não-IID.
    """
    if hasattr(train_dataset, 'targets'):
        labels = np.array(train_dataset.targets)
    else:
        labels = np.array(train_dataset.labels)
    
    num_amostras = len(labels)
    num_shards = num_clientes * classes_por_cliente
    tamanho_shard = num_amostras // num_shards 
    
    indices_ordenados = np.argsort(labels)
    shards_disponiveis = np.arange(num_shards)
    np.random.shuffle(shards_disponiveis)
    
    dados_dos_clientes = {i: np.array([], dtype='int64') for i in range(num_clientes)}
    
    print(f"\nDistribuindo dados de {nome_dataset} para {num_clientes} clientes ({classes_por_cliente} classes/shards por cliente)...")
    
    for i in range(num_clientes):
        shards_do_cliente = shards_disponiveis[i * classes_por_cliente : (i + 1) * classes_por_cliente]
        for shard in shards_do_cliente:
            inicio = shard * tamanho_shard
            fim = inicio + tamanho_shard
            indices_fatia = indices_ordenados[inicio:fim]
            dados_dos_clientes[i] = np.concatenate((dados_dos_clientes[i], indices_fatia), axis=0)
        np.random.shuffle(dados_dos_clientes[i])
        
    print(f"Distribuição de {nome_dataset} concluída com sucesso!")
    return train_dataset, dados_dos_clientes

def preparar_cifar10_non_iid(num_clientes=100, classes_por_cliente=2):
    print("Baixando e carregando o dataset CIFAR-10...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return preparar_dataset_non_iid(train_dataset, num_clientes, classes_por_cliente, "CIFAR-10")

def preparar_mnist_non_iid(num_clientes=100, classes_por_cliente=2):
    print("Baixando e carregando o dataset MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return preparar_dataset_non_iid(train_dataset, num_clientes, classes_por_cliente, "MNIST")
    
def exportar_amostra_para_c(dataset, indices_cliente, num_amostras=5, nome_arquivo="dados_cliente0.h", max_pixels=2048):
    print(f"\nExportando {num_amostras} amostras para o simulador Wokwi (C++)...")
    indices_amostra = indices_cliente[:num_amostras]
    
    with open(nome_arquivo, 'w') as f:
        f.write("#ifndef DADOS_CLIENTE_H\n")
        f.write("#define DADOS_CLIENTE_H\n\n")
        f.write(f"// Total de imagens embutidas na memoria flash: {num_amostras}\n")
        f.write(f"const int num_amostras_locais = {num_amostras};\n\n")
        
        # O tamanho do array deve ser consistente com INPUT_DIM no firmware
        f.write(f"const unsigned char dados_pixels[{num_amostras}][{max_pixels}] = {{\n")
        
        labels_amostra = []
        for idx in indices_amostra:
            imagem, label = dataset[idx] 
            labels_amostra.append(label)
            pixels_numpy = (imagem.numpy() * 255).astype(int).flatten()
            
            # Ajuste de tamanho para bater com max_pixels do firmware
            if len(pixels_numpy) > max_pixels:
                pixels_numpy = pixels_numpy[:max_pixels]
            elif len(pixels_numpy) < max_pixels:
                pixels_numpy = np.pad(pixels_numpy, (0, max_pixels - len(pixels_numpy)), 'constant')
            
            pixels_str = ", ".join(map(str, pixels_numpy))
            f.write(f"    {{{pixels_str}}},\n")
            
        f.write("};\n\n")
        labels_str = ", ".join(map(str, labels_amostra))
        f.write(f"const int labels_locais[{num_amostras}] = {{{labels_str}}};\n\n")
        f.write("#endif // DADOS_CLIENTE_H\n")
        
    print(f"Sucesso! Arquivo '{nome_arquivo}' gerado.")


# --- Cole isso lá no final do seu bloco __main__ ---
# exportar_amostra_para_c(dataset, dict_clientes[0], num_amostras=5)

# --- Testando a nossa função ---
if __name__ == "__main__":
    # Seguindo o padrão de 100 clientes com S=2 do FedBKD
    dataset, dict_clientes = preparar_cifar10_non_iid(num_clientes=100, classes_por_cliente=2)
    
    # Verificando a "Realidade de Paulo Freire" de um cliente aleatório (ex: Cliente 0)
    cliente_teste = 0
    indices_cliente_0 = dict_clientes[cliente_teste]
    labels_cliente_0 = np.array(dataset.targets)[indices_cliente_0]
    
    classes_unicas = np.unique(labels_cliente_0)
    
    print(f"\n--- Resumo do Cliente {cliente_teste} ---")
    print(f"Total de imagens que ele possui: {len(indices_cliente_0)}")
    print(f"Classes únicas que ele conhece (IDs): {classes_unicas}")
    
    exportar_amostra_para_c(dataset, indices_cliente_0, num_amostras=5, nome_arquivo="dados_cliente0.h")