import numpy as np
from torchvision import datasets, transforms

def preparar_cifar10_non_iid(num_clientes=100, classes_por_cliente=2):
    """
    Baixa o CIFAR-10 e divide os dados de forma Não-IID.
    Cada cliente receberá dados de exatamente 'classes_por_cliente' categorias.
    """
    print("Baixando e carregando o dataset CIFAR-10...")
    # Transformação básica (converte imagem para Tensor)
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Baixa o dataset de treinamento (50.000 imagens)
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Extrai as labels (gabaritos) de todas as 50.000 imagens
    labels = np.array(train_dataset.targets)
    
    # --- INÍCIO DA LÓGICA DE PARTIÇÃO (Método de Shards) ---
    
    num_amostras = len(labels)
    # Calculamos o total de 'fatias' necessárias. 
    # Ex: 100 clientes * 2 classes = 200 fatias.
    num_shards = num_clientes * classes_por_cliente
    tamanho_shard = num_amostras // num_shards # 50.000 / 200 = 250 imagens por fatia
    
    # Ordenamos os dados pela label. Assim, ficam todos os aviões juntos, depois carros, etc.
    indices_ordenados = np.argsort(labels)
    
    # Criamos uma lista com os IDs dos shards (0 a 199) e os embaralhamos
    shards_disponiveis = np.arange(num_shards)
    np.random.shuffle(shards_disponiveis)
    
    # Dicionário para guardar quais índices pertencem a qual cliente
    dados_dos_clientes = {i: np.array([], dtype='int64') for i in range(num_clientes)}
    
    print(f"\nDistribuindo dados para {num_clientes} clientes ({classes_por_cliente} classes/shards por cliente)...")
    
    # Entregamos as fatias para os clientes
    for i in range(num_clientes):
        # Pega 2 shards da lista embaralhada
        shards_do_cliente = shards_disponiveis[i * classes_por_cliente : (i + 1) * classes_por_cliente]
        
        for shard in shards_do_cliente:
            # Calcula o início e o fim dos índices reais do dataset para esse shard
            inicio = shard * tamanho_shard
            fim = inicio + tamanho_shard
            
            # Adiciona os índices das imagens ao 'balde' do cliente
            indices_fatia = indices_ordenados[inicio:fim]
            dados_dos_clientes[i] = np.concatenate((dados_dos_clientes[i], indices_fatia), axis=0)
            
        # Embaralha os dados dentro do cliente para não ficar tudo de uma classe, depois tudo de outra
        np.random.shuffle(dados_dos_clientes[i])
        
    print("Distribuição concluída com sucesso!")
    return train_dataset, dados_dos_clientes

    
def exportar_amostra_para_c(dataset, indices_cliente, num_amostras=5, nome_arquivo="dados_cliente0.h"):
    print(f"\nExportando {num_amostras} amostras para o simulador Wokwi (C++)...")
    
    # Seleciona apenas as primeiras N imagens do cliente
    indices_amostra = indices_cliente[:num_amostras]
    
    with open(nome_arquivo, 'w') as f:
        f.write("#ifndef DADOS_CLIENTE_H\n")
        f.write("#define DADOS_CLIENTE_H\n\n")
        
        f.write(f"// Total de imagens embutidas na memoria flash: {num_amostras}\n")
        f.write(f"const int num_amostras_locais = {num_amostras};\n\n")
        
        # O CIFAR-10 tem imagens de 3 canais (RGB) de 32x32 pixels = 3072 valores por imagem
        f.write(f"const unsigned char dados_pixels[{num_amostras}][3072] = {{\n")
        
        labels_amostra = []
        
        for idx in indices_amostra:
            # Pega a imagem (Tensor) e o gabarito (label)
            imagem, label = dataset[idx] 
            labels_amostra.append(label)
            
            # Converte de Tensor (0.0 a 1.0) para Pixels Inteiros (0 a 255)
            # O .flatten() transforma a matriz 3x32x32 em uma linha reta de 3072 números
            pixels_numpy = (imagem.numpy() * 255).astype(int).flatten()
            
            # Formata os números separados por vírgula para a sintaxe do C++
            pixels_str = ", ".join(map(str, pixels_numpy))
            f.write(f"    {{{pixels_str}}},\n")
            
        f.write("};\n\n")
        
        # Salva também os gabaritos (as respostas corretas)
        labels_str = ", ".join(map(str, labels_amostra))
        f.write(f"const int labels_locais[{num_amostras}] = {{{labels_str}}};\n\n")
        
        f.write("#endif // DADOS_CLIENTE_H\n")
        
    print(f"Sucesso! Arquivo '{nome_arquivo}' gerado na sua pasta.")

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