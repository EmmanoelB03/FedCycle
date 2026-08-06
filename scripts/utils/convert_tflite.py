import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys

# Adiciona a pasta de treinamento ao path para encontrar o modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))
from fedcycle_model import ModeloFedCycle

def main():
    print("[CONVERTER] Carregando o modelo PyTorch...")
    modelo_pt = ModeloFedCycle()
    modelo_pt.eval()

    print("[CONVERTER] Criando modelo equivalente em Keras...")
    # Criamos o modelo equivalente em Keras
    keras_model = models.Sequential([
        layers.Input(shape=(32, 32, 3), name='entrada_imagem'),
        layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', name='conv2d_1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'),
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', name='conv2d_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'),
        layers.Permute((3, 1, 2), name='permute_chw'),
        layers.Flatten(name='flatten')
    ])

    print("[CONVERTER] Transferindo pesos de PyTorch para Keras...")
    # Transferência de pesos e biases
    # Camada Conv 1 (PyTorch representacao[0])
    pt_conv1_w = modelo_pt.representacao[0].weight.detach().numpy()
    pt_conv1_b = modelo_pt.representacao[0].bias.detach().numpy()
    # Transpor pesos Conv2d: (out, in, h, w) -> (h, w, in, out)
    tf_conv1_w = np.transpose(pt_conv1_w, (2, 3, 1, 0))
    keras_model.get_layer('conv2d_1').set_weights([tf_conv1_w, pt_conv1_b])

    # Camada Conv 2 (PyTorch representacao[3])
    pt_conv2_w = modelo_pt.representacao[3].weight.detach().numpy()
    pt_conv2_b = modelo_pt.representacao[3].bias.detach().numpy()
    # Transpor pesos Conv2d: (out, in, h, w) -> (h, w, in, out)
    tf_conv2_w = np.transpose(pt_conv2_w, (2, 3, 1, 0))
    keras_model.get_layer('conv2d_2').set_weights([tf_conv2_w, pt_conv2_b])

    print("[CONVERTER] Verificando compatibilidade de inferência...")
    # Testar com uma entrada fictícia se as saídas coincidem
    x_test_pt = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out_pt = modelo_pt.representacao(x_test_pt).numpy()

    # Converter entrada para Keras (batch, height, width, channels)
    x_test_tf = np.transpose(x_test_pt.numpy(), (0, 2, 3, 1))
    out_tf = keras_model.predict(x_test_tf)

    print(f"  Shape Saída PyTorch: {out_pt.shape}")
    print(f"  Shape Saída Keras: {out_tf.shape}")
    diff = np.max(np.abs(out_pt - out_tf))
    print(f"  Diferença máxima entre PyTorch e Keras: {diff:.6e}")
    if diff > 1e-4:
        print("[WARNING] Diferença significativa detectada!")

    print("[CONVERTER] Preparando gerador de dataset representativo para quantização...")
    # Carrega CIFAR-10 para ter dados representativos reais
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    def representative_data_gen():
        for i in range(100):
            img, _ = cifar10[i]
            img_np = img.numpy()
            img_np = np.transpose(img_np, (1, 2, 0)) # (32, 32, 3)
            img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
            yield [img_np]

    print("[CONVERTER] Convertendo modelo Keras para TFLite com quantização INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Garantir que a quantização seja INT8 pura
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Salva o arquivo .tflite
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'models'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    caminho_tflite = os.path.join(output_dir, "representacao_congelada.tflite")
    with open(caminho_tflite, "wb") as f:
        f.write(tflite_model)
    print(f"[CONVERTER] Modelo TFLite salvo em: {caminho_tflite}")

    # Escrever o array C++
    print("[CONVERTER] Gerando array C++ (modelo_representacao.h)...")
    caminho_header = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'firmware', 'include', 'modelo_representacao.h'))
    
    with open(caminho_header, "w") as f:
        f.write("#ifndef MODELO_REPRESENTACAO_H\n")
        f.write("#define MODELO_REPRESENTACAO_H\n\n")
        f.write(f"// Modelo TFLite quantizado INT8 da Camada de Representacao\n")
        f.write(f"// Tamanho do arquivo: {len(tflite_model)} bytes\n\n")
        f.write(f"const unsigned int modelo_representacao_len = {len(tflite_model)};\n\n")
        f.write("alignas(8) const unsigned char modelo_representacao[] = {\n")
        
        # Escreve em formato hex com limite de 12 bytes por linha
        hex_data = [f"0x{b:02x}" for b in tflite_model]
        for i in range(0, len(hex_data), 12):
            linha = ", ".join(hex_data[i:i+12])
            if i + 12 < len(hex_data):
                f.write(f"    {linha},\n")
            else:
                f.write(f"    {linha}\n")
                
        f.write("};\n\n")
        f.write("#endif // MODELO_REPRESENTACAO_H\n")
        
    print(f"[CONVERTER] Cabeçalho C++ salvo em: {caminho_header}")

if __name__ == "__main__":
    main()
