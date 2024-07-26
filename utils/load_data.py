import numpy as np
import tensorflow as tf
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    # Carregar dataset exemplo (CIFAR-10) e filtrar apenas cães e gatos
    (train_images, train_labels), (val_images, val_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, val_images = train_images / 255.0, val_images / 255.0

    # Filtrar apenas as classes de cães e gatos (label 3 e 5 no CIFAR-10)
    train_filter = np.where((train_labels == 3) | (train_labels == 5))
    val_filter = np.where((val_labels == 3) | (val_labels == 5))
    
    train_images, train_labels = train_images[train_filter], train_labels[train_filter]
    val_images, val_labels = val_images[val_filter], val_labels[val_filter]

    # Alterar labels para binárias (0 = gato, 1 = cão)
    train_labels = np.where(train_labels == 3, 0, 1)
    val_labels = np.where(val_labels == 3, 0, 1)

    # One-hot encode as labels
    train_labels = tf.keras.utils.to_categorical(train_labels, 2)
    val_labels = tf.keras.utils.to_categorical(val_labels, 2)

    return (train_images, train_labels), (val_images, val_labels)

def load_my_data(data_dir, img_size=(128, 128), batch_size=32, val_split=0.2):

    """
    Carrega os dados de imagens dos diretórios especificados.

    Parâmetros:
    -----------
    data_dir: str
        O diretório raiz onde os dados estão armazenados.
    img_size: tuple
        O tamanho das imagens para redimensionamento. Padrão é (128, 128).
    batch_size: int
        O tamanho do batch para o treinamento. Padrão é 32.
    val_split: float
        A fração dos dados a serem usados para validação. Padrão é 0.2 (20%).

    Retorna:
    --------
    train_ds: tf.data.Dataset
        O dataset de treinamento.
    val_ds: tf.data.Dataset
        O dataset de validação.
    """
    
    # Gerador de dados com aumento de imagem para o conjunto de treinamento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,  # Divisão do conjunto de validação
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Gerador de dados sem aumento de imagem para o conjunto de validação
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split  # Divisão do conjunto de validação
    )

    train_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Conjunto de treinamento
    )

    val_ds = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Conjunto de validação
    )

    return train_ds, val_ds