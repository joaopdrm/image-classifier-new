import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from utils.adapter_tensorflow import Adaptertensorflow

class architectures:
    
    @staticmethod
    def cnn(input_shape=(128, 128, 3), num_classes=2):
        """
        Cria um modelo Keras para classificação de imagens de cães e gatos.

        Parâmetros:
        -----------
        input_shape: tuple
            A forma das imagens de entrada. Padrão é (128, 128, 3).
        num_classes: int
            O número de classes de saída. Padrão é 2 (cães e gatos).

        Retorna:
        --------
        model: tf.keras.Model
            O modelo Keras.
        """
        model = Sequential()

        # Primeira camada convolucional
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Segunda camada convolucional
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Terceira camada convolucional
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Camada flatten e densas
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return Adaptertensorflow(model)

    # para adicionar outras arquiteturas basta implementa-las nesta classe