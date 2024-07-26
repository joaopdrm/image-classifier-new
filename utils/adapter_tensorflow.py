from utils.adapter import Modeladapter
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

class Adaptertensorflow(Modeladapter):

    def __init__(self, model: tf.keras.Model):
        """
        Inicializa o adaptador com um modelo TensorFlow Keras.

        Parâmetros:
        -----------
        model: tf.keras.Model
            O modelo Keras a ser usado pelo adaptador.
        """
        self.model = model

    def train(self, train_ds, shuffle: bool = True, test_ds=None, val_ds=None, **kwargs):
        """
        Treina o modelo com o conjunto de dados fornecido.

        Parâmetros:
        -----------
        train_ds: Dataset de treino
        shuffle: bool, default=True
            Indica se os dados de treino devem ser embaralhados.
        test_ds: Dataset de teste, opcional
        val_ds: Dataset de validação, opcional
        kwargs: dict
            Parâmetros adicionais, como epochs e batch_size.
        """
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="/tmp/checkpoint.weights.h5",
            monitor="accuracy",
            mode='max',
            save_weights_only=True,
            save_best_only=True
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=5,
            verbose=1,
            mode="auto",
            restore_best_weights=True
        )

        tensorboard_callback = TensorBoard(
            log_dir="logs",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None
        )

        epochs = kwargs.get("epochs", 30)
        batch_size = kwargs.get("batch_size", 32)

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=epochs,
            decay_rate=1,
            staircase=False
        )

        print(type(self.model))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=[model_checkpoint_callback, early_stopping, tensorboard_callback],
            validation_data=val_ds if val_ds is not None else None
        )

    def predict_data(self, X_v: np.ndarray):
        """
        Faz previsões com o modelo treinado.

        Parâmetros:
        -----------
        X_v: np.ndarray
            Os dados de entrada para fazer previsões.

        Retorna:
        --------
        y_pred: np.ndarray
            As previsões feitas pelo modelo.
        """
        y_pred = self.model.predict(X_v)
        return y_pred

    def save(self, path_name: str):
        """
        Salva o modelo em um arquivo.

        Parâmetros:
        -----------
        path_name: str
            O caminho onde o modelo deve ser salvo.
        """
        self.model.save(path_name)
