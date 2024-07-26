from abc import ABC, abstractmethod 
import numpy as np 

class Modeladapter(ABC):

    @abstractmethod
    def train(self, train_ds: np.ndarray, shuffle:bool = True, test_ds: np.ndarray = None,
    val_ds: np.ndarray = None, **kwargs):

        """
        Método abstrato para treinamento de modelo

        Parâmetros:

        train_ds : np.ndarray.
            Dados de treinamento.
        shuffle : bool, opcional.
            Se os dados devem ser embaralhados (default é True)
        test_ds : np.ndarray.
            Dados de teste.
        val_ds : np.ndarray.
            Dados de validação do modelo.

        **kwargs : dict
            Argumentos adicionais para o treinamento
        """
        pass


    @abstractmethod
    def predict_data(self,X_v:np.ndarray):

        """
        Método abstrato para realizar predições em imagens.

        Parâmetros:
        ----------
        X_v : np.ndarray
            Dados de entrada para predição.

        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            Previsões brutais e previsões binárias.
        """
        pass

    @abstractmethod
    def save(self, path_name: str):
        """
        Método abstrato para salvar o modelo.

        Parâmetros:
        ----------
        path_name : str
            Caminho onde o modelo será salvo.
        """
        pass