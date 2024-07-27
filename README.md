# image-classifier-new

Projeto de Classificação de Imagens de Gatos e Cachorros
Este projeto é uma implementação de um modelo de aprendizado profundo para a classificação de imagens de gatos e cachorros usando TensorFlow e Keras. A arquitetura do modelo é baseada em uma rede neural convolucional (CNN) que é treinada para distinguir entre imagens de gatos e cachorros.

# Estrutura do Projeto
.
├── dataset
│   ├── cat (contém imagens de gatos)
│   └── dog (contém imagens de cachorros)
├── estrutura_projeto.txt
├── main.py
├── models
├── notebook
│   └── train.ipynb
└── utils
    ├── adapter.py
    ├── adapter_tensorflow.py
    ├── architecture.py
    ├── load_data.py
    └── __init__.py

* dataset/: Contém as imagens de treino e validação organizadas em subdiretórios cat e dog.
* main.py: Script principal para executar o treinamento e a avaliação do modelo.
* models/: Diretório para salvar os modelos treinados.
* notebook/: Contém o notebook Jupyter train.ipynb para experimentação e treinamento interativo do modelo.
* utils/: Contém módulos auxiliares como adapter.py, adapter_tensorflow.py, architecture.py e load_data.py.

# Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.

