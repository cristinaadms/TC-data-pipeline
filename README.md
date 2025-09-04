# TC-data-pipeline

Projeto de análise exploratória e pré-processamento dos conjuntos de dados **Adult (Census Income)** e **Titanic** utilizando Python e Jupyter Notebook.

## Estrutura do Projeto

```
TC-data-pipeline/
├── data-adult/
│   ├── adult_analysis.ipynb
│   └── adult.csv
├── data-titanic/
│   ├── titanic_analysis.ipynb
│   └── Titanic-Dataset.csv
├── README.md
└── LICENSE
```

## Descrição

Este projeto tem como objetivo realizar a análise exploratória, tratamento de dados ausentes, remoção de outliers, sumarização estatística e visualização dos dados dos conjuntos **Adult** e **Titanic**. Os notebooks apresentam passo a passo as principais técnicas de pré-processamento e análise de dados, incluindo:

- Importação e inspeção dos dados
- Tratamento de valores ausentes
- Remoção de outliers
- Conversão de variáveis categóricas
- Normalização de dados
- Sumarização estatística
- Visualizações gráficas
- Identificação de padrões

## Como executar

1. Clone o repositório
2. Instale manualmente as dependências necessárias:
   
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
   
4. Abra os notebooks `adult_analysis.ipynb` e `titanic_analysis.ipynb` no Jupyter ou VS Code e execute as células.

## Requisitos

- Python 3.8+
- Jupyter Notebook ou VS Code
- Bibliotecas:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## Dados Utilizados

- **Adult (Census Income):** Dados demográficos e de renda de adultos nos EUA.
- **Titanic:** Dados dos passageiros do Titanic, incluindo informações de sobrevivência.

## Resultados Esperados

- Relatórios de análise exploratória
- Gráficos e visualizações interativas
- Dados tratados e prontos para modelagem

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Créditos

Projeto desenvolvido por Cristina Sousa e Daniel Rodrigues para disciplina de Tópicos Especiais em Computação.
