# Detecção de Estresse e Esforço Físico a partir de Sinais Fisiológicos

O objetivo deste trabalho é investigar e aplicar métodos de aprendizado supervisionado para classificar o estado fisiológico de usuários (STRESS, ANAEROBIC e AEROBIC) utilizando séries temporais coletadas por sensores de dispositivos vestíveis (wearables). Este projeto faz parte da avaliacão
da disciplina de Aprendizado de Maquina da UFSCar em 2025.

## 🧷 Pipeline
 1. **Pré-processamento**: Limpeza de dados, tratamento de valores ausentes e outliers.
 2. **Engenharia de Atributos**: Extração de métricas estatísticas e de energia das séries temporais dos sensores.
 3. **Análise Exploratória**: Visualização da distribuição dos dados e identificação de padrões.
 4. **Modelagem**: Experimentos com diversos algoritmos, incluindo Gradient Boosting (XGBoost), SVM, KNN, Random Forest e Redes Neurais.
 5. **Avaliação**: Validação dos modelos utilizando métricas como AUC e curvas de aprendizado.

## 📂 Estrutura de Arquivos

A organização do projeto é a seguinte:

* `main.ipynb`: Notebook principal que orquestra todo o fluxo de trabalho, desde o carregamento dos dados até à análise dos resultados.
* `relatorio.pdf`: Relatório técnico detalhado descrevendo a metodologia, fundamentação teórica e discussão dos resultados.
* `figs/`: Diretório contendo imagens e logos utilizados no projeto.
* `scripts/`: Módulos Python auxiliares para manter o código do notebook limpo e organizado:
    * `preprocessamento.py`: Classes e funções para carregamento e limpeza de dados (ex: classe `DataLoader`).
    * `analise_exploratoria.py`: Funções para geração de gráficos e visualização de dados.
    * `experimentos.py`: Funções para treino, validação cruzada (`fit_kfold_grid_search`) e pipelines de modelos.
    * `analise_resultados.py`: Funções para avaliação de métricas e plotagem de curvas de aprendizado.

## 🛠️ Tecnologias e Dependências

O projeto foi desenvolvido em **Python** e utiliza as seguintes bibliotecas principais:

* `pandas` & `numpy`: Manipulação de dados e álgebra linear.
* `matplotlib` & `seaborn`: Visualização de dados.
* `scikit-learn`: Algoritmos de ML, pré-processamento e métricas.
* `xgboost`: Implementação otimizada de Gradient Boosting.

## ✒️ Autoria
* Aluno: Gabriel Araujo Streicher
* Instituição: UFSCar - Campus Sorocaba
* Disciplina: Aprendizado de Máquina (Prof. Dr. Tiago A. Almeida)
