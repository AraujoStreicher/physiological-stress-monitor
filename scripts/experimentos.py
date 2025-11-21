# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Gabriel Araujo Streicher
# RA: 822485
# ################################################################

# Arquivo com todas as funcoes e codigos referentes aos experimentos


from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def fit_kfold_grid_search(name_model, model, param_grid, X_train, Y_train, use_scaler=False, error_score=np.nan):
    """
    Funcao para realizar o pipeline de K-Fold Cross Validation com Grid Search.
    """
    ## Pipeline
    pipeline_list = []
    if use_scaler:
        pipeline_list.append(('scaler', StandardScaler()))
    
    pipeline_list.append((name_model, model))
    pipeline = Pipeline(pipeline_list)

    ## kfold cross validation com grid search
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='roc_auc_ovr',   
        cv=cv,
        verbose=2,
        n_jobs=-1,
        error_score=error_score
    )

    ## fit do grid search
    grid.fit(X_train, Y_train)
    return grid

def plot_learning_curve(model, X, Y, cv=3, scoring="roc_auc_ovr"):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        Y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)


    plt.figure(figsize=(10,6))
    plt.plot(train_sizes, train_mean, label="Treino", color="green")
    plt.plot(train_sizes, val_mean,   label="Validação", color="red")


    plt.title("Curva de Aprendizado")
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel(scoring.capitalize())
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()