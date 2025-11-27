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

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from xgboost import plot_importance


def plot_metrics(eval_dict):
    
    data = []
    for model_key, metrics in eval_dict.items():
        names = metrics.get('name', [])
        accs = metrics.get('acc', [])
        aucs = metrics.get('auc', [])
        
        for i in range(len(names)):
            data.append({
                'Modelo': names[i],
                'Acurácia': accs[i],
                'AUC': aucs[i]
            })
            
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars="Modelo", var_name="Métrica", value_name="Score")


    
    plt.figure(figsize=(15, 6))
    ax = sns.barplot(
        data=df_melted, 
        x="Modelo", 
        y="Score", 
        hue="Métrica", 
        palette="pastel",
        edgecolor="black", 
        linewidth=0.5
    )

    plt.title('Comparação de Performance', fontsize=14)
    plt.ylim(0, 1.15)
    plt.legend(loc='upper right')
    # 
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.show()



def plot_feature_importance_comparison(model_no):
    fig, axes = plt.subplots(1, 1, figsize=(9, 8))

    plot_importance(model_no, ax=axes, max_num_features=15, height=0.5, color='blue')
    axes.set_title('Top 15 Features Importantes')
    
    plt.tight_layout()
    plt.show()
