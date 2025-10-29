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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribuicoes_basicas(train_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    ## 1. Classes
    label_counts = train_df['Label'].value_counts()
    axes[0, 0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Distribuicao das Classes', fontsize=14, fontweight='bold')
    
    ## 2. Genero
    if 'Gender' in train_df.columns:
        gender_counts = train_df['Gender'].value_counts()
        sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=axes[0, 1], palette='pastel', hue=gender_counts.values)
        axes[0, 1].set_title('Distribuicao por Genero', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Genero')
        axes[0, 1].set_ylabel('Quantidade')
    
    ## 3. Idade
    if 'Age' in train_df.columns:
        train_df['Age'] = pd.to_numeric(train_df['Age'], errors='coerce')

        sns.histplot(train_df['Age'], bins=15, kde=True, ax=axes[0, 2], color='skyblue')
        axes[0, 2].set_title('Distribuicao de Idade', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Idade')
        axes[0, 2].set_ylabel('Frequencia')
        axes[0, 2].axvline(train_df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["Age"].mean():.1f}')
        axes[0, 2].legend()
    
    ## 4. Atividade fisica regular
    if 'Does physical activity regularly?' in train_df.columns:
        activity_counts = train_df['Does physical activity regularly?'].value_counts()
        sns.barplot(x=activity_counts.index, y=activity_counts.values, ax=axes[1, 0], palette='pastel', hue=activity_counts.values)
        axes[1, 0].set_title('Pratica Atividade Fisica Regular?', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Valor')
        axes[1, 0].set_ylabel('Contagem')

    ## 5. Peso
    if 'Weight (kg)' in train_df.columns:
        train_df['Weight (kg)'] = pd.to_numeric(train_df['Weight (kg)'], errors='coerce')

        sns.histplot(train_df['Weight (kg)'], bins=20, kde=True, ax=axes[1, 1], color='coral')
        axes[1, 1].set_title('Distribuicao de Peso', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Peso')
        axes[1, 1].set_ylabel('Frequencia')
        axes[1, 1].axvline(train_df['Weight (kg)'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["Weight (kg)"].mean():.1f}')
        axes[1, 1].legend()

    ## 6. Altura
    if 'Height (cm)' in train_df.columns:
        train_df['Height (cm)'] = pd.to_numeric(train_df['Height (cm)'], errors='coerce')

        sns.histplot(train_df['Height (cm)'], bins=20, kde=True, ax=axes[1, 2], color='mediumpurple')
        axes[1, 2].set_title('Distribuicao de Altura', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Altura')
        axes[1, 2].set_ylabel('Frequencia')
        axes[1, 2].axvline(train_df['Height (cm)'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["Height (cm)"].mean():.1f}')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('01_distribuicoes_basicas.png', dpi=300, bbox_inches='tight')
    plt.show()