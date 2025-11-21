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
import os

def plot_distribuicoes_basicas(train_df):
    """
    Plota as distribuições de algumas informacoes dos usuarios em relacao ao
    rotulo de saida.

    @param train_df: DataFrame que será analisado.
    """
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
    

    def save(train_df):
        label_counts = train_df['Label'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6,6))
        ax1.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        ax1.set_title('Distribuição das Classes', fontsize=14, fontweight='bold')
        fig1.tight_layout()
        fig1.savefig("distribuicao_classes.png", dpi=300, bbox_inches="tight")
        plt.close(fig1)


        activity_counts = train_df['Does physical activity regularly?'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6,5))
        sns.barplot(x=activity_counts.index, y=activity_counts.values, ax=ax2, palette='pastel', hue=activity_counts.values)
        ax2.set_title('Atividade Física Regular', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Valor')
        ax2.set_ylabel('Contagem')
        fig2.tight_layout()
        fig2.savefig("atividade_fisica.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(6,5))
        gender_counts = train_df['Gender'].value_counts()
        sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax3, palette='pastel', hue=gender_counts.values)
        ax3.set_title('Distribuicao por Genero', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Genero')
        ax3.set_ylabel('Quantidade')
        fig3.tight_layout()
        fig3.savefig("distribuicao_genero.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)

    save(train_df)

def plot_series_grouped_by_label(series_dict, sensor):
    """
    Recebe um dicionario com uma serie temporal para cada rotulo de saida possivel.
    Plota essas series temporais.

    @param series_dict: O dicionario de entrada.
    @param sensor: O nome do sensor que está sendo plotado.
    """
    min_length = min(len(serie_media) for serie_media in series_dict.values())

    plt.figure(figsize=(10, 5))
    for classe, serie_media in series_dict.items():
        plt.plot(serie_media[:min_length], label=classe)
    plt.title(f"Média das séries do sensor {sensor} por classe")
    plt.xlabel("Tempo (amostras)")
    plt.ylabel(f"Média do sinal ({sensor})")
    plt.legend()
    plt.show()


def boxplot_sensor(df, col=''):
    """
    Boxplot padrao para visualizar a distribuicao por classe de um sensor.
    @param df: DataFrame com os dados.
    @param col: Nome da coluna do sensor a ser plotado.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df,
        x='Label',
        y=col,   
        palette='pastel',
        linewidth=1.2,
        hue='Label',
        showfliers=True
    )

    plt.title(f'Distribuição de {col} por classe', fontsize=15, fontweight='bold')
    plt.xlabel('Classe', fontsize=12)
    plt.ylabel(f'{col}', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_sensor_by_label(df, base_path, dataloader, sensor_name, samples_per_class=1):
    """
    Plota a série temporal de um mesmo sensor para uma amostra de cada classe.
    
    df: dataframe contendo Id e Label
    base_path: caminho base do dataset
    dataloader: instância da classe DataLoader
    sensor_name: nome do arquivo do sensor, ex: 'BVP.csv', 'ACC.csv'
    samples_per_class: número de usuários por classe
    """

    plt.figure(figsize=(14, 6))

    for label in df['Label'].unique():
        users = df[df['Label'] == label]['Id'].head(samples_per_class)

        for user_id in users:
            file_path = os.path.join(base_path, 'dataset', 'wearables', user_id, sensor_name)
            df_sensor = dataloader.load_sensor_file(file_path)
            plt.plot(df_sensor.values, label=f"{label} - {user_id}")


    plt.title(f"Série Temporal de {sensor_name}")
    plt.xlabel("Tempo")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

