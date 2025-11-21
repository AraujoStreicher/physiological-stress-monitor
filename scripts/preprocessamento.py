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

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import pandas as pd
import os
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer

class DataLoader:
    """
    Classe responsavel por carregar os dados do dataset e extrair features das series temporais dos sensores.
    
    @param base_path: caminho base onde o diretorio dataset esta localizado.
    """
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self):
        """
        Carrega os arquivos de dados: train.csv, test.csv, users_info.txt e Data_Dictionary.csv.
        """

        file_path = os.path.join(self.base_path, 'dataset')
        
        train_df = pd.read_csv(os.path.join(file_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(file_path, 'test.csv'))
        users_info = pd.read_csv(os.path.join(file_path, 'users_info.txt'), sep=',')
        data_dict = pd.read_csv(os.path.join(file_path, 'Data_Dictionary.csv'), encoding='latin1')

        return train_df, test_df, users_info, data_dict


    def load_sensor_file(self, file_path):
        """
        Carrega um sensor individual de um usuario, removendo linhas invalidas e convertendo para numerico.
        Remove as duas primeiras linhas (timestamp e freq).
        Em caso de excecao, retorna um DataFrame vazio.

        @param file_path: caminho completo do arquivo CSV do sensor.
        """
        try:
            df = pd.read_csv(file_path, header=None, on_bad_lines='skip')
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()
        
        ## primeira linha é timestamp e a segunda é freq
        df = df.drop(index=[0,1])

        ## se for IBI pega só a segunda coluna
        if 'IBI' in file_path:
            df = df.drop(columns=[0])
        
        df = df.reset_index(drop=True).apply(pd.to_numeric, errors='coerce')

        ## remove valores absurdos
        df = df[(df.abs() < 1e6).all(axis=1)]
        
        return df



    def extract_features(self, df, sensor):
        """
        Recebe um dataframe de uma ou mais series temporais separados por colunas e extrai features da serie.
        Retorna um dicionario com as features MEAN, STD, MIN, MAX, MEDIAN, RANGE de cada coluna. 

        @param df: DataFrame com as series temporais do sensor.
        @param prefix: String com o nome do sensor.
        """ 
        feats = {}
        for col in df.columns:
            col_name = f"{sensor}_{col}"
            arr = df[col].dropna()
            feats[f"{col_name}_mean"] = arr.mean()
            feats[f"{col_name}_std"] = arr.std()
            feats[f"{col_name}_min"] = arr.min()
            feats[f"{col_name}_max"] = arr.max()
            feats[f"{col_name}_median"] = arr.median()
        return feats


    def build_user_features(self, user_id):
        """
        Recebe um id de usuário e constroi um dicionario com as caracteristicas
        de todos os sensores que foi utilizado.

        @param user_id: Id do usuário.
        """

        WEARABLES_DIR = os.path.join(self.base_path, 'dataset', 'wearables')
        user_dir = os.path.join(WEARABLES_DIR, user_id)
        features = {"Id": user_id}
        
        for fname in os.listdir(user_dir):
            if not fname.endswith(".csv") or fname == "tags.csv":
                continue
            sensor_name = fname.replace(".csv", "")
            sensor_path = os.path.join(user_dir, fname)
            
            try:
                df_sensor = self.load_sensor_file(sensor_path)
                feats = self.extract_features(df_sensor, sensor=sensor_name)
                features.update(feats)
            except Exception as e:
                print(f"Erro ao processar {user_id}/{fname}: {e}")
                continue
        
        return features


    def build_features_df(self, user_ids):
        """
        Constroi um DataFrame com todas as features das series temporais geradas
        pelos sensores de todos os usuarios informados pelo parâmetro user_ids.

        @param user_ids: Lista com todos os ids que devem ser inseridos nessa construção. 
        """
        all_feats = []
        for uid in user_ids:
            feats = self.build_user_features(uid)
            all_feats.append(feats)
        return pd.DataFrame(all_feats)


    def get_average_series(self, train_df, sensor):
        """
        Retorna um dicionario com a serie temporal da media de um sensor. Essa media
        e calculada separadamente para cada rotulo de saida.

        @param train_df: DataFrame de treino.
        @param sensor: Nome do sensor que sera feito a media.
        """
        user_ids = train_df['Id'].unique()
        series_dict = {}
        
        for class_label in train_df['Label'].unique():
            user_ids = train_df[train_df['Label'] == class_label]['Id'].unique()
            class_series = []
            for uid in user_ids:
                WEARABLES_DIR = os.path.join(self.base_path, 'dataset', 'wearables')
                sensor_path = os.path.join(WEARABLES_DIR, uid, f"{sensor}.csv")
                df_sensor = self.load_sensor_file(sensor_path)
                class_series.append(df_sensor)
            
            
            min_length = min([len(s) for s in class_series if not s.empty])
            aligned_series = np.stack([s[:min_length] for s in class_series if not s.empty])
            series_dict[class_label] = aligned_series.mean(axis=0)
        
        return series_dict
    

    def process_ACC(self,df, process_axis=True):
        """
        Processa os dados do acelerometro para calcular a magnitude do vetor de aceleracao.
        Retorna um DataFrame com o dado da magnitude e sem os dados originais do acelerometro.

        @param df: DataFrame com os dados originais para serem processados.
        """

        acc_columns = [col for col in df.columns if col.startswith('ACC_')]
        new_df = df.drop(columns=acc_columns, inplace=False)
        
        all_mags = []
        mag_df = pd.DataFrame(columns=['Id', 'ACC_Magnitude'])
        for uid in new_df['Id'].unique():
            acc = self.load_sensor_file(os.path.join(self.base_path, 'dataset', 'wearables', uid, 'ACC.csv'))
            ## Transforma negativos em positivos -> remove direcionalidade
            acc = acc.abs()
            
            ## Há sensores travados em alguns eixos, vamos ignorar esses eixos no calculo para nao perder o resto
            if(process_axis == True):
                valid_cols = []
                for i,col in enumerate(acc):
                    most_common_count = pd.Series(acc[col].values).value_counts().iloc[0]
                    frac_repeats = most_common_count / len(acc[col])

                    if frac_repeats < 0.7:
                        valid_cols.append(acc[col].values)
            else:
                valid_cols = [acc[col].values for col in acc.columns]


            ## Calcula a magnitude
            mag = np.sqrt((np.array(valid_cols)**2).sum(axis=0))
            mag_feats = {
                'ACC_Mean': mag.mean(),
                'ACC_Std': mag.std(),
                'ACC_Max': mag.max(),
                'ACC_Min': mag.min(),
                'ACC_Energy': np.sum(mag**2)
            }
            all_mags.append({'Id': uid, **mag_feats})
        
        mag_df = pd.DataFrame(all_mags)
        merged_df = pd.merge(new_df, mag_df, on='Id', how='left')
        return merged_df



def encoder_imputer(train_df, test_df):
    TARGET_COL = "Label" 
    ID_COL = "Id"

    ## Cria nosso encoder
    le_target = LabelEncoder()
    ## Codifica variavel alvo
    train_df[TARGET_COL] = le_target.fit_transform(train_df[TARGET_COL])


    X = train_df.drop(columns=[TARGET_COL, ID_COL])
    Y = train_df[TARGET_COL]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns
    
    ## Imputa valores faltantes com a moda para variáveis categoricas
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

    ## Codifica variaveis categoricas de treino e teste
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
        ## Teste faz o mapeamento correto
        encoders[col] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        test_df[col] = test_df[col].map(mapping).fillna(-1).astype(int)

    ## Imputa valores faltantes com a media para variáveis numericos
    imputer = SimpleImputer(strategy="mean")
    X[num_cols] = imputer.fit_transform(X[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])

    return X, Y, test_df, le_target

            