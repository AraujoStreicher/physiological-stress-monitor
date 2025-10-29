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

# Arquivo com todas as funcoes e codigos referentes a carregar dados e extrair features

import pandas as pd
import os


def load_data(base_path):

    file_path = os.path.join(base_path, 'dataset')
    
    train_df = pd.read_csv(os.path.join(file_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(file_path, 'test.csv'))
    users_info = pd.read_csv(os.path.join(file_path, 'users_info.txt'), sep=',')
    data_dict = pd.read_csv(os.path.join(file_path, 'Data_Dictionary.csv'), encoding='latin1')

    return train_df, test_df, users_info, data_dict


def load_sensor_file(file_path):
    try:
        df = pd.read_csv(file_path, header=None, on_bad_lines='skip')
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()
    
    ## primeira linha é timestamp e a segunda é calibracao
    df = df.drop(index=[0,1])
    
    df = df.reset_index(drop=True).apply(pd.to_numeric, errors='coerce')

    ## remove valores absurdos
    df = df[(df.abs() < 1e6).all(axis=1)]
    
    return df



def extract_features(df, prefix):
    feats = {}
    for col in df.columns:
        col_name = f"{prefix}_{col}"
        arr = df[col].dropna()
        feats[f"{col_name}_mean"] = arr.mean()
        feats[f"{col_name}_std"] = arr.std()
        feats[f"{col_name}_min"] = arr.min()
        feats[f"{col_name}_max"] = arr.max()
        feats[f"{col_name}_median"] = arr.median()
        feats[f"{col_name}_range"] = arr.max() - arr.min()
    return feats


def build_user_features(user_id, BASE_PATH):
    WEARABLES_DIR = os.path.join(BASE_PATH, 'dataset', 'wearables')
    user_dir = os.path.join(WEARABLES_DIR, user_id)
    features = {"Id": user_id}
    
    for fname in os.listdir(user_dir):
        if not fname.endswith(".csv") or fname == "tags.csv":
            continue
        sensor_name = fname.replace(".csv", "")
        sensor_path = os.path.join(user_dir, fname)
        
        try:
            df_sensor = load_sensor_file(sensor_path)
            feats = extract_features(df_sensor, prefix=sensor_name)
            features.update(feats)
        except Exception as e:
            print(f"Erro ao processar {user_id}/{fname}: {e}")
            continue
    
    return features


def build_features_df(user_ids, BASE_PATH):
    all_feats = []
    for uid in user_ids:
        feats = build_user_features(uid, BASE_PATH)
        all_feats.append(feats)
    return pd.DataFrame(all_feats)