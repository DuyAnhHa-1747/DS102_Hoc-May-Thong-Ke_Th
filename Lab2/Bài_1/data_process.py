import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_process_logistic(data_path: str = 'tiki_diengiadung.csv', test_size: float = 0.25, random_state: int = 42):
    np.random.seed(random_state)
    
    df = pd.read_csv(data_path)
    features = ['gia_goc', 'pham_tram_giam_gia', 'chi_so_danh_gia', 'so_luong_binh_luan', 'so_luong_ban']
    
    df[features] = df[features].fillna(df[features].median())
    
    median_price = df['gia_ban'].median()
    y = (df['gia_ban'] > median_price).astype(int).values
    X = df[features].values
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_bin = scaler.fit_transform(X_train_bin)
    X_test_bin = scaler.transform(X_test_bin)
    
    return X_train_bin, X_test_bin, y_train_bin, y_test_bin, scaler, features