import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_process_softmax(data_path: str = 'tiki_diengiadung.csv', test_size: float = 0.25, random_state: int = 42):
    np.random.seed(random_state)
    df = pd.read_csv(data_path)
    
    features = [
        'gia_ban', 'gia_goc', 'pham_tram_giam_gia', 
        'chi_so_danh_gia', 'so_luong_binh_luan', 'so_luong_ban'
    ]

    df_clean = df.dropna(subset=['nhan_gia_ca']).copy()
    df_clean[features] = df_clean[features].fillna(df_clean[features].median())

    X = df_clean[features].values
    
    class_names, y = np.unique(df_clean['nhan_gia_ca'], return_inverse=True)
    num_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, num_classes, class_names, features