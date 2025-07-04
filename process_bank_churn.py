from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def split_data(
    df: pd.DataFrame,
    target_col: str = 'Exited',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Розбиває дані на тренувальний та валідаційний набори.

    Args:
        df: Вхідний DataFrame
        target_col: Назва цільової змінної
        test_size: Розмір валідаційної вибірки
        random_state: random seed для відтворюваності

    Returns:
        Кортеж (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        df, stratify=df[target_col],
        test_size=test_size,
        random_state=random_state
    )
    return train_df, val_df


def scale_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler.

    Args:
        train_df: Тренувальні дані
        val_df: Валідаційні дані
        numeric_cols: Список числових колонок

    Returns:
        (масштабований train_df, масштабований val_df, fitted scaler)
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler


def encode_categorical_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Виконує one-hot кодування категоріальних ознак.

    Args:
        train_df: Тренувальні дані
        val_df: Валідаційні дані
        categorical_cols: Список категоріальних колонок

    Returns:
        (train_df з one-hot, val_df з one-hot, fitted encoder, список нових колонок)
    """
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist')
    encoder.fit(train_df[categorical_cols])
    encoded_train = encoder.transform(train_df[categorical_cols])
    encoded_val = encoder.transform(val_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_df[encoded_cols] = encoded_train
    val_df[encoded_cols] = encoded_val
    return train_df, val_df, encoder, encoded_cols


def preprocess_data(
    df: pd.DataFrame,
    scaler_numeric: bool = False
) -> Dict[str, object]:
    """
    Повний препроцесинг даних: розбиття, масштабування (опційно) та кодування.

    Args:
        df: Вхідний DataFrame
        scaler_numeric: чи масштабувати числові ознаки

    Returns:
        словник з X_train, X_val, y_train, y_val, input_cols, scaler, encoder
    """
    train_df, val_df = split_data(df)
    
    input_cols = list(df.columns)[3:-1]
    target_col = 'Exited'
    
    train_inputs = train_df[input_cols].copy()
    val_inputs = val_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_targets = val_df[target_col].copy()
    
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()
    
    scaler: Optional[MinMaxScaler] = None
    if scaler_numeric:
        train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
        
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(train_inputs, val_inputs, categorical_cols)
    
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]
    
    return {
        'X_train': X_train,
        'y_train': train_targets,
        'X_val': X_val,
        'y_val': val_targets,
        'input_cols': numeric_cols + encoded_cols,
        'scaler': scaler,
        'encoder': encoder
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Препроцесинг нових даних для прогнозування (test.csv).

    Args:
        new_df: новий DataFrame
        input_cols: список ознак, що використовувалися при тренуванні
        numeric_cols: список числових ознак
        categorical_cols: список категоріальних ознак
        scaler: вже навчений скейлер
        encoder: вже навчений енкодер

    Returns:
        препроцесований DataFrame
    """
    inputs = new_df[input_cols].copy()
    
    if scaler:
        inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
    
    encoded = encoder.transform(inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    inputs[encoded_cols] = encoded
    
    preprocessed_df = inputs[numeric_cols + encoded_cols]
    
    return preprocessed_df
