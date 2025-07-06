from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_data(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the input dataframe into training and validation sets with stratification.
    
    Args:
        df: The input dataframe.
        target_col: The name of the target column.
        test_size: Proportion of the validation set.
        random_state: Random seed for reproducibility.
        
    Returns:
        A tuple of (train_inputs, val_inputs, train_targets, val_targets).
    """
    train_df, val_df = train_test_split(
        df, stratify=df[target_col], test_size=test_size, random_state=random_state
    )
    input_cols = list(df.columns)[3:-1]
    
    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()
    
    return train_inputs, val_inputs, train_targets, val_targets


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scales numeric columns using MinMaxScaler.
    
    Args:
        train_inputs: Training features.
        val_inputs: Validation features.
        numeric_cols: List of numeric columns to scale.
        
    Returns:
        Tuple of scaled training features, validation features, and the fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    
    return train_inputs, val_inputs, scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    One-hot encodes categorical columns, dropping the first category.
    
    Args:
        train_inputs: Training features.
        val_inputs: Validation features.
        categorical_cols: List of categorical columns to encode.
        
    Returns:
        Tuple of encoded training features, validation features, the fitted encoder, and the list of encoded column names.
    """
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist')
    encoder.fit(train_inputs[categorical_cols])
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_encoded = encoder.transform(train_inputs[categorical_cols])
    val_encoded = encoder.transform(val_inputs[categorical_cols])
    
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_inputs.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_cols, index=val_inputs.index)
    
    train_inputs = pd.concat([train_inputs.drop(columns=categorical_cols), train_encoded_df], axis=1)
    val_inputs = pd.concat([val_inputs.drop(columns=categorical_cols), val_encoded_df], axis=1)
    
    return train_inputs, val_inputs, encoder, encoded_cols


def preprocess_data(
    df: pd.DataFrame, 
    scale_numeric: bool = False
) -> Dict[str, object]:
    """
    Main data preprocessing pipeline: splitting, scaling, encoding.
    
    Args:
        df: Raw input dataframe.
        scale_numeric: Whether to scale numeric features using MinMaxScaler.
        
    Returns:
        A dictionary containing processed datasets, encoders, and metadata.
    """
    # Split
    train_inputs, val_inputs, train_targets, val_targets = split_data(df, target_col='Exited')
    
    # Identify feature types
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()
    
    # Scale
    scaler = None
    if scale_numeric:
        train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    
    # Encode
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(
        train_inputs, val_inputs, categorical_cols
    )
    
    result = {
        'X_train': train_inputs,
        'y_train': train_targets,
        'X_val': val_inputs,
        'y_val': val_targets,
        'data_cols': [list(df.columns)[3:-1], numeric_cols, categorical_cols, encoded_cols],
        'scaler': scaler,
        'encoder': encoder,
    }
    
    return result


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_columns: List[List[str]],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Preprocesses new data with previously-fitted scaler and encoder.
    
    Args:
        new_df: New raw dataframe.
        input_columns: List of lists with columns (original, numeric, categorical, encoded).
        scaler: Previously fitted MinMaxScaler (can be None).
        encoder: Previously fitted OneHotEncoder.
        
    Returns:
        Processed dataframe ready for prediction.
    """
    df_proc = new_df[input_columns[0]].copy()
    
    if scaler:
        df_proc[input_columns[1]] = scaler.transform(df_proc[input_columns[1]])
    
    encoded_array = encoder.transform(df_proc[input_columns[2]])
    encoded_df = pd.DataFrame(encoded_array, columns=input_columns[3], index=df_proc.index)
    
    final_df = pd.concat([df_proc.drop(columns=input_columns[2]), encoded_df], axis=1)
    
    return final_df
