import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

def melt_dataframe(df, id_vars_='periodId', var_name_='farmId', value_name_='power'):
    " Melt a DataFrame to have a single column for power values. "

#     assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."
#     assert id_vars_ in df.columns, f"Column '{id_vars_}' not found in DataFrame."
#     assert var_name_ in df.columns, f"Column '{var_name_}' not found in DataFrame."
#     assert value_name_ in df.columns, f"Column '{value_name_}' not found in DataFrame."

    df.index.names = [id_vars_]
    df_melt_with_nan = pd.melt(df.reset_index(), id_vars=id_vars_, value_vars=list(df.columns), var_name=var_name_, value_name=value_name_)
    df_melt_without_nan = df_melt_with_nan.dropna().reset_index(drop=True)
    return df_melt_without_nan, df_melt_with_nan



def _block_shuffle(df, blocksize, seed):
    " Shuffle blocks of data. "

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."
    assert isinstance(blocksize, int), "Blocksize must be an integer."
    assert blocksize > 0, "Blocksize must be greater than 0."
    assert isinstance(seed, int), "Seed must be an integer."

    # Extract data as a NumPy array
    data = df.values
    num_rows = len(data)
    # Shuffle block indices
    block_indices = list(range(num_rows // blocksize))
    random.Random(seed).shuffle(block_indices)
    # Shuffle blocks
    shuffled_blocks = [data[i * blocksize: (i + 1) * blocksize] for i in block_indices]
    # Concatenate shuffled blocks
    shuffled_data = np.concatenate(shuffled_blocks)
    # Convert to DataFrame
    df_shuffled = pd.DataFrame(shuffled_data, columns=df.columns)
    # Convert 'power' column to numeric
    df_shuffled['power'] = pd.to_numeric(df_shuffled['power'])
    return df_shuffled


def split_train_test(df, test_size=0.2, block=True, blocksize=48, seed=42):
    " Split data into training and validation sets. "

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."
    assert isinstance(test_size, float), "Test size must be a float."
    assert 0 < test_size < 1, "Test size must be between 0 and 1."
    assert isinstance(block, bool), "Block must be a boolean."
    assert isinstance(blocksize, int), "Blocksize must be an integer."
    assert blocksize > 0, "Blocksize must be greater than 0."
    assert isinstance(seed, int), "Seed must be an integer."

    if block:
        # Shuffle blocks if block is True
        df_block = _block_shuffle(df, blocksize, seed)
        training_df, validation_df = train_test_split(df_block, test_size=test_size, shuffle=False)
    else:
        # Split randomly if block is False
        training_df, validation_df = train_test_split(df, test_size=test_size, random_state=seed)
    # Reset index for both training and validation DataFrames
    training_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    return training_df, validation_df


class Normalizer:
    " A class to normalize and unnormalize power values using min-max scaling. "

    @staticmethod
    def normalize_power(training_df, validation_df, id_col='farmId', power_col='power'):
        " Normalize power values using min-max scaling. "

        assert isinstance(training_df, pd.DataFrame), "training_df must be a pandas DataFrame."
        assert isinstance(validation_df, pd.DataFrame), "validation_df must be a pandas DataFrame."
        assert id_col in training_df.columns, f"Column '{id_col}' not found in training_df."
        assert id_col in validation_df.columns, f"Column '{id_col}' not found in validation_df."
        assert power_col in training_df.columns, f"Column '{power_col}' not found in training_df."
        assert power_col in validation_df.columns, f"Column '{power_col}' not found in validation_df."

        # Calculate max and min mappings for each farmId
        max_mapping_train = training_df.groupby(id_col)[power_col].max()
        min_mapping_train = training_df.groupby(id_col)[power_col].min()
        # Map max and min values to each row based on farmId
        training_df['max'] = training_df[id_col].map(max_mapping_train)
        training_df['min'] = training_df[id_col].map(min_mapping_train)
        validation_df['max'] = validation_df[id_col].map(max_mapping_train)
        validation_df['min'] = validation_df[id_col].map(min_mapping_train)
        # Normalize power values using min-max scaling
        norm_training_df = training_df.assign(power_z=lambda row: (row[power_col] - row['min']) / (row['max'] - row['min'])).drop(['max', 'min', power_col], axis=1)
        norm_validation_df = validation_df.assign(power_z=lambda row: (row[power_col] - row['min']) / (row['max'] - row['min'])).drop(['max', 'min', power_col], axis=1)
        return norm_training_df, norm_validation_df, max_mapping_train, min_mapping_train


    @staticmethod
    def unnormalize_power(norm_df, id_col, power_col, max_mapping, min_mapping, cols_to_unscale):
        " Unnormalize power values using min-max scaling. "

        assert isinstance(norm_df, pd.DataFrame), "norm_df must be a pandas DataFrame."
        assert id_col in norm_df.columns, f"Column '{id_col}' not found in norm_df."
        assert power_col in norm_df.columns, f"Column '{power_col}' not found in norm_df."
        assert isinstance(max_mapping, pd.Series), "max_mapping must be a pandas Series."
        assert isinstance(min_mapping, pd.Series), "min_mapping must be a pandas Series."
        assert all(col in norm_df.columns for col in cols_to_unscale), "All columns in cols_to_unscale must be present in norm_df."

        # Map max and min values to each row based on farmId
        norm_df['max'] = norm_df[id_col].map(max_mapping)
        norm_df['min'] = norm_df[id_col].map(min_mapping)
        # Unnormalize the specified columns
        for col in cols_to_unscale:
            norm_df[col] = norm_df['min'] + norm_df[col] * (norm_df['max'] - norm_df['min'])
        # Drop unnecessary columns
        norm_df.drop(['max', 'min', power_col], axis=1, inplace=True)
        return norm_df


def preprocess_ids(training_df, validation_df):
    " Preprocess period and farm IDs for factorization. "

    # Check input types
    if not all(isinstance(df, pd.DataFrame) for df in [training_df, validation_df]):
        raise TypeError("Both training_df and validation_df must be pandas DataFrames.")

    # Retain Ids of the training set in the validation set
    validation_df = validation_df[validation_df.farmId.isin(training_df.farmId)]
    validation_df = validation_df[validation_df.periodId.isin(training_df.periodId)].reset_index(drop=True)
    # Map datetimes
    periodId = training_df.periodId.unique()
    # Dictionary where the keys are the elements from periodId, and the values are the indices of those elements in the periodId
    datetime2id_mapping = dict(zip(periodId, range(len(periodId))))
    # Map the values in the periodId column of a pandas DataFrame using a mapping provided by datetime2id_mapping
    training_df.periodId = training_df.periodId.map(datetime2id_mapping)
    validation_df.periodId = validation_df.periodId.map(datetime2id_mapping)
    id2datetime_mapping = {v: k for k, v in datetime2id_mapping.items()}
    # Map farms
    farmId = training_df.farmId.unique()
    # Dictionary where the keys are the elements from farmId, and the values are the indices of those elements in the farmId
    farm2id_mapping = dict(zip(farmId, range(len(farmId))))
    # Map the values in the farmId column of a pandas DataFrame using a mapping provided by farm2id_mapping
    training_df.farmId = training_df.farmId.map(farm2id_mapping)
    validation_df.farmId = validation_df.farmId.map(farm2id_mapping)
    id2farm_mapping = {v: k for k, v in farm2id_mapping.items()}
    return training_df, validation_df, id2datetime_mapping, id2farm_mapping


def filter_data_by_common_periods_farms(training_df, validation_df):
    """
    Filter training and validation data to have common periods and farms.
    """

    if not all(isinstance(df, pd.DataFrame) for df in [training_df, validation_df]):
        raise TypeError("Both training_df and validation_df must be pandas DataFrames.")

    # Check column existence
    required_columns = ['periodId', 'farmId']
    for df, df_name in zip([training_df, validation_df], ['training_df', 'validation_df']):
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{df_name} is missing columns: {', '.join(missing_columns)}.")

    # Filter common periods
    common_periods = set(validation_df['periodId']).intersection(training_df['periodId'])
    filtered_training_df = training_df[training_df['periodId'].isin(common_periods)]
    filtered_validation_df = validation_df[validation_df['periodId'].isin(common_periods)]

    # Filter common farms
    common_farms = set(filtered_validation_df['farmId']).intersection(filtered_training_df['farmId'])
    filtered_training_df = filtered_training_df[filtered_training_df['farmId'].isin(common_farms)]
    filtered_validation_df = filtered_validation_df[filtered_validation_df['farmId'].isin(common_farms)]

    # Check for consistency
    if len(filtered_training_df['periodId'].unique()) != len(filtered_validation_df['periodId'].unique()) \
            or len(filtered_training_df['farmId'].unique()) != len(filtered_validation_df['farmId'].unique()):
        raise ValueError("Number of unique periodIds and farmIds in training and validation data must match.")

    return filtered_training_df, filtered_validation_df


def sqrt_transformation(df):
    " Apply square root transformation to power values. "

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."

    for col in list(df['farmId'].unique()):
        mask = df['farmId'] == col
        df.loc[mask, 'power_z'] = np.sqrt(df.loc[mask, 'power_z'])
    return df



def create_insample_df(train_periods, train_farms, train_targets, train_predictions, lst_farms):
    " Create DataFrame for insample data. "

    assert isinstance(train_periods, list), "train_periods must be a list."
    assert isinstance(train_farms, list), "train_farms must be a list."
    assert isinstance(train_targets, list), "train_targets must be a list."
    assert isinstance(train_predictions, list), "train_predictions must be a list."
    assert isinstance(lst_farms, list), "lst_farms must be a list."

    # Create DataFrame from input lists
    df_insample = pd.DataFrame({
        'periodId': train_periods,
        'farmId': train_farms,
        'targets': train_targets,
        'predictions': train_predictions
    })
    df_insample = df_insample.set_index('periodId')
    # Filter DataFrame by selected farm IDs
    df_insample = df_insample[df_insample['farmId'].isin(lst_farms)]
    # Calculate residuals
    df_insample['residuals'] = df_insample['targets'] - df_insample['predictions']
    return df_insample


def create_outsample_df(test_periods, test_farms, test_targets, test_predictions, lst_farms):
    " Create DataFrame for outsample data. "

    assert isinstance(test_periods, list), "test_periods must be a list."
    assert isinstance(test_farms, list), "test_farms must be a list."
    assert isinstance(test_targets, list), "test_targets must be a list."
    assert isinstance(test_predictions, list), "test_predictions must be a list."
    assert isinstance(lst_farms, list), "lst_farms must be a list."

    # Create DataFrame from input lists
    df_outsample = pd.DataFrame({
        'periodId': test_periods,
        'farmId': test_farms,
        'targets': test_targets,
        'predictions': test_predictions
    })
    # Set periodId as index
    df_outsample = df_outsample.set_index('periodId')
    # Filter DataFrame by selected farm IDs
    df_outsample = df_outsample[df_outsample['farmId'].isin(lst_farms)]
    return df_outsample
