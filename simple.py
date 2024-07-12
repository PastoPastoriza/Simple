import pandas as pd
import numpy as np

def flatten(df, window):
  """
  Le das un df y te crea uno flatten con los rows con su lag de WINDOW
  """
  dfs = [df]

  for i in range(1, window + 1):
      lagged_df = df.shift(i).add_suffix(f'_{i}')
      dfs.append(lagged_df)
  result_df = pd.concat(dfs, axis=1)
  return result_df


from itertools import combinations

def create_binary_features(data, features):
    """
    Create a binary DataFrame by comparing every possible combination of features.

    Parameters:
    - data: pd.DataFrame, the input DataFrame containing the features.
    - features: list, list of feature names to be compared.

    Returns:
    - binary_df: pd.DataFrame, the DataFrame containing binary features.
    """
    binary_df = pd.DataFrame(index=data.index)

    # Iterate through all combinations of features
    for (feat1, feat2) in combinations(features, 2):
        # Create binary features for comparisons
        binary_df[f'{feat1}_gt_{feat2}'] = (data[feat1] > data[feat2]).astype(int)

    return binary_df
