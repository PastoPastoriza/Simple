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
