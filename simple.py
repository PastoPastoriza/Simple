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
#############
# from sklearn.svm import SVC,NuSVC,LinearSVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier,StackingClassifier,BaggingClassifier,HistGradientBoostingClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,PassiveAggressiveClassifier,SGDClassifier,ElasticNet,MultiTaskElasticNet,Ridge
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB,BernoulliNB
# from sklearn.neural_network import MLPClassifier, BernoulliRBM
# from sklearn.random_projection import GaussianRandomProjection
# from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
  
  
# models = [
#     ("SVC (balanced)", SVC(class_weight="balanced")),
#     ("SVC", SVC()),
#     # ("NuSVC", NuSVC()),
#     ("Logistic Regression (balanced)", LogisticRegression(class_weight="balanced")),
#     ("Logistic Regression", LogisticRegression()),
#     ("Random Forest (balanced)", RandomForestClassifier(class_weight="balanced")),
#     ("Random Forest", RandomForestClassifier()),
#     ("Gradient Boosting", GradientBoostingClassifier()),
#     ("Extra Trees", ExtraTreesClassifier()),
#     ("AdaBoost", AdaBoostClassifier())
# ]

# all_model_results = pd.DataFrame()

# for name, model in models:
#     np.random.seed(42)
#     model.fit(X_train_array, y_train_array)
#     model_pred = model.predict(X_test_array)

#     # Assuming fun.true_dict returns a dictionary with evaluation metrics
#     print(name)
#     model_results = fun.true_dict(y_test_array, model_pred)
#     all_model_results[name] = model_results

# all_model_results = all_model_results.transpose()

# # Plot all results
# plt.style.use("seaborn-v0_8-whitegrid")
# all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
# plt.title('Model Performance Comparison')
# plt.show()

# # Filter and plot based on precision_1
# filtered_df = all_model_results[all_model_results['precision_1'] >= 0.4]
# filtered_df.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
# plt.title('Filtered Model Performance (Precision >= 0.4)')
# plt.show()

# filtered_df
#######################
