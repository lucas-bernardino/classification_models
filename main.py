import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('09.csv', header=None) 

features = data.iloc[:, :-1]  # features 
target = data.iloc[:, -1]   # target


def first_look_dataset(X, y):
    print("Dataset Shape:", data.shape)  # (105, 183)
    print("Number of Features:", X.shape[1])  # 182 features
    print("Class Distribution:\n", y.value_counts())  # check class balance
    print("Missing Values:", data.isnull().sum().sum())  # check for missing values (in this case, there are none)
    print("-----")

    print(X.describe()) # mean, std, min, max for each feature

first_look_dataset(features, target)

def pearson_correlation_filtering(X):
    """
    Highly correlated features may be redundant, as they provide similar information. 
    Removing one feature from such pairs can reduce dimensionality without significant information loss. 
    The code below is doing basically that by applying Pearson correlation to check if its > 0.95
    """
    corr_matrix = pd.DataFrame(X).corr().abs()

    # select upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find columns with correlation higher than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # drop these columns
    X_dropped = pd.DataFrame(X).drop(columns=to_drop)

    
    print(f"Original features shape: {X.shape} | After Pearson correlation {X_dropped.shape}")
    # After doing Pearson correlation, the features dataset went from a shape of (105, 182) to (105, 115)
    
    return X_dropped

def visualize_pca(X):
    pca = PCA()
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Explained Variance vs Number of Components')
    plt.grid(True)
    plt.show()

def apply_pca(X):
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    
    
    return X_pca

def standardization(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def train_and_evaluate_with_kfold(X, y, model_type='knn', n_splits=5, random_state=42):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = []

    if model_type == 'knn':
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean']}
        model = KNeighborsClassifier()
    elif model_type == 'dt':
        param_grid = {'max_depth': [3, 5, None], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
        model = DecisionTreeClassifier(random_state=random_state)
    # elif model_type == 'xgb':
    #     param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1], 'subsample': [0.6, 0.8]}
    #     model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)

    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y_encoded if model_type == 'xgb' else y)

    # Cross-validation scores
    cv_scores = grid_search.cv_results_['mean_test_score']
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    print(f"{model_type.upper()} - Best F1-score (CV): {best_score:.3f} with params {best_params}")

    return grid_search

X_correlation = pearson_correlation_filtering(features)
X_pca = apply_pca(features)

X_standardized_correlation = standardization(X_correlation)
X_standardized_pca = standardization(X_pca)

print("Results for training with no pre processing")
knn_scaled = train_and_evaluate_with_kfold(features, target, 'knn')
dt_scaled = train_and_evaluate_with_kfold(features, target, 'dt')

print("\nResults for X_correlation: ")
knn_scaled = train_and_evaluate_with_kfold(X_correlation, target, 'knn')
dt_scaled = train_and_evaluate_with_kfold(X_correlation, target, 'dt')
# xgb_scaled = train_and_evaluate_with_kfold(X_correlation, target, 'xgb')


print("\nResults for X_standardized_correlation: ")
knn_scaled = train_and_evaluate_with_kfold(X_standardized_correlation, target, 'knn')
dt_scaled = train_and_evaluate_with_kfold(X_standardized_correlation, target, 'dt')
# xgb_scaled = train_and_evaluate_with_kfold(X_standardized_correlation, target, 'xgb')


print("\nResults for X_pca: ")
knn_pca = train_and_evaluate_with_kfold(X_pca, target, 'knn')
dt_pca = train_and_evaluate_with_kfold(X_pca, target, 'dt')
# xgb_pca = train_and_evaluate_with_kfold(X_pca, target, 'xgb')

print("\nResults for X_standardized_pca: ")
knn_pca = train_and_evaluate_with_kfold(X_standardized_pca, target, 'knn')
dt_pca = train_and_evaluate_with_kfold(X_standardized_pca, target, 'dt')
# xgb_pca = train_and_evaluate_with_kfold(X_standardized_pca, target, 'xgb')
