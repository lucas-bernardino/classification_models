from os import remove
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

# import warnings
# warnings.filterwarnings("ignore")


# Load data
data = pd.read_csv('17.csv', header=None)
features = data.iloc[1:, :-1]
target = data.iloc[1:, -1]

def first_look_dataset(X, y):
    print("Dataset Shape:", data.shape)
    print("Number of Features:", X.shape[1])
    print("Class Distribution:\n", y.value_counts())
    print("Number of Unique Classes:", y.nunique())
    print("Missing Values:", data.isnull().sum().sum())
    print("Feature Data Types:", X.dtypes.unique())
    print("Target Data Type:", y.dtype)
    print("-----")
    print(X.describe())
    
    # Visualize class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig("class_distribution.png", dpi=300)
    plt.close()

def pearson_correlation_filtering(X, threshold):
    # removes redundant features that are very similar correlated > threshold
    
    corr_matrix = pd.DataFrame(X).corr().abs() # computes the absolute Pearson correlation
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # get only the upper triangle of the matrix to avoid duplicates
    features_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    X_dropped = pd.DataFrame(X).drop(columns=features_correlated)
    print(f"Original features shape: {X.shape} | After Pearson correlation: {X_dropped.shape}")
    return X_dropped

def visualize_pca(X, n_components):
    pca = PCA(n_components=0.95)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components')
    plt.grid(True)
    plt.savefig("pca_variance.png", dpi=300)
    plt.close()
    
    # Return number of components for 95% variance
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Components for 95% variance: {n_components}")
    return n_components

def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"Features after PCA: {X_pca.shape[1]}")
    return X_pca, pca

def standardization(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def remove_outliers(X, y):
    iso = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso.fit_predict(X)
    X_clean = X[outlier_labels == 1]
    y_clean = y[outlier_labels == 1]
    return X_clean, y_clean

def balance_classes(X, y):
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    return X, y

def train_and_evaluate_with_kfold(X, y, model_type='knn', n_splits=3, random_state=42):
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Adjust n_splits based on minimum class size
    min_class_size = pd.Series(y_encoded).value_counts().min()
    n_splits = min(n_splits, max(2, min_class_size))
    print(f"Using {n_splits}-fold CV due to class distribution")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    if model_type == 'knn':
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean']}
        model = KNeighborsClassifier()
    elif model_type == 'dt':
        param_grid = {'max_depth': [3, 5, None], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy'], 'class_weight': ['balanced', None]}
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_type == 'svm':
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 0.1], 'class_weight': ['balanced', None]}
        model = SVC(probability=True, random_state=random_state)
    
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X, y_encoded)
    
    # Results
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    print(f"{model_type.upper()} - Best Weighted F1-score (CV): {best_score:.3f} with params {best_params}")
    
    y_pred = grid_search.best_estimator_.predict(X)
    
    # Confusion matrix
    cm = confusion_matrix(y_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{model_type.upper()} Confusion Matrix")
    plt.savefig(f"{model_type}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # CV scores boxplot
    scores = cross_val_score(grid_search.best_estimator_, X, y_encoded, cv=cv, scoring='f1_weighted')
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=scores)
    plt.title(f"{model_type.upper()} Cross-Validation Weighted F1-Scores")
    plt.savefig(f"{model_type}_cv_scores.png", dpi=300)
    plt.close()
    
    return grid_search, le

# Run EDA
first_look_dataset(features, target)

print("\nResults without pre processing")
knn_corr, le_knn = train_and_evaluate_with_kfold(features, target, 'knn')
dt_corr, _ = train_and_evaluate_with_kfold(features, target, 'dt')
svm_corr, _ = train_and_evaluate_with_kfold(features, target, 'svm')

# Preprocessing
features, target = remove_outliers(features, target)
features, target = balance_classes(features, target)

X_corr = pearson_correlation_filtering(features, 0.9)
n_components = visualize_pca(features, 0.95)
X_pca, pca = apply_pca(features, n_components=0.95)
X_std_corr, scaler_corr = standardization(X_corr)
X_std_pca, scaler_pca = standardization(X_pca)

print("\nResults for X_correlation:")
knn_corr, le_knn = train_and_evaluate_with_kfold(X_corr, target, 'knn')
dt_corr, _ = train_and_evaluate_with_kfold(X_corr, target, 'dt')
svm_corr, _ = train_and_evaluate_with_kfold(X_corr, target, 'svm')

print("\nResults for X_pca:")
knn_pca, _ = train_and_evaluate_with_kfold(X_pca, target, 'knn')
dt_pca, _ = train_and_evaluate_with_kfold(X_pca, target, 'dt')
svm_pca, _ = train_and_evaluate_with_kfold(X_pca, target, 'svm')

print("\nResults for X_standardized_correlation:")
knn_corr, le_knn = train_and_evaluate_with_kfold(X_std_corr, target, 'knn')
dt_corr, _ = train_and_evaluate_with_kfold(X_std_corr, target, 'dt')
svm_corr, _ = train_and_evaluate_with_kfold(X_std_corr, target, 'svm')

print("\nResults for X_standardized_pca:")
knn_pca, _ = train_and_evaluate_with_kfold(X_std_pca, target, 'knn')
dt_pca, _ = train_and_evaluate_with_kfold(X_std_pca, target, 'dt')
svm_pca, _ = train_and_evaluate_with_kfold(X_std_pca, target, 'svm')