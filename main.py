from os import remove
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# import warnings
# warnings.filterwarnings("ignore")


# Load data
FILE_NAME = "17.csv"
data = pd.read_csv(FILE_NAME, header=None)
features = data.iloc[1:, :-1]
target = data.iloc[1:, -1]

### EDA
def first_look_dataset(X, y):
    print("File: ", FILE_NAME)
    print("Dataset Shape:", X.shape)
    print("Class Distribution:", y.value_counts())
    print("-----")
    
    # visualize class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig("class_distribution.png", dpi=300)
    plt.close()

### BASELINE
def dummy_baseline(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    dummy = DummyClassifier(random_state=42)
    scores = cross_val_score(dummy, X, y_enc, cv=5, scoring="f1_weighted")
    preds = cross_val_predict(dummy, X, y_enc, cv=5)
    print(f"Baseline (DummyClassifier): F1-weighted = {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"DummyClassifier - Classification Report:")
    print(classification_report(y_enc, preds, zero_division=0))

### PRE PROCESSING
def remove_outliers(X, y):
    # contamination is the proportion of outliers in the data set
    iso = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso.fit_predict(X)
    X_clean = X[outlier_labels == 1]
    y_clean = y[outlier_labels == 1]
    
    print(f"[Remove Outliers] Before: {X.shape} | After : {X_clean.shape}")
    return X_clean, y_clean

def balance_classes(X, y):
    """
    if some classes have way more samples than others, 
    it leads to a bias toward the majority class, and causes the model to ignore the minority class.
    
    SMOTE is used to balance imbalanced datasets by artificially generating new samples of the minority class.
    """
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"[Balance Classes] Before: {X.shape} | After: {X_balanced.shape}")
    
    return X_balanced, y_balanced

def visualize_pearson_correlation(X):
    """
    highly correlated features may be redundant, as they provide similar information. 
    removing one feature from such pairs can reduce dimensionality without significant information loss. 
    the code below is doing basically that by applying Pearson correlation with the threshold
    """
    corr_matrix = pd.DataFrame(X).corr().abs() # computes the absolute Pearson correlation
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title("Feature correlation")
    plt.savefig("feature_correlation.png")
    plt.close()
    
def pearson_correlation_filtering(X, threshold):
    corr_matrix = pd.DataFrame(X).corr().abs() # computes the absolute Pearson correlation
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # get only the upper triangle of the matrix to avoid duplicates
    features_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)] # get the features correlated
    X_dropped = pd.DataFrame(X).drop(columns=features_correlated) # remove the correlated features, leaving the others as it is.
    print(f"[Pearson correlation] Before: {X.shape} | After: {X_dropped.shape}")
    
    return X_dropped

def visualize_pca(X, threshold):
    """
    reduces the number of dimensions in large datasets to principal components that retain most of the original information.
    it does this by transforming potentially correlated variables into a smaller set of variables, called principal components.
    """
    
    # if n_components is < 1, it'll reduce dimensionality by retaining the value in percetage of the variance.
    # so if n_components is 0.95, PCA selects the smallest number of components such that the cumulative explained variance ratio is at least 0.95 (95%)
    pca = PCA(n_components=threshold)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.xlabel('Components')
    plt.ylabel('Variance')
    plt.title('PCA Visualization')
    plt.grid(True)
    plt.savefig("pca_variance.png")
    plt.close()
    
    # return number of components for 95% variance
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    pca = PCA(n_components=threshold)
    X_pca = pca.fit_transform(X)
    print(f"[PCA] Before: {X.shape} | After : {X_pca.shape}")
    
    return n_components

### TRAINING AND EVALUATING
def train_and_evaluate_pipeline(X, y, model_type, pca_components=None):
    #TODO: Explain
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    #TODO: Explain
    min_class = pd.Series(y_encoded).value_counts().min()
    n_splits = min(5, max(2, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    steps = [('scaler', StandardScaler())]
    if pca_components:
        steps.append(('pca', PCA(n_components=pca_components)))

    if model_type == 'knn':
        steps.append(('clf', KNeighborsClassifier()))
        grid = {'clf__n_neighbors': [3, 5, 7], 'clf__weights': ['uniform', 'distance']}
    elif model_type == 'dt':
        steps.append(('clf', DecisionTreeClassifier(random_state=42)))
        grid = {'clf__max_depth': [3, 5, None], 'clf__criterion': ['gini', 'entropy']}
    elif model_type == 'svm':
        steps.append(('clf', SVC(probability=True, random_state=42)))
        grid = {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf'], 'clf__gamma': ['scale', 0.1]}
    
    pipe = Pipeline(steps)
    search = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
    search.fit(X, y_encoded)
    
    scores = cross_val_score(search.best_estimator_, X, y_encoded, cv=cv, scoring="f1_weighted")
    print(f"{model_type.upper()} F1-weighted: {scores.mean():.3f} +/- {scores.std():.3f}")

    y_pred = cross_val_predict(search.best_estimator_, X, y_encoded, cv=cv)
    print(f"{model_type.upper()} Classification Report:")
    print(classification_report(y_encoded, y_pred, target_names=[str(c) for c in encoder.classes_]))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f"{model_type.upper()} Confusion Matrix")
    plt.savefig(f"{model_type}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # CV scores boxplot
    scores = cross_val_score(search.best_estimator_, X, y_encoded, cv=cv, scoring='f1_weighted')
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=scores)
    plt.title(f"{model_type.upper()} Cross-Validation Weighted F1-Scores")
    plt.savefig(f"{model_type}_cv_scores.png", dpi=300)
    plt.close()
    
    return search, encoder, scores

# run EDA
first_look_dataset(features, target)

# test dataset with dummy classifier
dummy_baseline(features, target)

print("---")

# preprocessing
features, target = remove_outliers(features, target)
features, target = balance_classes(features, target)

n_components = visualize_pca(features, 0.95)
visualize_pearson_correlation(features)
X_corr = pearson_correlation_filtering(pd.DataFrame(features), 0.9)

print("\n  RUNNING WITH ORIGINAL VALUES  \n")
for model_name in ['knn', 'dt', 'svm']:
    train_and_evaluate_pipeline(features, target, model_name, pca_components=None)

print("\n  RUNNING WITH PCA  \n")
for model_name in ['knn', 'dt', 'svm']:
    train_and_evaluate_pipeline(features, target, model_name, pca_components=n_components)

print("\n  RUNNING WITH CORRELATION \n")
for model_name in ['knn', 'dt', 'svm']:
    train_and_evaluate_pipeline(features, target, model_name, pca_components=None)