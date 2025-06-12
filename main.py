import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold


data = pd.read_csv('17.csv', header=None) 

X = data.iloc[:, :-1]  # features 
y = data.iloc[:, -1]   # target


def first_look_dataset():
    print("Dataset Shape:", data.shape)  # (70, 1626)
    print("Number of Features:", X.shape[1])  # 1625 features
    print("Class Distribution:\n", y.value_counts())  # check class balance
    print("Missing Values:", data.isnull().sum().sum())  # check for missing values (in this case, there are none)
    print("-----")

    print(X.describe()) # mean, std, min, max for each feature

first_look_dataset()

def exploratory_data_analysis():
    """
    Highly correlated features may be redundant, as they provide similar information. 
    Removing one feature from such pairs can reduce dimensionality without significant information loss. 
    This is critical given the high feature-to-sample ratio (1625 features vs. 69 samples), which risks overfitting.
    The code below is doing basically that by applying Pearson correlation to check if its > 0.95
    """
    corr_matrix = pd.DataFrame(X).corr().abs()

    # Seleciona upper triangle da matriz
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Encontra colunas com correlação acima de 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Dropa as colunas
    X_dropped = pd.DataFrame(X).drop(columns=to_drop)

    
    print(f"Original features shape: {X.shape} | After Pearson correlation {X_dropped.shape}")
    # After doing Pearson correlation, the features dataset went from a shape of (70, 1625) to (70, 27)
    
    return X_dropped

exploratory_data_analysis()

