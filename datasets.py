import pandas 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

def balance_training_set(X: pandas.DataFrame, y: pandas.Series):
    # Combine X and y into a single DataFrame
    df = pandas.concat([X, y], axis=1)
    
    # Find the number of instances in the minority class
    min_class_size = df[y.name].value_counts().min()
    
    # Create a list to hold the resampled DataFrames
    resampled_dfs = []
    
    # Downsample each class to the size of the minority class
    for class_label in df[y.name].unique():
        class_subset = df[df[y.name] == class_label]
        downsampled_class = resample(class_subset, 
                                      replace=False,  # without replacement
                                      n_samples=min_class_size,  # to match minority class
                                      random_state=42)  # reproducible results
        resampled_dfs.append(downsampled_class)
    
    # Concatenate all resampled DataFrames
    balanced_df = pandas.concat(resampled_dfs)
    
    # Separate the features and target again
    X_balanced = balanced_df.drop(columns=[y.name])
    y_balanced = balanced_df[y.name]
    
    return X_balanced, y_balanced

def load_and_prepare_data(datapath: str, validation: bool = False, classes: list = None):
    # Load the data
    train_data = pandas.read_csv(f'{datapath}/training_less_params_updated.csv').dropna()
    test_data = pandas.read_csv(f'{datapath}/test_less_params.csv').dropna()
    
    train_data = train_data.drop(columns=["Path"])
    test_data = test_data.drop(columns=["Path"])

    if classes is not None:
        train_data = train_data[train_data['Source'].isin(classes)]
        test_data = test_data[test_data['Source'].isin(classes)]

    # Separate features and target
    X = train_data.drop(columns=['Source'])
    y = train_data['Source']

    print("-- Dataset before balancing ---")
    unique, counts = np.unique(y, return_counts=True)
    for i, (u, c) in enumerate(zip(unique, counts)):
        print(f"\tClass {u}: {c} samples")
    print("--------------------------------")


    X, y = balance_training_set(X, y)
    
    print("-- Dataset after balancing ---")
    unique, counts = np.unique(y, return_counts=True)
    for i, (u, c) in enumerate(zip(unique, counts)):
        print(f"\tClass {u}: {c} samples")
    print("--------------------------------\n")
    
    x_test, y_test = test_data.drop(columns=['Source']), test_data['Source']
    
    if validation:
    # Split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        test_data_scaled = scaler.transform(test_data.drop(columns=['Source']))  # Assuming test data has a Source column too

        # Convert scaled data back to DataFrame for consistency
        X_train_scaled = pandas.DataFrame(X_train_scaled, columns=X.columns)
        X_val_scaled = pandas.DataFrame(X_val_scaled, columns=X.columns)
        test_data_scaled = pandas.DataFrame(test_data_scaled, columns=test_data.columns[:-1])  # Exclude 'Source' column

        return (X_train_scaled, y_train, X_val_scaled, y_val)
    
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.fit_transform(x_test)
        X_train_scaled = pandas.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pandas.DataFrame(X_test_scaled, columns=x_test.columns)
        return (X_train_scaled, y, X_test_scaled, y_test)

