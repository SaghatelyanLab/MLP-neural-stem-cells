import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(datapath: str):
    # Load the data
    train_data = pd.read_csv(f'{datapath}/training_less_params_updated.csv').dropna()
    test_data = pd.read_csv(f'{datapath}/test_less_params.csv').dropna()

    print(f"Train dataset size: {train_data.shape}")
    print(f"Test dataset size: {test_data.shape}")
    
    train_data = train_data.drop(columns=["Path"])
    test_data = test_data.drop(columns=["Path"])
    # Separate features and target
    X = train_data.drop(columns=['Source'])
    y = train_data['Source']
    
    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    test_data_scaled = scaler.transform(test_data.drop(columns=['Source']))  # Assuming test data has a Source column too

    # Convert scaled data back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
    test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns[:-1])  # Exclude 'Source' column

    return (X_train_scaled, y_train, X_val_scaled, y_val, test_data_scaled)

