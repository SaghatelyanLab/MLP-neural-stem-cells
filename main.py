from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from datasets import load_and_prepare_data
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--grid-search", action="store_true")
parser.add_argument("--datapath", type=str, default="./data")
parser.add_argument("--num-classes", type=int, default=6)
parser.add_argument("--num-repetitions", type=int, default=50)
parser.add_argument("--num-epochs", type=int, default=1000)
args = parser.parse_args()

def load_optimized_model():
    return MLPClassifier(activation="relu", alpha=0.0001, hidden_layer_sizes=(200,), solver="sgd", learning_rate="constant", )

# Load the data
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, test_data = load_and_prepare_data(datapath=args.datapath)

    if args.grid_search:
    # Define the MLP model
        mlp = MLPClassifier(max_iter=1000, verbose=1)

        # Define the parameter grid for hyper-parameter tuning
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (200,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }

        # Set up the grid search
        grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=5, scoring='accuracy')

        # Fit the model using grid search
        grid_search.fit(X_train, y_train)

        # Print the best parameters and best score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
