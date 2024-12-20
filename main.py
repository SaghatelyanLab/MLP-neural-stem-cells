from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from datasets import load_and_prepare_data
import argparse 
import os 
import numpy as np
import utils
from sklearn.metrics import accuracy_score
from scipy import stats
from typing import Union
import shap
from tqdm import trange
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--grid-search", action="store_true", help="Run grid search on the MLP hyper-parameters")
parser.add_argument("--datapath", type=str, default="./data", help="Path to the data")
parser.add_argument("--num-repetitions", type=int, default=50, help="Number of training repetitions")
parser.add_argument("--num-epochs", type=int, default=1000, help="Number of epochs per repetition")
parser.add_argument("--num-simulations", type=int, default=100_000, help="Number of simulations of the random classifier")
parser.add_argument("--num-classes", type=int, default=6, help="Number of classes")
parser.add_argument("--shap", action="store_true", help="Analyze feature importance")
args = parser.parse_args()

def load_optimized_model():
    return MLPClassifier(activation="tanh", alpha=0.0001, hidden_layer_sizes=(64,), solver="sgd", learning_rate_init=0.001, learning_rate="constant", max_iter=1000)

def load_random_distribution(labels: np.ndarray, num_classes: int, num_simulations: int):
    if not os.path.exists(f"./results/random_accuracy_distribution_{num_classes}classes.npy"):
        random_accuracies = utils.random_classification(labels, num_classes, num_simulations) 
    else:
        random_accuracies = np.load(f"./results/random_accuracy_distribution_{num_classes}classes.npy")
    return random_accuracies

def compute_statistics(accuracies: np.ndarray, query: Union[float, np.ndarray]):
    if isinstance(query, float):
        query = np.array([query])
    print(accuracies.shape, query.shape)
    merged_acc = np.r_[accuracies, query]
    z_scores = stats.zscore(merged_acc)
    z = z_scores[-1]
    p_value = 1 - stats.norm.cdf(z)
    return p_value

def analyze_feature_importance(model, X_train, X_test, c: int):
    """Analyze feature importance using SHAP values."""
    # Initialize the SHAP explainer
    classname = utils.get_classes(args.num_classes)[c]
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    
    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)  # Using first 100 test samples for efficiency
    print(shap_values.shape)
    
    # Plot the SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, c], X_test, show=False)
    plt.tight_layout()
    plt.savefig(f'./results/{args.num_classes}classes-{classname}-feature_importance.png')
    plt.close()
    
    # Return the mean absolute SHAP values for each feature
    mean_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
    return mean_shap

# Load the data
if __name__ == "__main__":

    if args.grid_search:
        classes = utils.get_classes(args.num_classes)
        X_train, y_train, X_val, y_val = load_and_prepare_data(datapath=args.datapath, validation=True, classes=classes)
    # Define the MLP model
        mlp = MLPClassifier(max_iter=1000, verbose=1)

        # Define the parameter grid for hyper-parameter tuning
        param_grid = {
            'hidden_layer_sizes': [(16,), (32,), (64,), (32, 32), (16, 16)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }

        # Set up the grid search
        grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy')

        # Fit the model using grid search
        grid_search.fit(X_train, y_train)

        # Print the best parameters and best score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    
    else:
        classes = utils.get_classes(args.num_classes)
        accuracies = []
        X_train, y_train, X_test, y_test = load_and_prepare_data(datapath=args.datapath, validation=False, classes=classes)
        print("--- Running/loading random classification results ---")
        random_accuracies = load_random_distribution(labels=y_train, num_classes=args.num_classes, num_simulations=args.num_simulations)
        for i in trange(args.num_repetitions, desc="Training MLP...."):
            mlp = load_optimized_model()
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        print("--- Computing statistics --- ")
        p = compute_statistics(random_accuracies, np.mean(accuracies))
        print(f"====== Results for {args.num_classes} classes ({classes}) ======")
        print(f"\nRandom accuracy: {np.mean(random_accuracies)} ± {np.std(random_accuracies)}")
        print(f"\tMLP Accuracy: {np.mean(accuracies)} ± {np.std(accuracies)}")
        print(f"\tP-value: {p}")
        print("==================================================")
        np.savez(f"./results/{args.num_classes}classes-results", accuracies=accuracies, p_value=p, random_accuracies=random_accuracies)

        if args.shap: 
            print("--- Analyzing feature importance ---")
            for c in range(args.num_classes):
                analyze_feature_importance(mlp, X_train, X_test, c=c)
