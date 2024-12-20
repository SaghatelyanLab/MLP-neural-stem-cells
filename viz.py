import numpy as np
import matplotlib.pyplot as plt 
import argparse
from scipy.stats import sem

def load_results(path: str):
    results = {}
    for key in [str(item) for item in range(2, 7)]:
        data = np.load(f"{path}/{key}classes-results.npz")
        accuracies, p_value, random_acc = data["accuracies"], data["p_value"], data["random_accuracies"]
        results[key] = {
            "accuracies": accuracies,
            "p_value": p_value,
            "random_acc": random_acc
        }
    return results


def plot_results(results: dict):
    keys = list(results.keys())
    accuracies = [results[key]["accuracies"] for key in keys]
    accuracies = [np.mean(item) for item in accuracies]
    random_accuracies = [results[key]["random_acc"] for key in keys]
    random_accuracies = [np.mean(item) for item in random_accuracies]
    acc_diff = [item1 - item2 for item1, item2 in zip(accuracies, random_accuracies)]
    p_values = [results[key]["p_value"] for key in keys]
    for k, p in zip(keys, p_values):
        print(f"Number of classes: {k}, p-value: {p}")

    fig = plt.figure()
    plt.bar(keys, acc_diff)
    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy vs. random")
    fig.savefig("accuracy_vs_random.png", bbox_inches="tight")

if __name__=="__main__":
    results = load_results(path="./results")
    plot_results(results=results)