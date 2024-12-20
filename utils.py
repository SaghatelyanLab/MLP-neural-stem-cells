import numpy as np
from tqdm import tqdm


def random_classification(y: np.ndarray, num_classes: int, num_simulations: int) -> np.ndarray:
    uniques = np.unique(y)
    assert len(uniques) == num_classes, "Number of classes does not match the number of unique values in y"
    N = y.shape[0]
    accuracy_distribution = np.zeros(num_simulations)
    for i in tqdm(range(num_simulations)):
        preds = np.random.choice(uniques, size=N)
        accuracy_distribution[i] = (np.sum(preds == y) / N)
    np.save(f"./results/random_accuracy_distribution_{num_classes}classes.npy", accuracy_distribution)
    return accuracy_distribution

def get_classes(num_classes: int):
    """
    Generates a list of class names based on the specified number of classes.

    Parameters:
    num_classes (int): The number of classes to generate names for. 
                       Valid values are between 2 and 6.

    Returns:
    list: A list of class names corresponding to the number of classes.

    Raises:
    ValueError: If num_classes is not between 2 and 6.
    """
    if num_classes == 2:
        return ["EGFR", "Blood_vessel"]
    elif num_classes == 3:
        return ["EGFR", "Blood_vessel", "DCX"]
    elif num_classes == 4:
        return ["EGFR", "Blood_vessel", "DCX", "Ki67"]
    elif num_classes == 5:
        return ["EGFR", "Blood_vessel", "DCX", "Ki67", "GFAP"]
    elif num_classes == 6:
        return ["EGFR", "Blood_vessel", "DCX", "Ki67", "GFAP", "Iba1"]
    else:
        raise ValueError(f"Invalid number of classes: {num_classes}")
