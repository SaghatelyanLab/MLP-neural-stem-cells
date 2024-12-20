# MLP-cell-stem-cell-pipeline

## Installation 
1. Install [https://www.python.org/downloads/](Python)
2. Clone the repository  
   ```
   git clone https://github.com/fredbeaupre/MLP-cell-stem-cell.git
   ```
3. Move into root
   ```
   cd MLP-cell-stem-cell
   ```
4. Create virtual environment
   ```
   python -m venv VENV
   source VENV/bin/activate
   ``` 
5. Install requirements
   ```
   pip install -r requirements.txt
   ```

# Scripts
#### Running grid search on the MLP hyper-parameters.
Run the main script with the --grid-search argument  
```
python main.py --grid-search
``` 

#### Train the MLP 
The default arguments to the main script will run 100 000 simulations of the random classifier, train the MLP with optimal hyper-parameters for 1000 epochs 50 times and save the results in the `results/` folder, for the 6-class classification scenario. It also assumes that the data can be found in the `data/` folder.  
To run the training with a different number of classes, you can specify the `--num-classes` argument, e.g:  
```
python main.py --num-classes 2
```   

Adding the `--shap` argument will run the feature importance analysis using shap values at the end of the script and save the results in the `results/` folder. Note that this is nearly as computationally  expensive as the MLP training.

Other arguments are described inside the script.

#### Reproduce figures from the obtained results
```
python viz.py
```

#### Option to use models other than MLP will be pushed to this repo soon.


