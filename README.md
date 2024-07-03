# CS4375 Assignment 2
## Description
This repository contains the implementation for CS4375 Assignment 2. The assignment involves the implementation of a neural network to analyze the Iris dataset. The project includes preprocessing the dataset, training the neural network with different hyperparameters, and evaluating the performance of each configuration. The code is designed to read the dataset directly from a GitHub URL.

## Neural Network Analysis
### How to Run
1. Ensure you have Python 3 installed.
2. Clone the repository:
 
```bash
git clone https://github.com/TamerAlaeddin/CS4375-Assignment2.git
cd CS4375-Assignment2
```

3. Create a virtual environment and activate it:
 
```bash
python3 -m venv myenv
source myenv/bin/activate
```

4. Install the required libraries:
 
```bash
pip install numpy pandas scikit-learn matplotlib
```

5. Run the script:
 
```bash
python3 NeuralNet.py
```

### Output
The script will read the dataset directly from GitHub, preprocess the data, train the neural network with different hyperparameters, and evaluate the performance. The results will be saved in `results.csv`, and model history plots will be saved as `model_history_part1.png` (and additional parts if necessary).

## Dataset
The dataset used in this assignment is the "Iris" dataset from the UCI ML Repository. It is hosted on GitHub and can be accessed at the following URL:
[Iris Dataset](https://raw.githubusercontent.com/TamerAlaeddin/CS4375-Assignment2/master/iris.data)

## Additional Information
### Pre-processing Steps
1. Removed null or NA values.
2. Standardized the features using `StandardScaler` from `scikit-learn`.

### Training and Test Split
The dataset is split into training and test parts with an 80/20 ratio.

### Hyperparameters
- Activation Functions: `logistic`, `tanh`, `relu`
- Learning Rates: `0.01`, `0.1`
- Epochs: `300`, `500`
- Hidden Layers: `2`, `3`

### Evaluation Metrics
- Training Accuracy
- Test Accuracy

### Logging
The model history (loss vs. epochs) is plotted and saved as image files (`model_history_part1.png`, `model_history_part2.png`, etc.) if necessary.

### Hosting Data on GitHub
The dataset is hosted on GitHub to ensure that the scripts can fetch the data directly without needing a local copy. This makes the scripts more portable and easier to run on any machine.

## Results
The results of the neural network training and evaluation are summarized in `results.csv`. The file contains the following columns:
- Activation Function
- Learning Rate
- Epochs
- Number of Layers
- Training Accuracy
- Test Accuracy

## Best Performing Model
The best performing activation function was `relu` with a learning rate of `0.01`, 500 epochs, and 2 layers. This combination achieved a training score of 0.958333 and a test score of 1.000000. The `relu` activation function generally performed better across different configurations, achieving higher accuracy and better convergence.

## Contributors
- Tamer Alaeddin
- Joseph Saber