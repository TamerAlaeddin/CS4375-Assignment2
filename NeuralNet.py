import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, dataFile, header=None):
        self.raw_input = pd.read_csv(dataFile, header=header)

    def preprocess(self):
        # Handling missing values
        self.processed_data = self.raw_input.dropna()

        # Standardization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = self.processed_data.iloc[:, :-1]
        scaled_features = scaler.fit_transform(features)
        self.processed_data.iloc[:, :-1] = scaled_features

        return self.processed_data

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols - 1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Hyperparameters
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.01, 0.1]
        max_iterations = [300, 500]
        num_hidden_layers = [2, 3]

        results = []
        history = {}

        plot_count = 1
        plt.figure(figsize=(10, 8))
        lines_per_plot = 8
        line_count = 0

        for activation in activations:
            for lr in learning_rates:
                for epochs in max_iterations:
                    for layers in num_hidden_layers:
                        hidden_layer_sizes = (100,) * layers
                        model = MLPClassifier(activation=activation, learning_rate_init=lr, max_iter=epochs,
                                              hidden_layer_sizes=hidden_layer_sizes, early_stopping=True)
                        model.fit(X_train, y_train)
                        train_score = model.score(X_train, y_train)
                        test_score = model.score(X_test, y_test)
                        results.append((activation, lr, epochs, layers, train_score, test_score))

                        label = f'{activation}, lr={lr}, epochs={epochs}, layers={layers}'
                        history[label] = model.loss_curve_

                        if line_count % lines_per_plot == 0 and line_count != 0:
                            plt.xlabel('Epochs')
                            plt.ylabel('Loss')
                            plt.title(f'Model History Part {plot_count}')
                            plt.legend()
                            plt.savefig(f'model_history_part{plot_count}.png')
                            plt.figure(figsize=(10, 8))
                            plot_count += 1

                        plt.plot(model.loss_curve_, label=label)
                        line_count += 1

        # Save the final plot if not already saved
        if line_count % lines_per_plot != 0:
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Model History Part {plot_count}')
            plt.legend()
            plt.savefig(f'model_history_part{plot_count}.png')

        # Output the results in tabular format
        results_df = pd.DataFrame(results, columns=['Activation', 'Learning Rate', 'Epochs', 'Layers', 'Train Score', 'Test Score'])
        print(results_df)

        return results_df

if __name__ == "__main__":
    neural_network = NeuralNet("iris.data")
    neural_network.preprocess()
    results = neural_network.train_evaluate()
    results.to_csv("results.csv", index=False)
