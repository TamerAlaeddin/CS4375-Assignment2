# Neural Network Assignment Report

## Dataset
[Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

## Summary of Results

| Activation | Learning Rate | Epochs | Layers | Train Score | Test Score |
|------------|----------------|--------|--------|-------------|------------|
| logistic   | 0.01           | 300    | 2      | 0.650000    | 0.733333   |
| logistic   | 0.01           | 300    | 3      | 0.816667    | 0.766667   |
| logistic   | 0.01           | 500    | 2      | 0.866667    | 0.900000   |
| logistic   | 0.01           | 500    | 3      | 0.908333    | 0.900000   |
| logistic   | 0.10           | 300    | 2      | 0.666667    | 0.666667   |
| logistic   | 0.10           | 300    | 3      | 0.350000    | 0.266667   |
| logistic   | 0.10           | 500    | 2      | 0.333333    | 0.333333   |
| logistic   | 0.10           | 500    | 3      | 0.350000    | 0.266667   |
| tanh       | 0.01           | 300    | 2      | 0.900000    | 0.900000   |
| tanh       | 0.01           | 300    | 3      | 0.841667    | 0.866667   |
| tanh       | 0.01           | 500    | 2      | 0.966667    | 0.966667   |
| tanh       | 0.01           | 500    | 3      | 0.941667    | 0.933333   |
| tanh       | 0.10           | 300    | 2      | 0.816667    | 0.800000   |
| tanh       | 0.10           | 300    | 3      | 0.875000    | 0.866667   |
| tanh       | 0.10           | 500    | 2      | 0.941667    | 0.933333   |
| tanh       | 0.10           | 500    | 3      | 0.891667    | 0.900000   |
| relu       | 0.01           | 300    | 2      | 0.975000    | 0.966667   |
| relu       | 0.01           | 300    | 3      | 0.950000    | 1.000000   |
| relu       | 0.01           | 500    | 2      | 0.958333    | 1.000000   |
| relu       | 0.01           | 500    | 3      | 0.958333    | 0.966667   |
| relu       | 0.10           | 300    | 2      | 0.925000    | 0.900000   |
| relu       | 0.10           | 300    | 3      | 0.958333    | 0.900000   |
| relu       | 0.10           | 500    | 2      | 0.991667    | 0.966667   |
| relu       | 0.10           | 500    | 3      | 0.850000    | 0.900000   |

## Best Performing Model
The best performing activation function was `relu` with a learning rate of `0.01`, 500 epochs, and 2 layers. This combination achieved a training score of 0.958333 and a test score of 1.000000. The `relu` activation function generally performed better across different configurations, achieving higher accuracy and better convergence.

## Assumptions
1. The dataset is assumed to be clean and well-formatted.
2. Early stopping was used to prevent overfitting.
3. Standardization of features was performed to improve model performance.
