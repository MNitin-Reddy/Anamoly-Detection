# Anomaly Detection

We will implement the anomaly detection algorithm and apply it to detect failing servers on a network.

## Table of Contents
1. [Packages](#packages)
2. [Anomaly Detection](#anomaly-detection)
   - [Problem Statement](#problem-statement)
   - [Dataset](#dataset)
   - [Gaussian Distribution](#gaussian-distribution)
   -  [High Dimensional Dataset](#high-dimensional-dataset)

## 1. Packages

Ensure you have the required libraries installed to run the code.
```bash
pip install numpy matplotlib
```

## 2. Anomaly Detection

### 2.1 Problem Statement

The task involves implementing an anomaly detection algorithm to detect anomalous behavior in server computers. The dataset contains two features:

- Throughput (mb/s)
- Latency (ms)

The dataset consists of 307 examples representing the servers' behavior, with the majority of examples being "normal" while some may represent anomalies. The goal is to use a Gaussian distribution model to detect these anomalies.

### 2.2 Dataset

We begin by loading the dataset. The dataset is split into the following:
```python
def load_data():
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    return X, X_val, y_val

X_train, X_val, y_val = load_data()
```

- **X_train**: Used to fit a Gaussian distribution.
- **X_val, y_val**: Used as a cross-validation set to determine a threshold for anomaly detection.

### 2.3 Gaussian Distribution

To perform anomaly detection, the algorithm needs to fit a Gaussian distribution to the data's features and then calculate the probability of each example under this distribution. The Gaussian distribution is characterized by two parameters: mean ($\mu$) and variance ($\sigma^2$).

For each feature in the dataset, the algorithm estimates the mean and variance, which are then used to evaluate the probability density of each data point.
```python
def estimate_gaussian(X): 
    m, n = X.shape 
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var
```

### Visualize the Data

Before analyzing further, it's important to visualize the data using a scatter plot that shows the relationship between Throughput and Latency.
```python
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', marker='x')
plt.xlabel('Throughput (mb/s)')
plt.ylabel('Latency (ms)')
plt.title('Server Behavior Visualization')
plt.show()
```

### Multivariate Gaussian

To evaluate the probability density function, the algorithm uses the multivariate Gaussian distribution, which calculates the probability for each data point based on the estimated mean and variance.
```
def multivariate_gaussian(X, mu, var):
    k = len(mu)
    if var.ndim == 1:
        var = np.diag(var)
    X = X - mu
    p = (2 * np.pi) ** (-k / 2) * np.linalg.det(var) ** (-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    return p
```

### Selecting the Threshold ($\epsilon$)

The threshold for detecting anomalies is chosen based on cross-validation. The goal is to find the best threshold using the $F_1$ score, which balances precision and recall. The algorithm evaluates various thresholds and selects the one that maximizes the $F_1$ score.
```python
def select_threshold(y_val, p_val): 
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon).astype(int)
        tp = np.sum((predictions == 1) & (y_val == 1))  # True positives
        fp = np.sum((predictions == 1) & (y_val == 0))  # False positives
        fn = np.sum((predictions == 0) & (y_val == 1))  # False negatives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            F1 = (2 * precision * recall) / (precision + recall)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
```

## 2.4 High Dimensional Dataset

Next, we apply the anomaly detection algorithm to a more realistic dataset with 11 features, capturing more properties of the servers. This step involves:

- Estimating Gaussian parameters for the new dataset.
- Evaluating probabilities for both the training and validation sets.
- Using the cross-validation set to find the best threshold.
```python
X_train_high, X_val_high, y_val_high = load_data_high()

mu_high, var_high = estimate_gaussian(X_train_high)
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e' % epsilon_high)
print('Best F1 on Cross Validation Set: %f' % F1_high)
print('# Anomalies found: %d' % sum(p_high < epsilon_high))
```
