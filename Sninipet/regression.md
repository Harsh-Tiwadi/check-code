linear regression:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)  # Feature
y = 2 * X.flatten() + 1 + np.random.normal(0, 2, 100)  # Target with noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression Fit')
plt.legend()
plt.title('Linear Regression')
plt.show()

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
```

Ridge regression:

```python
from sklearn.linear_model import Ridge

# Ridge Regression
ridge_model = Ridge(alpha=1.0)  # Adjust alpha
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)

plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression Fit')
plt.plot(X_test, ridge_y_pred, color='green', label='Ridge Regression Fit')
plt.legend()
plt.title('Ridge Regression')
plt.show()

print(f"Ridge Intercept: {ridge_model.intercept_}")
print(f"Ridge Coefficient: {ridge_model.coef_[0]}")
```

Lasso regression:

```python
from sklearn.linear_model import Lasso

# Lasso Regression
lasso_model = Lasso(alpha=0.1)  # Adjust alpha
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)

plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression Fit')
plt.plot(X_test, lasso_y_pred, color='blue', label='Lasso Regression Fit')
plt.legend()
plt.title('Lasso Regression')
plt.show()

print(f"Lasso Intercept: {lasso_model.intercept_}")
print(f"Lasso Coefficient: {lasso_model.coef_[0]}")
```

Elastic-Net regression:

```python
from sklearn.linear_model import ElasticNet

# Elastic Net Regression
elastic_net_model = ElasticNet(alpha=0.5, l1_ratio=0.5)  # Adjust alpha and l1_ratio
elastic_net_model.fit(X_train, y_train)
elastic_net_y_pred = elastic_net_model.predict(X_test)

plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression Fit')
plt.plot(X_test, elastic_net_y_pred, color='magenta', label='Elastic Net Regression Fit')
plt.legend()
plt.title('Elastic Net Regression')
plt.show()

print(f"Elastic Net Intercept: {elastic_net_model.intercept_}")
print(f"Elastic Net Coefficient: {elastic_net_model.coef_[0]}")
```

Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor

# Decision Tree Regression with Parameters
tree_model = DecisionTreeRegressor(
    criterion='mse',  # Function to measure the quality of a split (default: mean squared error)
    splitter='best',  # Strategy for choosing the best split (default: considers all features)
    max_depth=3,      # Maximum depth of the tree (controls complexity, default: None)
    min_samples_split=2,  # Minimum number of samples required to split a node (default: 2)
    min_samples_leaf=1,   # Minimum number of samples allowed in a leaf node (default: 1)
    min_weight_fraction_leaf=0.0,  # Minimum weighted fraction of the leaf (to avoid empty leafs, default: 0.0)
    max_features=None,   # Number of features to consider at each split (default: all features)
    random_state=None   # Seed for random number generation (for reproducibility, default: None)
)

# Train the model
tree_model.fit(X_train, y_train)

# Make predictions
tree_y_pred = tree_model.predict(X_test)

# Visualization
import matplotlib.pyplot as plt

plt.scatter(X, y, label='Data')
plt.plot(X_test, tree_y_pred, color='black', label='Decision Tree Regression Fit')
plt.legend()
plt.title('Decision Tree Regression')
plt.show()
```

Random Forest:

```python
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Random Forest Regression with Parameters
rf_model = RandomForestRegressor(
    n_estimators=100,       # Number of trees in the forest (default: 100)
    criterion='mse',      # Function to measure the quality of a split (default: 'mse')
    max_depth=None,       # Maximum depth of the tree (default: None)
    min_samples_split=2,  # Minimum number of samples required to split an internal node (default: 2)
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node (default: 1)
    min_weight_fraction_leaf=0.0,  # Minimum weighted fraction of the sum total of weights (of all the input samples) to be at a leaf node. Samples have equal weight when sample_weight is not provided.
    max_features='auto',  # Number of features to consider when looking for the best split (default: 'auto')
    max_leaf_nodes=None,  # Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    min_impurity_decrease=0.0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    bootstrap=True,       # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    oob_score=False,      # Whether to use out-of-bag samples to estimate the R^2 on unseen data.
    n_jobs=None,          # The number of jobs to run in parallel. -1 means using all processors.
    random_state=None,    # Seed for random number generation (for reproducibility)
    verbose=0,            # Controls the verbosity when fitting and predicting.
    warm_start=False,     # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
    ccp_alpha=0.0,        # Complexity parameter used for Minimal Cost-Complexity Pruning.
    max_samples=None,    # If bootstrap is True, the number of samples to draw from X to train each base estimator.
)

# Train the model
rf_model.fit(X, y)

# Make predictions
rf_y_pred = rf_model.predict(X_test)

# Visualization
plt.scatter(X, y, label='Data')
plt.plot(X_test, rf_y_pred, color='black', label='Random Forest Regression Fit')
plt.legend()
plt.title('Random Forest Regression')
plt.show()
```

Gradient Boosting:

```python
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np

# Gradient Boosting Regression with Parameters
gb_model = GradientBoostingRegressor(
    loss='ls',            # How we measure how wrong the model is. 'ls' (least squares) is most common for regression. Other options for different data types.
    learning_rate=0.1,    # How much each new tree corrects the mistakes of the previous ones. Smaller values mean slower learning but often better results. (Range: 0.001 - 1.0, Start low)
    n_estimators=100,      # How many trees are built. More trees can improve accuracy but take longer to train. (Range: 50 - 500+, Start around 100)
    subsample=1.0,        # What fraction of the training data is used to train each tree. Less than 1.0 makes it "stochastic" (random) which can prevent overfitting. (Range: 0.5 - 1.0, Start high)
    criterion='friedman_mse', # How we decide the best way to split the data in each tree. 'friedman_mse' is usually a good default.
    min_samples_split=2,  # The minimum number of samples a node needs to have before it can be split. Prevents making very specific rules based on tiny amounts of data. (Range: 2 - 10+, Start low)
    min_samples_leaf=1,   # The minimum number of samples a leaf (end node) can have. Also helps prevent overfitting. (Range: 1 - 5+, Start low)
    min_weight_fraction_leaf=0., # Similar to min_samples_leaf, but uses fractions of the total sample weights. Rarely needed unless you have weighted data.
    max_depth=3,          # The maximum depth of each tree. Limits how complex the rules can be. (Range: 3 - 10, Start around 3-5)
    min_impurity_decrease=0.,# A node will be split if this split reduces impurity by more than this value. helps prevent overfitting.
    init=None,            # An initial prediction to start the boosting process. Usually not needed.
    random_state=None,    # A random seed for reproducibility. Set a number if you want the same results every time you run the code.
    max_features=None,    # How many features are considered at each split. Limits the search space and can prevent overfitting. (Range: 1 - n_features or "sqrt", "log2", Start with "sqrt" or None)
    alpha=0.9,            # Only used if loss='huber'. It controls how much the model is sensitive to outliers.
    verbose=0,            # How much information is printed during training.
    max_leaf_nodes=None,  # Limits the total number of leaves in each tree, another way to control complexity.
    warm_start=False,     # If True, you can train more trees incrementally.
    validation_fraction=None, # The fraction of training data to use for early stopping.
    n_iter_no_change=None, # Number of iterations with no improvement to wait before stopping training early.
    tol=0.0001,           # Tolerance for early stopping.
)

# Train the model
gb_model.fit(X, y)

# Make predictions
gb_y_pred = gb_model.predict(X_test)

# Visualization
plt.scatter(X, y, label='Data')
plt.plot(X_test, gb_y_pred, color='black', label='Gradient Boosting Regression Fit')
plt.legend()
plt.title('Gradient Boosting Regression')
plt.show()
```

XGBoost:

```

```
