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

```python
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np



# XGBoost Regression with Parameters
xgbr = xgb.XGBRegressor(
    objective='reg:squarederror',  # The learning objective. 'reg:squarederror' is standard for regression.
    n_estimators=100,              # The number of boosting rounds (trees). (Range: 50-500+, Start around 100)
    learning_rate=0.1,            # The step size shrinkage used to prevent overfitting. (Range: 0.001-1.0, Start low)
    max_depth=3,                  # Maximum depth of a tree. Controls complexity. (Range: 3-10, Start around 3-6)
    min_child_weight=1,           # Minimum sum of instance weight (hessian) needed in a child. Higher values prevent overfitting. (Range: 1-10+, Start low)
    gamma=0,                      # Minimum loss reduction required to make a further partition on a leaf node. (Range: 0-1+, Start at 0)
    subsample=1,                  # Subsample ratio of the training instance. Similar to subsample in GradientBoosting. (Range: 0.6-1.0, Start high)
    colsample_bytree=1,           # Subsample ratio of columns when constructing each tree. (Range: 0.6-1.0, Start around 0.8-1)
    colsample_bylevel=1,          # Subsample ratio of columns for each level.
    reg_alpha=0,                  # L1 regularization term on weights. Increasing this value will make model more conservative.
    reg_lambda=1,                 # L2 regularization term on weights. Increasing this value will make model more conservative.
    random_state=None,            # Random seed for reproducibility.
    n_jobs=-1,                    # Use all available cores.
)

# Train the model
xgbr.fit(X, y)

# Make predictions
xgb_y_pred = xgbr.predict(X_test)

# Visualization
plt.scatter(X, y, label='Data')
plt.plot(X_test, xgb_y_pred, color='black', label='XGBoost Regression Fit')
plt.legend()
plt.title('XGBoost Regression')
plt.show()
```

DT CV:

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Example with Ridge Regression using cross-validation
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv = RidgeCV(alphas=alphas, cv=5) # cv = cross validation fold
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.alpha_
print(f"Best alpha for Ridge: {best_alpha}")

# Similar approach for LassoCV and ElasticNetCV
lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(X_train, y_train)
best_alpha_lasso = lasso_cv.alpha_
print(f"Best alpha for Lasso: {best_alpha_lasso}")

elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5)
elastic_cv.fit(X_train, y_train)
best_alpha_elastic = elastic_cv.alpha_
best_l1_ratio_elastic = elastic_cv.l1_ratio_
print(f"Best alpha for Elastic: {best_alpha_elastic}")
print(f"Best l1_ratio for Elastic: {best_l1_ratio_elastic}")
```

Grid search:

```python
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_regression, make_classification

# Regression
X_reg, y_reg = make_regression(n_samples=100, n_features=5, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

param_grid_tree_reg = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree_reg = DecisionTreeRegressor(random_state=42)
grid_search_tree_reg = GridSearchCV(tree_reg, param_grid_tree_reg, cv=5, scoring='neg_mean_squared_error')
grid_search_tree_reg.fit(X_train_reg, y_train_reg)

print("Best parameters for Tree Regressor:", grid_search_tree_reg.best_params_)

# Classification
X_clf, y_clf = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

param_grid_tree_clf = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'] # For classifier
}

tree_clf = DecisionTreeClassifier(random_state=42)
grid_search_tree_clf = GridSearchCV(tree_clf, param_grid_tree_clf, cv=5, scoring='accuracy')
grid_search_tree_clf.fit(X_train_clf, y_train_clf)

print("Best parameters for Tree Classifier:", grid_search_tree_clf.best_params_)
```

RF CV:

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

param_grid_rf_reg = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_reg = RandomForestRegressor(random_state=42)
grid_search_rf_reg = GridSearchCV(rf_reg, param_grid_rf_reg, cv=5, scoring='neg_mean_squared_error')
grid_search_rf_reg.fit(X_train_reg, y_train_reg)

print("Best parameters for Random Forest Regressor:", grid_search_rf_reg.best_params_)

param_grid_rf_clf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

rf_clf = RandomForestClassifier(random_state=42)
grid_search_rf_clf = GridSearchCV(rf_clf, param_grid_rf_clf, cv=5, scoring='accuracy')
grid_search_rf_clf.fit(X_train_clf, y_train_clf)

print("Best parameters for Random Forest Classifier:", grid_search_rf_clf.best_params_)
```

GB CV:

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

param_grid_gb_reg = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

gb_reg = GradientBoostingRegressor(random_state=42)
grid_search_gb_reg = GridSearchCV(gb_reg, param_grid_gb_reg, cv=5, scoring='neg_mean_squared_error')
grid_search_gb_reg.fit(X_train_reg, y_train_reg)

print("Best parameters for Gradient Boosting Regressor:", grid_search_gb_reg.best_params_)

param_grid_gb_clf = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

gb_clf = GradientBoostingClassifier(random_state=42)
grid_search_gb_clf = GridSearchCV(gb_clf, param_grid_gb_clf, cv=5, scoring='accuracy')
grid_search_gb_clf.fit(X_train_clf, y_train_clf)

print("Best parameters for Gradient Boosting Classifier:", grid_search_gb_clf.best_params_)
```

XGB CV:

```python
import xgboost as xgb

param_grid_xgb_reg = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'reg_alpha':[0,1],
    'reg_lambda':[1,2]
}

xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search_xgb_reg = GridSearchCV(xgbr, param_grid_xgb_reg, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb_reg.fit(X_train_reg, y_train_reg)

print("Best parameters for XGBoost Regressor:", grid_search_xgb_reg.best_params_)

param_grid_xgb_clf = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'reg_alpha':[0,1],
    'reg_lambda':[1,2]
}

xgbc = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
grid_search_xgb_clf = GridSearchCV(xgbc, param_grid_xgb_clf, cv=5, scoring='accuracy')
grid_search_xgb_clf.fit(X_train_clf, y_train_clf)

print("Best parameters for XGBoost Classifier:", grid_search_xgb_clf.best_params_)
```
