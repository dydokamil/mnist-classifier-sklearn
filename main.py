import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mnist = fetch_mldata('MNIST original')
X = mnist['data']
y = mnist['target']

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier()),
])

param_grid = [
    dict(classifier__n_estimators=[30, 60, 90, 150, 300, 600, 900],
         classifier__max_features=['auto', 'sqrt', 'log2'],
         classifier__n_jobs=[-1]),
]

grid_search = GridSearchCV(pipeline, param_grid=param_grid)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.score(X_test, y_test))
