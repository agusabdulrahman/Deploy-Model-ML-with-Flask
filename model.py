import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LinearRegression


# dataset
X = np.array([20, 25, 30, 35, 40]).reshape(-1, 1)
Y = np.array([100, 120, 150, 180, 200])

# call model regresion
model = LinearRegression().fit(X, Y)

# save model
filename = 'model.sav'
dump(model, filename)

# Load model
loaded_model = load(filename)

# Predict
prediction = loaded_model.predict(np.array([[100]]))
print(prediction)
