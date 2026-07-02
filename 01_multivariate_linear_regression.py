#!/usr/bin/env python

# Multivariate linear regression model that can predict the product sales 
# based on the advertising budget allocated to different channels. i
# The features are TV Budget ($), Radio Budget ($), Newspaper Budget ($) and the output is Sales (units)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Inputs (features): TV, Radio, Newspaper budgets
#Output (target): Sales
# Dataset
data = pd.DataFrame({
    'TV':[230.1,44.5,17.2,151.5,180.8,8.7,57.5,120.2,144.1,111.6],
    'Radio':[37.8,39.3,45.9,41.3,10.8,48.9,32.8,19.6,16.0,12.6],
    'Newspaper':[69.2,45.1,69.3,58.5,58.4,75.0,23.5,11.6,40.3,37.9],
    'Sales':[22.1,10.4,9.3,18.5,12.9,7.2,11.8,13.2,15.6,12.2]
})

X = data[['TV','Radio','Newspaper']]
y = data['Sales']

model = LinearRegression()
model.fit(X,y)

# Predictions
y_pred = model.predict(X)

# Plot
plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Predicted vs Actual Sales")

# Line y=x
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R2 Score:", model.score(X,y))

