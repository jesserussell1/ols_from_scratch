# How to calculate an ordinary least squares (OLS) regression from scratch
## Jesse Russell
### September 30, 2024

#### For documentation on the dataset, [see here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).

# Load packages
# Get the California housing dataset
from sklearn.datasets import fetch_california_housing

# Use Pandas to manage the data
import pandas as pd

# Numbers
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
housing = fetch_california_housing()

# Get features from the housing dataset
X = pd.DataFrame(housing.data, columns=housing.feature_names)

# Get the target, average house value in units of 100,000
y = pd.Series(housing.target)

# We'll use only the 'AveRooms' feature (average number of rooms)
X_rm = X['AveRooms']

# Assumptions Check
# Linearity

# Scatter plot to check for linearity
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_rm, y=y, alpha=0.1)
plt.title('Scatter Plot of X_rm vs y')
plt.xlabel('X_rm')
plt.ylabel('y')
plt.show()

# 🏡 It looks like there are some very large X_rm values
### Who has a house with 140 rooms?
### Let's drop those and replot


# Calculate the threshold for the top 1% of X_rm values
threshold = np.percentile(X_rm, 99)

# Filter out the top 1% of X_rm values
filtered_indices = X_rm <= threshold
X_rm_filtered = X_rm[filtered_indices]

# And get the y values just for the filtered X values
y_filtered = y[filtered_indices]

# Plot scatter plot to check for linearity after dropping outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_rm_filtered, y=y_filtered, alpha=0.1)
plt.title('Scatter Plot of X_rm vs y (Top 1% Outliers Removed)')
plt.xlabel('X_rm')
plt.ylabel('y')
plt.show()

# Note that y values are capped at 5.0


# Calculate the Mean Values

# Calculate the means of X and y
mean_x = X_rm_filtered.mean()
mean_y = y_filtered.mean()

# Print the means as a check
print(f"Mean of X (RM): {mean_x}")
print(f"Mean of y (Price): {mean_y}")

# Calculate Deviations from the Means for X and y

# X deviations from the X mean and Y deviations from the Y mean
x_dev = X_rm_filtered - mean_x
y_dev = y_filtered - mean_y


# Multiply X and Y Deviations

# Multiply the two pandas series (element-wise)
xy_dev_product = x_dev * y_dev

# As a check, print the first few results
print(xy_dev_product.head())

# Square the X Deviations

# Each element of the X deviations is squared
x_dev_squared = x_dev ** 2

 # As a check, print the first few results
print(x_dev_squared.head())

# Sum the product of deviations and the squared X deviations
# When you use the sum() method on a pandas series, it calculates the sum of all the elements in the series

# Calculate the sums
sum_xy_dev_product = sum(xy_dev_product)
sum_x_dev_squared = sum(x_dev_squared)

print(f"Sum of X*Y deviations: {sum_xy_dev_product}")
print(f"Sum of X deviations squared: {sum_x_dev_squared}")

# Calculate the slope (beta_1)

slope = sum_xy_dev_product / sum_x_dev_squared

print(f"Slope (Coefficient): {slope}")

# Calculate the intercept (beta_0)

intercept = mean_y - slope * mean_x
print(f"Intercept: {intercept}")



# Predict y values based on the calculated regression equation

# Use the calculated intercept, slope, and X values
y_pred = intercept + slope * X_rm_filtered

# Print the first few rows
print(y_pred.head())


# Compute the Total Sum of Squares (SST)
# This is the sum of the squared differences between the actual values
# and the mean of the dependent variable

sst = np.sum((y_filtered - mean_y) ** 2)

# Calculate the Residual Sum of Squares (SSE)
# This is the sum of the squared differences between the actual values
# and the predicted values

sse = np.sum((y_filtered - y_pred) ** 2)

# Calculate R-squared

# Use sse and sst
r_squared = 1 - (sse / sst)

# Print the result
print(f"R-squared: {r_squared}")
