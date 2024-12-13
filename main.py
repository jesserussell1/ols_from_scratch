# Calculate an ordinary least squares (OLS) regression from scratch
# Jesse Russell
# September 30, 2024

# For documentation on the dataset, [see here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).

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

# Visualize the relationship between rooms and price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_rm, y=y, alpha=0.1, color='orange')
plt.title('Scatter Plot of X_rm vs y')
plt.xlabel('X_rm')
plt.ylabel('y')
plt.show()

# It looks like there are some very large X_rm values (who has a house with 140 rooms?)
# Drop those and replot
# Calculate the threshold for the top 1% of X_rm values
threshold = np.percentile(X_rm, 99)

# Filter out the top 1% of X_rm values
filtered_indices = X_rm <= threshold
X_rm_filtered = X_rm[filtered_indices]

# And get the y values just for the filtered X values
y_filtered = y[filtered_indices]

# Receck the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_rm_filtered, y=y_filtered, alpha=0.1, color='orange')
plt.title('Scatter Plot of X_rm vs y (Top 1% Outliers Removed)')
plt.xlabel('X_rm')
plt.ylabel('y')
plt.show()

# Note that y values are capped at 5.0

# Calculate the means of X and y
mean_x = X_rm_filtered.mean()
mean_y = y_filtered.mean()

# Print the means as a check
print(f"Mean of X (RM): {mean_x}")
print(f"Mean of y (Price): {mean_y}")

# X deviations from the X mean and Y deviations from the Y mean
x_dev = X_rm_filtered - mean_x
y_dev = y_filtered - mean_y

# Multiply X and Y Deviations (element-wise)
xy_dev_product = x_dev * y_dev

# Square the X Deviations
x_dev_squared = x_dev ** 2

# Sum the product of deviations and the squared X deviations
sum_xy_dev_product = sum(xy_dev_product)
sum_x_dev_squared = sum(x_dev_squared)

print(f"Sum of X*Y deviations: {sum_xy_dev_product}")
print(f"Sum of X deviations squared: {sum_x_dev_squared}")

# Calculate the slope from the sum of xy deviations and the sum of the squared x deviations
slope = sum_xy_dev_product / sum_x_dev_squared
print(f"Slope (Coefficient): {slope}")

# Calculate the intercept from the mean of y minus the slope times the mean of x
intercept = mean_y - slope * mean_x
print(f"Intercept: {intercept}")

# Calculate predicted values of y values based on the calculated intercept and slope
y_pred = intercept + slope * X_rm_filtered

# Compute the Total Sum of Squares (SST)
# This is the sum of the squared differences between the actual values and the mean of the dependent variable
sst = np.sum((y_filtered - mean_y) ** 2)

# Calculate the Residual Sum of Squares (SSE)
# This is the sum of the squared differences between the actual values and the predicted values
sse = np.sum((y_filtered - y_pred) ** 2)

# Calculate R-squared
# Use sse and sst calculated above
r_squared = 1 - (sse / sst)

# Print the R-squared result
print(f"R-squared: {r_squared}")
