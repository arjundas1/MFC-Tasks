# 1. Importing all the necessary libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# 2. Reading the csv file
data = pd.read_csv("case_time_series.csv")
print(data.head())

# 3. A Kernel Density Estimation graph of daily corona cases
sns.set()
vis1 = sns.kdeplot(data["Daily Confirmed"], shade=True, color="maroon")
plt.xlabel("Daily confirmed cases")
plt.ylabel("Number of people")
plt.title("Daily Corona Cases")
plt.show()

# 4. The increasing trend of total confirmed cases of corona
total = np.array(data['Total Confirmed']).reshape(-1,1)
plt.plot(total, '-m')
plt.show()

# 5. A pairplot that helps us visualise the relationship of each pair of dataset column
vis2 = sns.pairplot(data)
sns.show()

# 6. Using Linear Regression algorithm to get the accuracy score of for the given dataset.
y = np.array(data['Total Confirmed']).reshape(-1,1)
x = np.linspace(1, 544, num=544).reshape(-1,1)
polyFeat = PolynomialFeatures(degree=4)
x = polyFeat.fit_transform(x)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'Accuracy: {round(accuracy*100,3)} %')
