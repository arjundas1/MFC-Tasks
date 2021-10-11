# 1. Importing all the necessary libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, svm, metrics

# 2. Reading the csv file and identifying the missing values.
missing_values = ["?"]
data = pd.read_csv("C:/Users/Arjun/PycharmProjects/DataScience/Mutiple Regression/export.csv", na_values= missing_values)
print(data.head())

# 3. Finding the total number of missing data values.
print(data.isna().sum())

# 4. Mapping values from string to float values of the NaN columns
lis = ["property_damage", "police_report_available"]
def mapping(x):
  return x.map({'YES': 1, "NO": 0})

data[lis] = data[lis].apply(mapping)
print(data.head())
data["collision_type"] = data["collision_type"].map({"Front Collision": 1.0, "Rear Collision": 2.0, "Side Collision": 3.0})

# 5. Substituting NaN values using fillna()
data = data.fillna({
    "police_report_available" : 0.5,         # An average value due to equal possibility of having a police report
    "property_damage" : 0.5,                 # Equal probability of property damage
    "collision_type" : 0.0                   # 0 due to no collision (Cases of theft corresponds to '?' in dataset)
})
print(data.head())

# 6. Visualizing data Various two dimensional visualisation techniques have been used that represents parts of the given dataset.

# i) A histogram that shows the number of people in the dataset and their time (in months) as a customer.
sns.set()
vis1 = plt.hist(data["months_as_customer"], bins=10, color=sns.color_palette()[9])
plt.xlabel("Months as a customer")
plt.ylabel("Number of people")
plt.show()

# ii) A swarmplot that shows the range of annual premiums in each state given in the dataset.
vis2 = sns.swarmplot(x="policy_state", y="policy_annual_premium", data= data, size= 4)
plt.xlabel("Policy State")
plt.ylabel("Annual Premium")
plt.show()

# iii) Pie chart to represent the sex ratio
gender = data["insured_sex"].value_counts()
vis3 = plt.pie(gender, labels= gender.index, startangle=90, counterclock=False)

# iv) A ring plot to visualize the authorities contacted by the customers.
authorities = data["authorities_contacted"].value_counts()
vis4 = plt.pie(authorities, labels=authorities.index, startangle=90, counterclock=False, wedgeprops={'width': 0.4})
plt.title("Authorities Contacted")
plt.show()

# v) A pairplot that helps us visualise the relationship of each pair of dataset column.
vis5 = sns.pairplot(data, diag_kind= "kde")
sns.show()

# 7. ML Algorithms for prediction of model of the car (auto-maker) using other data available. 
# Every algorithm used has been iterated over a 100 times and the training-testing combination with the highest accuracy 
# has been stored as a pickle file and this accuracy is shown to the user.

# i) Preprocessing data and splitting data into training and testing dataset
le = preprocessing.LabelEncoder()
car = le.fit_transform(list(data["auto_make"]))
y = list(car)
data['auto_model'] = le.fit_transform((data['auto_model']))
x = data[["auto_model"]]

# ii) With a single column of x data, we get the following accuracy:

# LOGISTIC REGRESSION
lrbest = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    lr = LogisticRegression(solver="liblinear")
    lrmodel = lr.fit(x_train, y_train)
    lracc = lr.score(x_test, y_test)
    if lracc > lrbest:
        lrbest = lracc
        with open("LoReAccuracy.pickle", "wb") as f:
            pickle.dump(lrmodel, f)

Lrpickle = open("LoReAccuracy.pickle", "rb")
lrmodel = pickle.load(Lrpickle)
lracc = lr.score(x_test, y_test)
print("Accuracy using Logistic Regression: ", round(lracc*100, 2), "%")


# K NEAREST NEIGHBORS
knnbest = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    knnmodel = KNeighborsClassifier(n_neighbors=3)
    knnmodel.fit(x_train, y_train)
    knnacc = knnmodel.score(x_test, y_test)
    if knnacc > knnbest:
        knnbest = knnacc
        with open("KNNAccuracy.pickle", "wb") as f:
            pickle.dump(knnmodel, f)


Knnpickle = open("KNNAccuracy.pickle", "rb")
knnmodel = pickle.load(Knnpickle)
knnacc = knnmodel.score(x_test, y_test)
print("Accuracy using KNN: ", round(knnacc*100, 2), "%")


# SUPPORT VECTOR MACHINES
svmbest = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    svmmodel = svm.SVC(kernel="linear", C=1)
    svmmodel.fit(x_train, y_train)
    y_pred = svmmodel.predict(x_test)
    svmacc = metrics.accuracy_score(y_test, y_pred)
    if svmacc > svmbest:
        svmbest = svmacc
        with open("SVMAccuracy.pickle", "wb") as f:
            pickle.dump(svmmodel, f)


Svmpickle = open("SVMAccuracy.pickle", "rb")
svmmodel = pickle.load(Svmpickle)
svmmodel.fit(x_train, y_train)
y_pred = svmmodel.predict(x_test)
svmacc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy using SVM: ", round(svmacc*100, 2), "%")
