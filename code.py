# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# import numpy as np

# # Provided data
# data = {
#     "YearsExperience": [1.1, 1.3, 1.5, 2, 2.2, 2.9, 3, 3.2, 3.2, 3.7, 3.9, 4, 4, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6, 6.8, 7.1, 7.9, 8.2, 8.7, 9, 9.5, 9.6, 10.3, 10.5],
#     "Salary": [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]
# }
# df = pd.DataFrame(data)

# # Set visual style
# sns.set(style="whitegrid")

# # 1. Scatter Plot
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x='YearsExperience', y='Salary', data=df)
# plt.title("Experience vs. Salary")
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
# plt.show()

# # 2. Histogram of Years of Experience
# plt.figure(figsize=(6, 4))
# sns.histplot(df['YearsExperience'], kde=True, bins=10)
# plt.title("Distribution of Years of Experience")
# plt.xlabel("Years of Experience")
# plt.ylabel("Frequency")
# plt.show()

# # 3. Histogram of Salaries
# plt.figure(figsize=(6, 4))
# sns.histplot(df['Salary'], kde=True, bins=10)
# plt.title("Distribution of Salaries")
# plt.xlabel("Salary")
# plt.ylabel("Frequency")
# plt.show()

# # 4. Boxplot of Salary
# plt.figure(figsize=(6, 4))
# sns.boxplot(y='Salary', data=df)
# plt.title("Salary Boxplot")
# plt.ylabel("Salary")
# plt.show()

# # 5. Regression Line Plot
# plt.figure(figsize=(6, 4))
# sns.regplot(x='YearsExperience', y='Salary', data=df, ci=None, line_kws={"color": "red"})
# plt.title("Linear Regression: Experience vs. Salary")
# plt.xlabel("Years of Experience")
# plt.ylabel("Salary")
# plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("salary_data.csv")  # Make sure to save your data with this name

# Split features and target
X = df[['YearsExperience']]  # Features must be 2D
y = df['Salary']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Print metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Model Coefficient (slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)

# Plotting
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience")
plt.legend()
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("annotations.csv", skiprows=1, header=None)
df.columns = ['label', 'x', 'y', 'w', 'h', 'filename', 'img_w', 'img_h']

# Drop filename column
df = df.drop(columns=['filename'])

# Encode labels to numbers
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Separate features and labels
X = df[['x', 'y', 'w', 'h', 'img_w', 'img_h']]
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

