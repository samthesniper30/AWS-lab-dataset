import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Provided data
data = {
    "YearsExperience": [1.1, 1.3, 1.5, 2, 2.2, 2.9, 3, 3.2, 3.2, 3.7, 3.9, 4, 4, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6, 6.8, 7.1, 7.9, 8.2, 8.7, 9, 9.5, 9.6, 10.3, 10.5],
    "Salary": [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]
}
df = pd.DataFrame(data)

# Set visual style
sns.set(style="whitegrid")

# 1. Scatter Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x='YearsExperience', y='Salary', data=df)
plt.title("Experience vs. Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# 2. Histogram of Years of Experience
plt.figure(figsize=(6, 4))
sns.histplot(df['YearsExperience'], kde=True, bins=10)
plt.title("Distribution of Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Frequency")
plt.show()

# 3. Histogram of Salaries
plt.figure(figsize=(6, 4))
sns.histplot(df['Salary'], kde=True, bins=10)
plt.title("Distribution of Salaries")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# 4. Boxplot of Salary
plt.figure(figsize=(6, 4))
sns.boxplot(y='Salary', data=df)
plt.title("Salary Boxplot")
plt.ylabel("Salary")
plt.show()

# 5. Regression Line Plot
plt.figure(figsize=(6, 4))
sns.regplot(x='YearsExperience', y='Salary', data=df, ci=None, line_kws={"color": "red"})
plt.title("Linear Regression: Experience vs. Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
