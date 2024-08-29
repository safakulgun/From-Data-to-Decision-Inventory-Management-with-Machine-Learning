# From-Data-to-Decision-Inventory-Management-with-Machine-Learning
From Data to Decision: Inventory Management with Machine Learning

## Library

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTETomek

inventory_df = pd.read_excel('/Users/munzurulgun/Downloads/09_Inventory (1).xlsx')

print(inventory_df.head())

<img width="494" alt="Screenshot 2024-08-27 at 17 20 07" src="https://github.com/user-attachments/assets/fce423bd-199e-4773-a0d9-e19282fe694f">

print(inventory_df.shape)

(7853, 8)

print(inventory_df.isna().sum())

<img width="288" alt="Screenshot 2024-08-27 at 17 24 04" src="https://github.com/user-attachments/assets/e423d18b-de6c-400e-ba2f-1f0d87df8f39">

inventory_df.info()

<img width="433" alt="Screenshot 2024-08-27 at 17 26 55" src="https://github.com/user-attachments/assets/4363e128-6b58-4332-9e4e-349ede875d73">

#Converting categorical variables into factors


inventory_df['Ship Mode'] = inventory_df['Ship Mode'].astype('category')

inventory_df['Product Container'] = inventory_df['Product Container'].astype('category')

inventory_df['Product Sub-Category'] = inventory_df['Product Sub-Category'].astype('category')

#Dropping irrelevant variables

inventory_df.drop(columns=['Order Date', 'Order ID', 'Product Name'], inplace=True)

print(inventory_df.shape)

(7853, 5)

inventory_df.info()

<img width="433" alt="Screenshot 2024-08-27 at 17 42 17" src="https://github.com/user-attachments/assets/e29299e6-9376-4eb7-8c09-ebce9068a2d6">

#Check for outliers

plt.figure(figsize=(10, 5))

sns.boxplot(data=inventory_df[['Order Quantity', 'Sales']], palette="Set2")

plt.show()

![Figure_1](https://github.com/user-attachments/assets/76a807f5-824a-4236-8deb-92c3cdaed30f)

#Outlier Treatment

def outlier_capping(x):

    qnt = np.percentile(x, [25, 75])
    
    caps = np.percentile(x, [5, 95])
    
    
    iqr = qnt[1] - qnt[0]
    
    lower_bound = qnt[0] - 1.5 * iqr
    
    upper_bound = qnt[1] + 1.5 * iqr

    
    # Capping Outliers
    
    x = np.where(x < lower_bound, caps[0], x)
    x = np.where(x > upper_bound, caps[1], x)

    return x

inventory_df['Sales'] = outlier_capping(inventory_df['Sales'])

plt.boxplot(inventory_df['Sales'], patch_artist=True, boxprops=dict(facecolor="orange"))


plt.title("Sales Column After Outlier Capping")

plt.show()    
![Figure_2](https://github.com/user-attachments/assets/97dea136-14f7-4694-8b1f-3ee6c8150507)

#Univariate - Histograms

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.hist(inventory_df['Order Quantity'], bins=20, color='red', edgecolor='black')

plt.title('Histogram of Order Quantity')

plt.subplot(1, 2, 2)

plt.hist(inventory_df['Sales'], bins=20, color='red', edgecolor='black')

plt.title('Histogram of Sales')

plt.show()



![Figure_3](https://github.com/user-attachments/assets/cd930b59-5263-4608-9e22-b4ad314217af)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
inventory_df['Product Container'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Product Container')

plt.subplot(1, 3, 2)
inventory_df['Product Sub-Category'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Product Sub-Category')

plt.subplot(1, 3, 3)
inventory_df['Ship Mode'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Ship Mode')
plt.show()

![Figure_4](https://github.com/user-attachments/assets/40b65ab7-9786-4492-b725-1ca6be17984e)

#Fixing Spaces and Other Characters in Domain Names

inventory_df.columns = inventory_df.columns.str.replace(' ', '.').str.replace(',', '').str.replace('-', '')

#Converting the 'Ship Mode' Column to Numeric

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

inventory_df['Ship.Mode'] = le.fit_transform(inventory_df['Ship.Mode'])

inventory_df['Ship.Mode']

# Analyzing the Class Proportions in the 'Ship Mode' Column

class_counts = inventory_df['Ship.Mode'].value_counts(normalize=True)

print(class_counts)

out:

2    0.747358

0    0.135362

1    0.117280

#%74.74: Regular Air , %13.54: Delivery Truck , %11.73: Express Air

class_counts.plot(kind='bar', title='Class Distribution in Ship Mode')

plt.show()

![Figure_5](https://github.com/user-attachments/assets/1931eb10-2ce6-495c-8a05-42c0998613bd)

# Comment: There is a significant imbalance in the Ship.Mode column. Class 2 has far more examples compared to the other two classes. This imbalance is a factor that needs to be considered during modeling. For instance, due to this imbalance, your model might predict class 2 more heavily and potentially neglect the other classes. 

#We will use the SMOTE technique to help the model learn all classes more effectively
