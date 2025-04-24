import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_excel("HousePricePrediction.xlsx")
print(dataset.head(5))
print(dataset.shape)

# Separate columns by data type
object_cols = list(dataset.select_dtypes(include=['object']).columns)
num_cols = list(dataset.select_dtypes(include=['int64']).columns)
fl_cols = list(dataset.select_dtypes(include=['float64']).columns)
print("Categorical variables:", len(object_cols))
print("Integer variables:", len(num_cols))
print("Float variables:", len(fl_cols))

# Correlation heatmap for numerical features
numerical_dataset = dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)

# Unique values in categorical features
unique_values = [dataset[col].nunique() for col in object_cols]
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)

# Distribution of categorical features
rows = (len(object_cols) + 3) // 4
plt.figure(figsize=(18, 4 * rows))
plt.title('Categorical Features: Distribution')
index = 1
for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(rows, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
plt.tight_layout()
plt.show()

# Drop ID and handle missing values
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset = dataset.dropna()
print(new_dataset.isnull().sum())

# One-hot encoding for categorical features
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', len(object_cols))

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Train-test split
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Support Vector Regression
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred = model_SVR.predict(X_valid)
print("SVR MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred))

# Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
print("Random Forest MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred))

# Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
print("Linear Regression MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred))
