# House-Price-Prediction-using-Machine-Learning-in-Python

📌 Introduction

Predicting house prices accurately is crucial for buyers, sellers, and real estate professionals. This project leverages machine learning algorithms to estimate house prices based on features like lot size, building type, and construction year. It highlights how data-driven models can enhance accuracy and support real estate decision-making.


💡 Problem Statement

-> House prices are influenced by multiple factors such as location, size, and age of the property.

-> Traditional pricing methods can be subjective and inconsistent.

-> A machine learning-based solution can bring in objectivity, consistency, and higher accuracy in predictions.



🎯 Objectives

-> Build a machine learning model to predict housing prices based on historical data.

-> Apply preprocessing, feature selection, and evaluation techniques.

-> Compare multiple algorithms (Linear Regression, SVR, Random Forest) to identify the best-performing model.

-> Evaluate model performance using metrics like R² Score and Mean Absolute Percentage Error (MAPE).



📊 Dataset

The dataset contains various features like:

1 Id	- To count the records.

2	MSSubClass	-  Identifies the type of dwelling involved in the sale.

3	MSZoning	- Identifies the general zoning classification of the sale.

4	LotArea	-  Lot size in square feet.

5	LotConfig	- Configuration of the lot

6	BldgType	- Type of dwelling

7	OverallCond	- Rates the overall condition of the house

8	YearBuilt	- Original construction year

9	YearRemodAdd	- Remodel date (same as construction date if no remodeling or additions).

10	Exterior1st	- Exterior covering on house

11	BsmtFinSF2	- Type 2 finished square feet.

12	TotalBsmtSF	- Total square feet of basement area

13	SalePrice	- To be predicted



🧰 Libraries Used

Pandas: For data loading, manipulation, and handling missing values.

NumPy: For efficient numerical operations.

Matplotlib: For visualizing data and trends.

Seaborn: For correlation heatmaps and advanced plots.

Scikit-learn: For preprocessing, model training, and evaluation.



🔍 Machine Learning Models Used

-> Linear Regression

-> Support Vector Regression (SVR)

-> Random Forest Regressor



📏 Evaluation Metrics

-> R² Score

-> Mean Absolute Percentage Error (MAPE)




🚀 How to Run

Clone this repository.

Install required packages:

pip install -r requirements.txt
