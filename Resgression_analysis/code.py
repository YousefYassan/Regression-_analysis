# Life Expectancy Prediction
## About Dataset
### Context
Although there have been lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition and mortality rates. It was found that affect of immunization and human development index was not taken into account in the past. Also, some of the past research was done considering multiple linear regression based on data set of one year for all the countries. Hence, this gives motivation to resolve both the factors stated previously by formulating a regression model based on mixed effects model and multiple linear regression while considering data from a period of 2000 to 2015 for all the countries. Important immunization like Hepatitis B, Polio and Diphtheria will also be considered. In a nutshell, this study will focus on immunization factors, mortality factors, economic factors, social factors and other health related factors as well. Since the observations this dataset are based on different countries, it will be easier for a country to determine the predicting factor which is contributing to lower value of life expectancy. This will help in suggesting a country which area should be given importance in order to efficiently improve the life expectancy of its population.
### Content
The project relies on accuracy of data. The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of the health status as well as many other related factors for all countries The data-sets are made available to public for the purpose of health data analysis. The data-set related to life expectancy, health factors for 193 countries has been collected from the same WHO data repository website and its corresponding economic data was collected from United Nation website. Among all categories of health-related factors only those critical factors were chosen which are more representative. It has been observed that in the past 15 years , there has been a huge development in health sector resulting in improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project we have considered data from year 2000-2015 for 193 countries for further analysis. The individual data files have been merged together into a single data-set. On initial visual inspection of the data showed some missing values. As the data-sets were from WHO, we found no evident errors. Missing data was handled in R software by using Missmap command. The result indicated that most of the missing data was for population, Hepatitis B and GDP. The missing data were from less known countries like Vanuatu, Tonga, Togo, Cabo Verde etc. Finding all data for these countries was difficult and hence, it was decided that we exclude these countries from the final model data-set. The final merged file(final dataset) consists of 22 Columns and 2938 rows which meant 20 predicting variables. All predicting variables was then divided into several broad categories:​Immunization related factors, Mortality factors, Economical factors and Social factors.
### Acknowledgements
The data was collected from WHO and United Nations website with the help of Deeksha Russell and Duan Wang.


### Project Goals 
- We will be exploring the data, cleaning, analyizng, and perfroming regression analysis on our data.
- Our aim is to build a robust model with high R-squared score and low mean square error for predicting life expectency depending our selected features while addressing the model's assumptions.
_________________________________________________________________________________________________________________________________________________________________________
## EDA 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('Life Expectancy Data.csv')
pd.set_option('display.max_columns', None)
df.sample(5)
df.info()
Seems that there is some inconssistency in column naming which we will address.
column_rename_mapping = {
    'Life expectancy ': 'life_expectancy',
    'Measles ': 'measles',
    ' BMI ': 'bmi',
    'under-five deaths ': 'under_five_deaths',
    'Diphtheria ': 'diphtheria',
    'thinness__1-19_years': 'thinness_10_19_years'
}

df.rename(columns=column_rename_mapping, inplace=True)
df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)
df.rename(columns=column_rename_mapping, inplace=True)
df['Status'] = np.where(df['status'] == 'Developing', 0, 1)
df.columns
df.duplicated().sum()
df.isna().sum().to_frame()
df.describe()
There seems to be a large number of nulls in our dataset with some having large standard deviation, so instead of imputing the nulls with an average metric, we will use an advanced technique like `IterativeImputer` from sickit learn and use `RandomForestRegrressor` for estimating/predicting the null values  
columns_with_missing_values = ['alcohol', 'bmi', 'schooling', 'income_composition_of_resources', 
                                'gdp', 'thinness_10_19_years', 'thinness_5-9_years', 'polio', 
                                'diphtheria', 'hepatitis_b', 'population', 'total_expenditure','life_expectancy','adult_mortality']

imputer = IterativeImputer(estimator=RandomForestRegressor(n_jobs=-1, random_state=42))

df_imputed = df.copy()

df_imputed[columns_with_missing_values] = imputer.fit_transform(df_imputed[columns_with_missing_values])

df_imputed.isnull().sum().to_frame()
Next, We see some columns have extreme value that are very far from the 75th percentile which indicates the presence of outliers  
# Checking outlier percentage based on iqr
def calculate_outlier_percentage(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (column < lower_bound) | (column > upper_bound)
    return outliers.mean() * 100

outlier_percentages = df_imputed.select_dtypes(include=['int64', 'float64']).apply(calculate_outlier_percentage)

# Print outlier percentages
print("Outlier Percentages in Each Column:")
print(outlier_percentages)

numerical_columns = df_imputed.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
numerical_columns.boxplot()
plt.title('Box Plot of Numerical Columns (Excluding Population)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
numerical_columns = df_imputed.select_dtypes(include=['int64', 'float64']).drop(columns=['population'])

plt.figure(figsize=(12, 8))
numerical_columns.boxplot()
plt.title('Box Plot of Numerical Columns (Excluding Population)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
Due to the large variation in our data and the existance of some extreme outliers, while some of them may be true outliers, they would impact our models performance and removing these outliers would cause a huge loss of data so we will use `winsorization` to reduce the extreme outliers to achieve more interpretable analysis and better results in our model.

- While this doesn't completly eleminate outliers it reduces their impact. there may be still columns with extreme values, but we will adress this columns while building our model and see how they affect our model.
numerical_columns = df_imputed.select_dtypes(include=['int64', 'float64']).columns

# Function to winsorize outliers
def winsorize_outliers(df, columns, lower_percentile=0.05, upper_percentile=0.95):
    # Calculate lower and upper bounds for winsorization
    lower_bound = df[columns].quantile(lower_percentile)
    upper_bound = df[columns].quantile(upper_percentile)
    
    # Winsorize outliers
    df_winsorized = df.copy()
    for col in columns:
        df_winsorized[col] = np.where(df_winsorized[col] < lower_bound[col], lower_bound[col], df_winsorized[col])
        df_winsorized[col] = np.where(df_winsorized[col] > upper_bound[col], upper_bound[col], df_winsorized[col])
    
    return df_winsorized


df_imputed_win = winsorize_outliers(df_imputed, numerical_columns)
_________________________________________________________________________________________________________________________________________________________________________
## Analysis
plt.figure(figsize=(15,5))
sns.lineplot(x = 'year', y = 'life_expectancy', data = df_imputed, marker = 'o' ,errorbar=('ci', False))
plt.show()
An analysis reveals a significant increase in life expectancy over the years, suggesting a corresponding decline in mortality rates
import plotly.express as px
px.scatter(df_imputed,y = 'adult_mortality',x='life_expectancy',
 color='country',size='life_expectancy',
 template='plotly_dark',opacity=0.6,
 height= 600, width = 800,
 title='<b> Life Expectancy Versus Adult Mortality')
Zimbabwe stands out with the highest adult mortality rate among the listed countries, reporting 723 deaths per 1000 people. Additionally, Zimbabwe also records a lower life
expectancy compared to all other countries, although it's not the lowest among them.
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
Target = ['life_expectancy']
sns.kdeplot(data=df_imputed, x=Target[0], y='measles', ax=axs[0][0])
axs[0][0].set_title('Measles vs. Life Expectancy')
axs[0][0].set_xlabel('Life Expectancy')
axs[0][0].set_ylabel('Measles')
sns.kdeplot(data=df_imputed, x=Target[0], y='hiv/aids', ax=axs[0][1])
axs[0][1].set_title('HIV/AIDS vs. Life Expectancy')
axs[0][1].set_xlabel('Life Expectancy')
axs[0][1].set_ylabel('HIV/AIDS')
sns.kdeplot(data=df_imputed, x=Target[0], y='polio', ax=axs[1][0])
axs[1][0].set_title('Polio vs. Life Expectancy')
axs[1][0].set_xlabel('Life Expectancy')
axs[1][0].set_ylabel('Polio')
sns.kdeplot(data=df_imputed, x=Target[0], y='hepatitis_b', ax=axs[1][1])
axs[1][1].set_title('Hepatitis B vs. Life Expectancy')
axs[1][1].set_xlabel('Life Expectancy')
axs[1][1].set_ylabel('Hepatitis B')
plt.tight_layout()
plt.show()

Based on the analysis of diseases, it has been observed that measles has a relatively minor impact on life expectancy. Conversely, HIV/AIDS demonstrates a strong negative
correlation with life expectancy, indicating that as HIV/AIDS prevalence increases, life expectancy tends to decrease. Additionally, there is a weak correlation observed between
polio, hepatitis B, and life expectancy.
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_imputed, x='status', y='life_expectancy')
plt.title('Comparison of Life Expectancy between Developing and Developed Countries')
plt.xlabel('Status (0: Developing, 1: Developed)')
plt.ylabel('Life Expectancy')
plt.show()
It's clear from the analysis that in developed countries, where people enjoy better living conditions, life expectancy tends to be higher compared to developing nations.
sns.lmplot(x = 'bmi', y = 'life_expectancy', data = df_imputed )
plt.xlabel("BMI")
plt.ylabel("Life expectancy")
plt.title("BMI Vs Life expectancy")
Analyzed data reveals a robust and strong positive correlation between BMI and life expectancy.
sns.histplot(data=df_imputed, x="life_expectancy", kde=True, bins=31)
plt.title('Normalized Life expectancy distribution')
plt.show()
It almost have a normal distribution with negative skew.
df_imputed_win.describe()
_________________________________________________________________________________________________________________________________________________________________________
### Kolmogorov-Smirnov test for distributional differences
#### Significance level 0.05
import pandas as pd
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov test
def perform_ks_test(original_data, winsorized_data):
    # Perform KS test
    ks_statistic, p_value = ks_2samp(original_data, winsorized_data)
    
    return ks_statistic, p_value

# Perform KS test for each numerical column
ks_results = {}
for col in numerical_columns:
    ks_statistic, p_value = perform_ks_test(df_imputed[col], df_imputed_win[col])
    reject_null = p_value < 0.05
    ks_results[col] = {'KS Statistic': ks_statistic, 'P-value': p_value, 'Reject Null Hypothesis': reject_null}

ks_results_df = pd.DataFrame.from_dict(ks_results, orient='index')

ks_results_df

We notice here that most of our data distributions are different from the original data so we will base our analysis on the original imputed data and use the winsorized data for modeling  
df_imputed.head()
_________________________________________________________________________________________________________________________________________________________________________
## Model Building
We will be using different types of multiregression and performing feature selection to achieve a strong robust model while addressing model assumptions. 
### Model Assumption: Independance
The independent observation assumption states that each observation in the dataset is independent. Since data has some observations (but not all) over different years some observations are not independant. The assumption is partially violated. 
### Model Assumption: Linearity
numerical_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('year')

plt.figure(figsize=(20, 15))
for i, column in enumerate(numerical_cols):
    plt.subplot(4, 5, i+1)
    plt.scatter(df_imputed[column], df_imputed['life_expectancy'], alpha=0.5)
    plt.xlabel(column)
    plt.ylabel('Life Expectancy')
    plt.title(f'{column} vs Life Expectancy')

plt.tight_layout()
plt.show()

There is a clear linear relation between life expectany and other columns. So, linearity assumption is met.

### Model Assumption: No Multicollinearity
corr = df_imputed.drop(columns=['status', 'country']).corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr, annot=True)
plt.show()
Features such as `under_five_death` and `infant_deaths`, `gdp` and `percentage_expenditure`, `schooling` and `income_composition_of_resources`, `thinness_10_19_years`and`thinness_5-9_years`, and `polio` and `diphtheria`  are collerated and therefore will not be used together.
Next, we will go ahead and start building our model by first running an algorithm iteratively to determine the features that results in the best score 
### Multilinear Regression
# X = df_imputed.drop(columns=['life_expectancy', 'country', 'status', 'percentage_expenditure','income_composition_of_resources','under_five_deaths','thinness_10_19_years'])  # Exclude the target column
# y = df_imputed['life_expectancy']

# # Initialize variables to store the best features and scores
# best_features = list(X.columns)
# best_score = -np.inf
# best_adj_r_squared = -np.inf

# # Iterate over features and remove one feature at a time
# for i in range(len(X.columns)):
#     # Exclude one feature
#     X_temp = X.drop(columns=[X.columns[i]])
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)
    
#     # Initialize and fit the Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # Predict the target variable on the testing set
#     y_pred = model.predict(X_test)
    
#     # Calculate R-squared
#     r_squared = r2_score(y_test, y_pred)
    
#     # Calculate adjusted R-squared
#     n = len(y_test)
#     p = X_test.shape[1]
#     adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
#     # Check if current model is better than the best model so far
#     if adj_r_squared > best_adj_r_squared:
#         best_adj_r_squared = adj_r_squared
#         best_features = list(X_temp.columns)

# # Print the best features and scores
# print("Best Features:", best_features)
# print("Best Adjusted R-squared:", best_adj_r_squared)

X = df_imputed_win.drop(columns=['life_expectancy','country','status' ,'population','measles','gdp','under_five_deaths','polio'])
y = df_imputed_win['life_expectancy']
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train_linear, y_train_linear)  

y_pred_linear = model.predict(X_test_linear)

mse = mean_squared_error(y_test_linear, y_pred_linear)

n = len(y_test_linear)
p = X_test_linear.shape[1]  
r_squared = r2_score(y_test_linear, y_pred_linear)  
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
print("Adjusted R-squared:", adjusted_r_squared)

### Polynomial Regression
# # Define the predictor variables (features) and target variable
# X = df_imputed.drop(columns=['life_expectancy','country','status' ,'population','measles','gdp'])  # Exclude the target column
# y = df_imputed['life_expectancy']

# # Create polynomial features of degree 2
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X)

# # Initialize lists to store results
# best_features = list(X.columns)
# best_score = -np.inf

# # Iterate over features and remove one feature at a time
# for i in range(len(X_poly[0])):
#     # Exclude one feature
#     X_temp = np.delete(X_poly, i, axis=1)
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)
    
#     # Initialize and fit the Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # Predict the target variable on the testing set
#     y_pred = model.predict(X_test)
    
#     # Calculate R-squared
#     r_squared = r2_score(y_test, y_pred)
    
#     # Calculate adjusted R-squared
#     n = len(y_test)
#     p = X_test.shape[1]
#     adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
#     # Check if current model is better than the best model so far
#     if adjusted_r_squared > best_score:
#         best_score = adjusted_r_squared
#         best_features = [f for j, f in enumerate(X.columns) if j != i]

# # Print the best features
# print("Best Features:", best_features)
# print("Best Adjusted R-squared:", best_score)

X = df_imputed_win.drop(columns=['life_expectancy','country','status' ,'population','measles','gdp','under_five_deaths','polio'])  # Exclude the target column
y = df_imputed_win['life_expectancy']

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train_poly, y_train_poly)

y_pred_poly = model.predict(X_test_poly)

mse = mean_squared_error(y_test_poly, y_pred_poly)

n = len(y_test_poly)
p = X_test_poly.shape[1]
r_squared = r2_score(y_test_poly, y_pred_poly)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
print("Adjusted R-squared:", adjusted_r_squared)

### Model Assumption: Normality
residuals_linear = y_test_linear - y_pred_linear

residuals_poly = y_test_poly - y_pred_poly

plt.figure(figsize=(20, 12))

# Linear Regression Plots
plt.subplot(2, 2, 1)
plt.hist(residuals_linear, bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Residuals (Linear Regression)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
stats.probplot(residuals_linear, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Linear Regression)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

# Polynomial Regression Plots
plt.subplot(2, 2, 3)
plt.hist(residuals_poly, bins=20, color='green', edgecolor='black')
plt.title('Histogram of Residuals (Polynomial Regression)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
stats.probplot(residuals_poly, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Polynomial Regression)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

plt.tight_layout()
plt.show()



We can see that for both the linear regression and polynomial regression the residuals are approximately following linear distributions also supported by their qq plots. Which in this case normality assumption is met.
### Model assumption: Homoscedasticity
# Linear regression
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_linear, residuals_linear, color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values (Linear Regression)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Polynomial regression
plt.subplot(1, 2, 2)
plt.scatter(y_pred_poly, residuals_poly, color='green', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values (Polynomial Regression)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()
We can see that in each model the variance of residuals is random and constant with no obvious patterns indicating that Homoscedasticity is met.
## Lazy Models
Now, let's suppose we took a more traditional approach in cleaning, not decreasing the effect f outliers, and not doing feature selction
df_filled = df.fillna(df.mean())

X = df_filled.drop(columns=['life_expectancy', 'country', 'status'])
y = df_filled['life_expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
print("Adjusted R-squared:", adjusted_r_squared)

X = df_filled.drop(columns=['life_expectancy', 'country', 'status'])
y = df_filled['life_expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test_poly.shape[1] - 1)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
print("Adjusted R-squared:", adjusted_r_squared)
## Lazy Models
### Linear Regression Results:
Mean Squared Error: 15.242936417691073

R-squared: 0.8240562394233331

Adjusted R-squared: 0.8178501103024629

### Polynomial Regression Results:
Mean Squared Error: 9.055163867039644

R-squared: 0.8954794837590591

Adjusted R-squared: 0.8281413360408059


## Chosen Models
### Linear Regression Results:

Mean Squared Error: 9.182071355569573

R-squared: 0.8833824134566243

Adjusted R-squared: 0.880324259963354


### Polynomial Regression Results:
Mean Squared Error: 4.072333757853935

R-squared: 0.9482790194010007

Adjusted R-squared: 0.9328313813902376

- We can see that our "more advanced" techniques of handling missing data and outliers and feature selection caused our models  performance to increase significantly compared to the lazy models in all aspects, as polynomial regression adjusted R-squared increased my more than 10% while the error decreased by a big amount.    
- We can see that polynomial regression model outperformes the linear regression for this dataset.
- Both of our models have met the assumptions while partially violiting the independance assumption
- Polynomial regression explain 94.8% of the values using R squared while 93.3% using adjusted R squared 
- High performance score on our test data indicates the success of our project goals in building an accurate stronge model
- As our performance is high there is no need to try to regualize the model using lasso or ridge regression, and polynomial regression is sufficient for our final model 
## Conclusion
In conclusion, our study aimed to address gaps in previous research on life expectancy by developing a comprehensive regression model using data from 2000 to 2015 for 193
countries. By incorporating critical factors such as immunization, mortality, economy, and social determinants, we sought to provide insights into the complex interplay of factors
nfluencing life expectancy.

Our analysis involved meticulous preprocessing of data, including handling missing values and merging datasets from multiple sources. Through exploratory data analysis (EDA),
we gained a deeper understanding of the distributions and relationships among key variables.

Utilizing regression modeling techniques, we identified significant predictors of life expectancy while ensuring adherence to key assumptions. Feature selection and rigorous
model evaluation enabled us to build robust models capable of accurately predicting life expectancy

By leveraging accurate and reliable data from sources like the World Health Organization (WHO) and the United Nations, our study contributes to the growing body of knowledge
on global public health. The insights gained can inform policymakers and healthcare practitioners in prioritizing interventions to improve life expectancy and overall population
health outcomes.

We extend our gratitude to the WHO, the United Nations, and our collaborators for their invaluable contributions to this research endeavor.

Through collaborative efforts and continued research, we can work towards addressing health disparities and promoting equitable access to healthcare, ultimately striving for
healthier and more prosperous communities worldwide.

Trend of Increasing Life Expectancy: There's a noticeable trend of increasing life expectancy over the years, suggesting a corresponding decline in mortality rates globally.
Zimbabwe's Unique Situation: Zimbabwe stands out with the highest adult mortality rate among the listed countries, indicating significant health challenges. Despite this, its life
expectancy is not the lowest, suggesting other factors at play. Impact of Diseases on Life Expectancy: Measles has a relatively minor impact on life expectancy compared to
HIV/AIDS, which shows a strong negative correlation with life expectancy. Polio and hepatitis B show a weaker correlation. This indicates the significant impact of HIV/AIDS
prevalence on life expectancy. Disparity Between Developed and Developing Countries: Developed countries generally exhibit higher life expectancy due to better living
conditions compared to developing nations. Positive Correlation Between BMI and Life Expectancy: There's a robust and strong positive correlation between Body Mass Index
(BMI) and life expectancy, suggesting that higher BMI is associated with longer life expectancy. Distribution Characteristics: The distribution of the data appears to be almost
normal but with a negative skew, indicating that while the majority of countries may exhibit increasing life expectancy, there are outliers with lower life expectancies affecting the
overall distribution.
