# Regression-_analysis

### Context
Although there have been lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition and mortality rates. It was found that affect of immunization and human development index was not taken into account in the past. Also, some of the past research was done considering multiple linear regression based on data set of one year for all the countries. Hence, this gives motivation to resolve both the factors stated previously by formulating a regression model based on mixed effects model and multiple linear regression while considering data from a period of 2000 to 2015 for all the countries. Important immunization like Hepatitis B, Polio and Diphtheria will also be considered. In a nutshell, this study will focus on immunization factors, mortality factors, economic factors, social factors and other health related factors as well. Since the observations this dataset are based on different countries, it will be easier for a country to determine the predicting factor which is contributing to lower value of life expectancy. This will help in suggesting a country which area should be given importance in order to efficiently improve the life expectancy of its population.

### Content
The project relies on accuracy of data. The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of the health status as well as many other related factors for all countries The data-sets are made available to public for the purpose of health data analysis. The data-set related to life expectancy, health factors for 193 countries has been collected from the same WHO data repository website and its corresponding economic data was collected from United Nation website. Among all categories of health-related factors only those critical factors were chosen which are more representative. It has been observed that in the past 15 years , there has been a huge development in health sector resulting in improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project we have considered data from year 2000-2015 for 193 countries for further analysis. The individual data files have been merged together into a single data-set. On initial visual inspection of the data showed some missing values. As the data-sets were from WHO, we found no evident errors. Missing data was handled in R software by using Missmap command. The result indicated that most of the missing data was for population, Hepatitis B and GDP. The missing data were from less known countries like Vanuatu, Tonga, Togo, Cabo Verde etc. Finding all data for these countries was difficult and hence, it was decided that we exclude these countries from the final model data-set. The final merged file(final dataset) consists of 22 Columns and 2938 rows which meant 20 predicting variables. All predicting variables was then divided into several broad categories:​Immunization related factors, Mortality factors, Economical factors and Social factors.

### Acknowledgements
The data was collected from WHO and United Nations website with the help of Deeksha Russell and Duan Wang.



### Project Goals 
- We will be exploring the data, cleaning, analyizng, and perfroming regression analysis on our data.
- Our aim is to build a robust model with high R-squared score and low mean square error for predicting life expectency depending our selected features while addressing the model's assumptions.






![image](https://github.com/user-attachments/assets/0a930d77-e7f2-4892-9c74-c7118552fea5)

![image](https://github.com/user-attachments/assets/6b25e0d9-2472-4526-8b8b-f385180227f1)













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




