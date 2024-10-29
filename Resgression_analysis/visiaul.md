![image](https://github.com/user-attachments/assets/974ef8e9-e453-4497-8f4a-d747ce4e9f23)
![image](https://github.com/user-attachments/assets/9567824f-af57-4c67-ae9f-f43d8b3a83b1)
![image](https://github.com/user-attachments/assets/756c0aa9-9d34-478c-8f81-019f2c5026ea)
![image](https://github.com/user-attachments/assets/5b20af8d-e132-49c3-810b-a65d5eac9853)
![image](https://github.com/user-attachments/assets/872e2919-0c85-4f57-a5ba-787a7e4cffd0)
![image](https://github.com/user-attachments/assets/39666b4c-6db9-41e5-8ad7-3fe096c79b39)
![image](https://github.com/user-attachments/assets/12633436-a44c-4596-8839-cecbc282b69b)
![image](https://github.com/user-attachments/assets/6a526f3d-6c28-4cb2-afbe-953181d6eb6a)
![image](https://github.com/user-attachments/assets/3d1dfd88-eebf-455b-a13d-b7e04efc21d4)
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
