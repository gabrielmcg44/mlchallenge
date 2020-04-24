# 3778's Machine Learning Challenge, My Solutions

# Dataset
  The dataset consists of a time-series of several economic and health indicators of the last four decades (beginning from the 80s) from hundreds of countries. The train set has no data for the years of 2013, 2014 and 2015 and the indicators behaved as shown below:
  FIGURA 1
  Each of those curves represents one indicator for one country among the years. Obviously, there is one diferent behaving for each combination of country and indicator. Since the influence from those atributes is probably far from linear, those atributes were merged into one and generated a ticker. For each ticker, a regression curve was made.
  
# First Approach: Linear Regression
  The first model implemented at __init__.py was a multiple linear regression and, from the trainning, results an array Nx2, where N is the number of tickers, with the two coeficients (angular and linear) for all the N regressions. One of the N regressions can be seen below:
  
  As you can see, the linar function is a good aproximation, but probably not the bast performance.
  The evaluations metrics for the linear regression were:
  
# Second Approach: Third Degree Polinomial Regression
  Since the behaving of the indicators are not exactly straight, the polinomial regression has been used. It is natural that this approach performs better than the previous one, since the linear regression is a particular case of the Third Degree Polinomial Regression. One of the N regressions can be seen below:
  
  Which give us a better fit than the first approach.
  The evaluations metrics for the third degree polinomial regression were:
  

## Dependencies
- Python 3.6 or superior
- Modules listed in `requirements.txt`
