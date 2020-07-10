## M5-Hierarchical-Time-Series-Forecasting

How much camping gear will one store sell each month in a year? To the uninitiated, calculating sales at this level may seem as difficult as predicting the weather. Both types of forecasting rely on science and historical data. While a wrong weather forecast may result in you carrying around an umbrella on a sunny day, inaccurate business forecasts could result in actual or opportunity losses.

In this Makridakis competition, the fifth iteration, we are given hierarchical sales data for Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

## Implementation

Recursive features

Rollings are calculated this way because we need to update our Target before rolling calculations.

Example:

We need to predict day 1920
We have rolling mean feature with 7 days window
To calculate such feature we need Target for days 1913, 1914, 1915, 1916, 1917, 1918, 1919
With training set in our hands we have only Target for day 1913 (Test set Target is Nan)
Thats why we do recursive predictions and rolling calculations for 1914, 1915, 1916, 1917, 1918, 1919 … days.
-- Predict day 1914
-- calculate rollings for 1915, predict 1915
-- calculate rollings for 1916, predict 1916
etc

To avoid Nans and feed our model with somehow valuable information.


```
1) Model
LightGBM (single model)
objective = tweedie
Tweedie regression - These models are designed to deal with right-skewed data with most of the data distribution "concentrated" around 0. 
If you think the underlying data has a tweedie-distribution, you might want to use tweedie regression.

2) Validation
1 holdout (d1914-d1941)

3) Model split
for each store (10 stores - CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3)
    model s1 predicts F01, F02, …, F28
    model s2 predicts F01, F02, …, F28
    model s3 predicts F01, F02, …, F28
    .
    .
    .
    model s10 predicts F01, F02, …, F28

4)Features
Weekdays lags
Rolling lags
Calendar features and events 
Price features

Features
We used the following features. All features are concatenated and fed to the network.

Sale values
Lag 1 value
Moving average of 7, 28 days
Calendar: all values are normalized to [-0.5,0.5]
wday
month
year
week number
day
Event
Event type : use embedding
Event name : use embedding
SNAP : [0, 1]
Price
raw value
Normalized across time
Normalized within the same dept_id
Category
stateid, storeid, catid, deptid, item_id : use embedding
Zero sales
Continuous zero-sale days until today


5) Recursive training strategy
The recursive strategy involves using a one-step model multiple times where the prediction for the prior time step
is used as an input for making a prediction on the following time step. An advantage of using the recursive strategy
is that only one model is required, saving significant computational time, especially when a large number of time
series and forecast horizons are involved.
```

## Other ideas

```
1) Validation
https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
5 holdout (d1578-d1605, d1830-d1857, d1858-d1885, d1886-d1913, d1914-d1941)

2) Model split
for each store (10 stores - CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3)
  for each dept (7 departments - HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2, FOODS_1, FOODS_2, FOODS_3)
    model d1 predicts F01, F02, …, F28
    model d2 predicts F01, F02, …, F28
    model d3 predicts F01, F02, …, F28
    .
    .
    .
    model 7 predicts F01, F02, …, F28
    
for each store (10 stores - CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3)
  for each cat (3 Categories - HOBBIES, HOUSEHOLD, FOODS)
    model c1 predicts F01, F02, …, F28
    model c2 predicts F01, F02, …, F28
    model c3 predicts F01, F02, …, F28
    
for each store
  for each week
    model w1 predicts F01, F02, …, F07
    model w2 predicts F08, F09, …, F14
    model w3 predicts F15, F16, …, F21
    model w4 predicts F22, F23, …, F28
```


