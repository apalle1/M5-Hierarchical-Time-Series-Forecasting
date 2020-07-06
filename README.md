## M5-Hierarchical-Time-Series-Forecasting

How much camping gear will one store sell each month in a year? To the uninitiated, calculating sales at this level may seem as difficult as predicting the weather. Both types of forecasting rely on science and historical data. While a wrong weather forecast may result in you carrying around an umbrella on a sunny day, inaccurate business forecasts could result in actual or opportunity losses.

In this Makridakis competition, the fifth iteration, we are given hierarchical sales data for Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

## Implementation

```
**Model**
LightGBM (single)
objective = tweedie

**Validation**
1 holdout (d1914-d1941)

**Model split**
for each store
    model s1 predicts F01, F02, …, F28
    model s2 predicts F01, F02, …, F28
    model s3 predicts F01, F02, …, F28
    .
    .
    .
    model s10 predicts F01, F02, …, F28

**Features**
Weekdays lags
Rolling lags
Calendar features and events 
Price features

**Recursive training strategy**
The recursive strategy involves using a one-step model multiple times where the prediction for the prior time step
is used as an input for making a prediction on the following time step. An advantage of using the recursive strategy
is that only one model is required, saving significant computational time, especially when a large number of time
series and forecast horizons are involved.
```

## Other ideas

'''
**Validation**
5 holdout (d1578-d1605, d1830-d1857, d1858-d1885, d1886-d1913, d1914-d1941)

**Model split**
for each store
  for each week
    model w1 predicts F01, F02, …, F07
    model w2 predicts F08, F09, …, F14
    model w3 predicts F15, F16, …, F21
    model w4 predicts F22, F23, …, F28

for each store
  for each dept
    model s1 predicts F01, F02, …, F28
    model s2 predicts F01, F02, …, F28
    model s3 predicts F01, F02, …, F28
    .
    .
    .
    model s10 predicts F01, F02, …, F28
'''

write about lightgbm ?

CV

Explain about data 



