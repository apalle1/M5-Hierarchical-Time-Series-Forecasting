## M5-Hierarchical-Time-Series-Forecasting

How much camping gear will one store sell each month in a year? To the uninitiated, calculating sales at this level may seem as difficult as predicting the weather. Both types of forecasting rely on science and historical data. While a wrong weather forecast may result in you carrying around an umbrella on a sunny day, inaccurate business forecasts could result in actual or opportunity losses.

In this Makridakis competition, the fifth iteration, we are given hierarchical sales data for Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

## Recursive features - Update Target before lag/rolling calculations.

The recursive strategy involves using a one-step model multiple times where the prediction for the prior time step
is used as an input for making a prediction on the following time step. An advantage of using the recursive strategy
is that only one model is required, saving significant computational time, especially when a large number of time
series and forecast horizons are involved.

Iterative feature engineering is the key. In below example, by predicting d_1914 we can calculate features for predicting d_1915 and so on…

![alt text](https://github.com/apalle1/M5-Hierarchical-Time-Series-Forecasting/blob/master/Recursive%20Features.PNG)

Detailed Explanation:

* Assume that we want to predict sales for day 1920
* We will need rolling_mean_1_7 (rolling mean for a window of 7 days) as an input feature to our model
* To calculate this feature we need sales for days 1913, 1914, 1915, 1916, 1917, 1918, 1919
* With training set in our hands we have only sales for day 1913 
* Thats why we do recursive predictions and rolling calculations for 1914, 1915, 1916, 1917, 1918, 1919 … days.
    * calculate rollings for day 1914, predict sales for day 1914
    * calculate rollings for day 1915, predict sales for day 1915
    * calculate rollings for day 1916, predict sales for day 1916 etc
* We avoid Nans and feed our model with valuable information.

The above table was just an example to show the idea behind recursive prediction. I don't use lags under 7 in my model.
If you use recent demand values, the model will predict almost the same values for the next 28 days. And lag_1 seems to 
be the worst one to be added. If you use recent demand values, for example lag_1, the model will give high importance to
that feature. Then if you are using predictions to make other predictions from period t0 to t28 each time you predict
you will increase the error of the prediction. As your model consider that lag_1 is a strong feature and you are extracting
that feature from predictions you have really big chance of overfitting unless your model is really, really good.

## Lag & Rolling Window Features:

https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/

**I found the idea of using lag and rolling window features fascinating. So here are the notes and the "intuition" behind it:**

![alt text](https://github.com/apalle1/M5-Hierarchical-Time-Series-Forecasting/blob/master/Lag-Rolling%20Features.png)

**First off what each feature mathematically does**

* `lag_7`: sales shifted 7 steps downwards for each group. The example above focuses on one group only as an example. That is why the first value appears on the 7th index.
* **lag_28**: sales shifted 28 steps downwards. That is why the first value appears on the 28th index.
* **rmean_7_7**: rolling mean sales of a window size of 7 over column lag_7. First value (0.2857) appears on the 13th index because means including nan are nan.
* **rmean_7_28**: rolling mean sales of a window size of 28 over column lag_7. First value (0.357) appears on the 34th index because that is the first time the mean formula gets all 28 non-nan values.
* **rmean_28_7**: rolling mean sales of a window size of 7 over column lag_28. First value (0.2857) appears on the 3th index because it is the first time the mean formula gets 7 non-nan values.
* **rmean_28_28**: rolling mean sales of a window size of 28 over column lag_28. First value appears on 55th index because that is the first time the formula here all non-nan values.

**The intuition as far as I can understand is the following:**

* Captures the week-on-week similarity and that too of just the past week. In other words, people are likely to shop this monday similar to the last monday (except it is some special occassion).

* Captures the weekly similarity from a month-to-month perspective. Example: people in the 1st weekend of a month shop more so that weekend looks more similar to first weeks of other months than the previous weekend. (Though 28 is arguable here. A month is generally 30. Interesting would be a variable window depending on when the comparative week starts. Dealing with edge cases like week divided into 2 months will be tricky).

**Since individual data points are prone to erratic spikes or troughs, mean provides a more "representative" picture.**

* Captures the information regarding the sales of the whole previous week ending 7 days in the past i.e. if we are at day 14, then the average is of sales from days 1-7 NOT days 7-14. This provides the information about the whole week and not just a single day sale comparison like lag_7 to bring the lag_7 value into "better weekly context".

* Captures the information regarding the sales of the entire previous 4 weeks ending 7 days in the past i.e. if we are at day 35, then the average is sales from days 1-28.

* Captures the information regarding the sales of the whole week ending 4 weeks ago i.e. if we are on day 35, then the average is of sales from day 1-7. (Assuming for simplicity the month is 28 days), this provides the information of not just a month-to-month comparison of the same day (day 7 of month one vs day 7 of month two), but the entire week leading up to day 7. Again the idea I believe is to capture the whole week and not just a single day sale comparison like lag_28 to bring the lag_28 value into "better weekly context".

* Captures the information regarding the sales of the entire previous 4 weeks ending 4 weeks in the past i.e. if we are at day 56, then the average is of days 1-28. (Assuming for simplicity the month is 28 days), the idea again is to bring the point value of lag_28 into a better context (i.e. of day 28 when being compared to day 56) into a "better monthly context".

## Model

* LightGBM (single model)
* objective = tweedie
   * Tweedie regression - These models are designed to deal with right-skewed data with most of the data distribution "concentrated" around 0. If you think the underlying data has a tweedie-distribution, you might want to use tweedie regression.

## Validation
* 1 holdout (d1914-d1941)

## Model split

```
for each store (10 stores - CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3)
    model s1 predicts F01, F02, …, F28
    model s2 predicts F01, F02, …, F28
    model s3 predicts F01, F02, …, F28
    .
    .
    .
    model s10 predicts F01, F02, …, F28
```

## Features

We used the following features. All features are concatenated and fed to the network.

* Sale values
   * Lag 28-42 days
   * Rolling mean and std - 7, 14, 30, 60, 180 days
   * Rolling mean of a window size of [7,14,30,60] over column [lag_1, lag_7, lag_14]

* Calendar: all values are normalized to [-0.5,0.5]
   * day
   * week
   * month
   * year
   * week of month
   * day of week
   * weekend
  
* Event
   * Event type : [SuperBowl, ValentinesDay, PresidentsDay, LentStart, LentWeek2, StPatricksDay, Purim End, OrthodoxEaster,
   Pesach End, Cinco De Mayo, Mother's day, MemorialDay, NBAFinalsStart, NBAFinalsEnd, Father's day, IndependenceDay, Ramadan starts, Eid al-Fitr, LaborDay, ColumbusDay, Halloween, EidAlAdha, VeteransDay, Thanksgiving, Christmas, Chanukah End, NewYear,
OrthodoxChristmas, MartinLutherKingDay, Easter]
   * Event name : [Sporting, Cultural, National, Religious]

* Supplement Nutrition Assistance Program (SNAP) : [0, 1]
  
* Price features
 
* Category
   * stateid, storeid, catid, deptid, item_id 

## Other ideas


* Validation
   * 5 holdout (d1802-d1829, d1830-d1857, d1858-d1885, d1886-d1913, d1914-d1941)

https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection

2) Model split
```
for each store (10 stores - CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3)
  for each dept (7 departments - HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2, FOODS_1, FOODS_2, FOODS_3)
    model d1 predicts F01, F02, …, F28
    model d2 predicts F01, F02, …, F28
    model d3 predicts F01, F02, …, F28
    .
    .
    .
    model 7 predicts F01, F02, …, F28
```
```
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


