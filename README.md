# Test technique de 111 capital

## Instructions
The attached data set contains 2 csv files: train.csv and test.csv.

Each file contains 4 columns:

* ts: (string) timestamp;
* ref_price: (float) reference price time series;
* alt_price: (float) alternative price time series;
* target: (float) target variable. 

The goal is to build a system that uses ref_price and alt_price to predict the target variable. This means designing and implementing a full data processing and modeling pipeline that includes (or not, that is your choice) the following steps:

1. data loading and pre-processing;
2. creating features;
3. model selection/fitting;
4. performance evaluation. 

Independently of your system design choices, you should select a performance criterium (step 4 in the list above) and report it for your final system.

More important than the absolute value of that performance criteria will be your: methodology; and code readability and organization (for example, name variables descriptively, include comments when necessary, etc).

Please include a document explaining your approach, assumptions (if any), data/model choices, etc. (This does not have to be long, just enough for a reviewer to quickly get an idea of your approach.)