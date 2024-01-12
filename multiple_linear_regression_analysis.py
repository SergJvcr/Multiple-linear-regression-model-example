# Packages for numerics + dataframes
import pandas as pd
import numpy as np
# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Packages for date conversions for calculating trip durations
from datetime import date, datetime, timedelta
# Packages for OLS, MLR, confusion matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics # For confusion matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load dataset into dataframe 
df0=pd.read_csv("google_data_analitics\\2017_Yellow_Taxi_Trip_Data.csv")

# Explore data with EDA
print(f'Our dataset contains {df0.shape[0]} rows and {df0.shape[1]} columns.')
print(df0.info())

# Check for missing data and duplicates using .isna() and .drop_duplicates() or .dropna()
missing_data = df0.isna().sum()
print(f'Our data contains missing values: \n {missing_data}')

df_clean = df0.dropna(axis=0)
missing_data_clean = df_clean.isna().sum()

print(f'After cleaning our data contains missing values: \n {missing_data_clean}')
print(df_clean.describe(include='all'))

# Convert pickup & dropoff columns to datetime
# Check the format of the data
print(df_clean.dtypes)
# Convert datetime columns to datetime
print(f"An example of the data in columns with time: {df_clean['tpep_pickup_datetime'][0]}")
print(f"The datatype of the tpep_pickup_datetime coluumn is '{df_clean['tpep_pickup_datetime'].dtype}'")
print(f"The datatype of the tpep_dropoff_datetime coluumn is '{df_clean['tpep_dropoff_datetime'].dtype}'")

df_clean['tpep_pickup_datetime'] = pd.to_datetime(df_clean['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df_clean['tpep_dropoff_datetime'] = pd.to_datetime(df_clean['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

print(f"An example of the data in columns with time after cleaning: {df_clean['tpep_pickup_datetime'][0]}")
print(f"The datatype of the tpep_pickup_datetime coluumn after cleaning is '{df_clean['tpep_pickup_datetime'].dtype}'")
print(f"The datatype of the tpep_dropoff_datetime coluumn after cleaning is '{df_clean['tpep_dropoff_datetime'].dtype}'")

# Create duration column
df_clean['duration'] = (df_clean['tpep_dropoff_datetime'] - df_clean['tpep_pickup_datetime']) / np.timedelta64(1,'m')

# Working with outliers
print(df_clean.info())

print(f"An example of the data in the 'duration' column: {df_clean['duration'][0]}")
print(f"The datatype of the 'duration' coluumn is '{df_clean['duration'].dtype}'")

# Box plots for cheking outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for outlier detection')

sns.boxplot(ax=axes[0], x=df_clean['trip_distance'], color='orange')
axes[0].title.set_text('Boxplot for trip_distance')

sns.boxplot(ax=axes[1], x=df_clean['fare_amount'], color='blue')
axes[1].title.set_text('Boxplot for fare_amount')

sns.boxplot(ax=axes[2], x=df_clean['duration'], color='green')
axes[2].title.set_text('Boxplot for duration')

plt.tight_layout()
plt.show()

# trip_distance outliers
print(sorted(set(df_clean['trip_distance']))[0:50])
# Calculate the count of rides where the trip_distance is zero.
print(f"The sum of the 0.00 trip_distance is {len(df_clean[df_clean['trip_distance'] == 0.00])}.")
# or
print(sum(df_clean['trip_distance'] == 0.00))

# fare_amount outliers
# Impute values less than $0 with 0
print(df_clean['fare_amount'].describe())
df_clean.loc[df_clean['fare_amount'] < 0, 'fare_amount'] = 0
print(f"After fixing the min value in the fare_amount column is {df_clean['fare_amount'].min()}")
print(df_clean['fare_amount'].describe())
# impute the maximum value as Q3 + (6 * IQR)
def outlier_handling(column_list, iqr_factor):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    
    for column in column_list:
        # Reassign minimum to zero
        df_clean.loc[df_clean[column] < 0, column] = 0
        # Calculate upper threshold
        q_1 = df_clean[column].quantile(0.25)
        q_3 = df_clean[column].quantile(0.75)
        iqr = q_3 - q_1
        upper_threshold = q_3 + (iqr_factor * iqr)
        
        print(column)
        print(f'The first quartile (q_1) is {q_1}')
        print(f'The third quartile (q_3) is {q_3}')
        print(f'The upper threshold is {upper_threshold}')
        
        # Reassign values > threshold to threshold
        df_clean.loc[df_clean[column] > upper_threshold, column] = upper_threshold
        print(df_clean[column].describe())

outlier_handling(['fare_amount'], 6)

# duration outliers
print(df_clean['duration'].describe())
# Impute a 0 for any negative values
df_clean.loc[df_clean['duration'] < 0, 'duration'] = 0
print(f"After fixing the min value in the 'duration' column is {df_clean['duration'].min()}")
print(df_clean['duration'].describe())
# Impute the high outliers
outlier_handling(['duration'], 6)

# Checking outliers after cleaning
fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for outlier detection after handling with outliers (cleaning)')

sns.boxplot(ax=axes[0], x=df_clean['trip_distance'], color='orange')
axes[0].title.set_text('Boxplot for trip_distance')

sns.boxplot(ax=axes[1], x=df_clean['fare_amount'], color='red')
axes[1].title.set_text('Boxplot for fare_amount')

sns.boxplot(ax=axes[2], x=df_clean['duration'], color='green')
axes[2].title.set_text('Boxplot for duration')

plt.tight_layout()
plt.show()

# Feature engineering
# Create mean_distance column
# Create `pickup_dropoff` column
df_clean['pickup_dropoff'] = df_clean['PULocationID'].astype(str) + ' ' + df_clean['DOLocationID'].astype(str)
print(df_clean['pickup_dropoff'].head(5))
print(f"The datatype of pickup_dropoff column is '{df_clean['pickup_dropoff'].dtype}'.")
print(df_clean.head(2))

grouped = df_clean.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
print(grouped.head(5))
print(f'Amount of values in pickup_dropoff column is {len(grouped)}')
# Convert `grouped` to a dictionary
grouped_dict = grouped.to_dict()
# Reassign to only contain the inner dictionary
grouped_dict = grouped_dict['trip_distance']
print(grouped_dict)
# Create a mean_distance column that is a copy of the pickup_dropoff helper column
df_clean['mean_distance'] = df_clean['pickup_dropoff']
# Map `grouped_dict` to the `mean_distance` column
df_clean['mean_distance'] = df_clean['mean_distance'].map(grouped_dict)
# Confirm that it worked
print(df_clean.head(5))
print('--------------------------------------------------------------------')
print(df_clean[(df_clean['PULocationID']==100) & (df_clean['DOLocationID']==231)][['mean_distance']])

# Create mean_duration column
grouped_duration = df_clean.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
print(grouped_duration.head(5))
print(f'Amount of values in pickup_dropoff column is {len(grouped)}')
# Create a dictionary where keys are unique pickup_dropoffs and values are
# mean trip duration for all trips with those pickup_dropoff combos
grouped_dict_duration = grouped_duration.to_dict()
grouped_dict_duration = grouped_dict_duration['duration']
df_clean['mean_duration'] = df_clean['pickup_dropoff']
df_clean['mean_duration'] = df_clean['mean_duration'].map(grouped_dict_duration)
# Confirm that it worked
print(df_clean.head(5))
print('--------------------------------------------------------------------')
print(df_clean[(df_clean['PULocationID']==100) & (df_clean['DOLocationID']==231)][['mean_duration']])

# Create day and month columns
# Create 'day' col
df_clean['day'] = df_clean['tpep_pickup_datetime'].dt.day_name().str.lower()
# Create 'month' col
df_clean['month'] = df_clean['tpep_pickup_datetime'].dt.strftime('%b').str.lower()
print(df_clean.head(5))

# Create rush_hour column (1 -yes and 0 - no)
# Define rush hour as:
# - Any weekday (not Saturday or Sunday) AND
# - Either from 06:00–10:00 or from 16:00–20:00
df_clean['rush_hour'] = df_clean['tpep_pickup_datetime'].dt.hour
print(df_clean.head(5))
print('--------------------------------------------------')
# If day is Saturday or Sunday, impute 0 in `rush_hour` column
df_clean.loc[df_clean['day'].isin(['saturday', 'sunday']), 'rush_hour'] = 0
print(df_clean.head(5))

def rush_hourizer(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val

# Apply the `rush_hourizer()` function to the rush column
df_clean.loc[(df_clean.day != 'saturday') & (df_clean.day != 'sunday'), 'rush_hour'] = df_clean.apply(rush_hourizer, axis=1)
df_clean.head(5)

#  Create a scatterplot to visualize the relationship between variables of interest (mean_duration and fare_amount)
sns.set(style='whitegrid')
fig = plt.figure(figsize=(5,5))

sns.regplot(x=df_clean['mean_duration'], y=df_clean['fare_amount'],
            scatter_kws={'s':5, 'alpha':0.5, 'color':'orange'}, line_kws={'color':'green'})

plt.xticks(np.arange(0, 91, step=10))
plt.yticks(np.arange(0, 91, step=10))
plt.title('mean_duration VS fare_amount')
plt.show()

# Check the value of the rides in the second horizontal line in the scatter plot
print(df_clean[df_clean['fare_amount'] > 50]['fare_amount'].value_counts())
# Examine the first 30 of these trips
pd.set_option('display.max_columns', None) # Set pandas to display all columns
print(df_clean[df_clean['fare_amount'] == 52]) 

# It seems that almost all of the trips in the first 30 rows where the fare amount 
# was $52 either begin or end at location 132, and all of them have a RatecodeID of 2.
# There is no readily apparent reason why PULocation 132 should have so many fares of 52 dollars. 
# They seem to occur on all different days, at different times, with both vendors, in all months. 
# However, there are many toll amounts of $5.76 and $5.54. This would seem to indicate that 
# location 132 is in an area that frequently requires tolls to get to and from. It's likely this is an airport.
# The data dictionary says that RatecodeID of 2 indicates trips for JFK, which is 
# John F. Kennedy International Airport. A quick Google search for "new york city taxi flat rate $52" 
# indicates that in 2017 (the year that this data was collected) there was indeed a flat fare 
# for taxi trips between JFK airport (in Queens) and Manhattan.

# Isolate modeling variables
print(df_clean.info())

df_for_modelling = df_clean[['VendorID', 'passenger_count', 'fare_amount', 'mean_distance', 'mean_duration', 'rush_hour']]

print(df_clean.info())
print('-------------------------------------------------')
print(df_for_modelling.info())

# Create a pairplot to visualize pairwise relationships between fare_amount, mean_duration, and mean_distance
sns.pairplot(df_for_modelling[['fare_amount', 'mean_distance', 'mean_duration']],
            plot_kws={'color':'red', 's':5, 'alpha':0.4})
plt.show()

# Identify correlations
# Create correlation matrix containing pairwise correlation of columns, using pearson correlation coefficient
print(df_for_modelling.corr(method='pearson'))
# Visualize a correlation heatmap of the data
plt.figure(figsize=(7, 7))
sns.heatmap(df_for_modelling.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation Heatmap', fontsize=20)
plt.show()
# Exemplar response: mean_duration and mean_distance are both highly correlated with 
# the target variable of fare_amount They're also both correlated with each other, 
# with a Pearson correlation of 0.87. Highly correlated predictor variables can be bad 
# for linear regression models when you want to be able to draw statistical inferences 
# about the data from the model. However, correlated predictor variables can still be used 
# to create an accurate predictor if the prediction itself is more important 
# than using the model as a tool to learn about your data.
# This model will predict fare_amount, which will be used as a predictor variable 
# in machine learning models. Therefore, try modeling with both variables even though they are correlated.

# Split data into outcome variable and features
print(df_for_modelling.info())

X = df_for_modelling.drop(columns=['fare_amount']) # features
y = df_for_modelling[['fare_amount']] # an outcome variable

print(X.head(5))
print(y.head(5))

# Pre-process data
# Convert VendorID to string
print(X.info())
X['VendorID'] = X['VendorID'].astype(str)
print(X.info())
# Get dummies
X = pd.get_dummies(X, drop_first=True)
print(X['VendorID_2'].dtype)
print(X.head(5))
print(X.info())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
# Standardize the X variables
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)

# Fit the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Evaluate model
# Train data
# Evaluate the model performance on the training data
r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)
y_pred_train = lr.predict(X_train_scaled)
print('R^2:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))
# Test data
# Scale the X_test data
X_test_scaled = scaler.transform(X_test)
# Evaluate the model performance on the testing data
r_sq_test = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
y_pred_test = lr.predict(X_test_scaled)
print('R^2:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Results
# Create a `results` dataframe
results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
print(results.head())

# Visualize model results
# Create a scatterplot to visualize `predicted` over `actual`
plt.figure(figsize=(7,7))
sns.set(style='whitegrid')
sns.scatterplot(x=results['actual'], y=results['predicted'], alpha=0.5)
# Draw an x=y line to show what the results would be if the model were perfect
plt.plot([0,60], [0,60], c='red', linewidth=2)
plt.title('Actual values of Y VS Predicted values of Y')
plt.show()

# Visualize the distribution of the `residuals`
sns.histplot(results['residual'], bins=np.arange(-15,15.5,0.5))
plt.title('Distribution of the residuals')
plt.xlabel('Residual value')
plt.ylabel('Count')
plt.show()
# or
sns.histplot(results['residual'], color='purple')
plt.xticks(np.arange(-20, 61, step=10))
plt.title('Distribution of the residuals')
plt.xlabel('Residual value')
plt.ylabel('Count')
plt.show()

# Calculate residual mean
print(f"The residual mean is {results['residual'].mean()}.")
# Create a scatterplot of `residuals` over `predicted values`
sns.scatterplot(x=results['predicted'], y=results['residual'], color='gray')
plt.axhline(0, c='red')
plt.title('Scatterplot of Residuals VS Predicted values')
plt.xlabel('Predicted values')
plt.ylabel('Residual values')
plt.show()
# Exemplar note: The model's residuals are evenly distributed above and below zero, 
# with the exception of the sloping lines from the upper-left corner to the lower-right corner, 
# which we know are the imputed maximum of $62.50 and the flat rate of $52 for JFK airport trips.

# Coefficients
# Output the model's coefficients
coefficients = pd.DataFrame(lr.coef_, columns=X.columns)
print(coefficients)
# The coefficients reveal that mean_distance was the feature with the greatest weight in 
# the model's final prediction. Be careful here! A common misinterpretation is that 
# for every mile traveled, the fare amount increases by a mean of $7.13. This is incorrect. 
# Remember, the data used to train the model was standardized with StandardScaler(). 
# As such, the units are no longer miles. In other words, you cannot say "for every mile traveled...", 
# as stated above. The correct interpretation of this coefficient is: controlling for other variables,
# for every +1 change in standard deviation, the fare amount increases by a mean of $7.13.
# Translate this back to miles instead of standard deviation (i.e., unscale the data):
# 1. Calculate SD of `mean_distance` in X_train data
print(X_train['mean_distance'].std())
# 2. Divide the model coefficient by the standard deviation
print(7.133867 / X_train['mean_distance'].std())
# 1. Calculate SD of `mean_duration` in X_train data
print(X_train['mean_duration'].std())
# 2. Divide the model coefficient by the standard deviation
print(2.812115 / X_train['mean_duration'].std())
# Now you can make a more intuitive interpretation: for every 3.57 miles traveled, 
# the fare increased by a mean of $7.13. Or, reduced: for every 1 mile traveled, 
# the fare increased by a mean of $2.00.

# Conclusion
# When the mean_distance and mean_duration columns were computed, 
# the means were calculated from the entire dataset. These same columns were 
# then used to train a model that was used to predict on a test set. 
# A test set is supposed to represent entirely new data that the model has not seen before, 
# but in this case, some of its predictor variables were derived using data that was in the test set. 
# This is known as data leakage. Data leakage is when information from your training data 
# contaminates the test data. If your model has unexpectedly high scores, 
# there is a good chance that there was some data leakage. 
# To avoid data leakage in this modeling process, it would be best to compute the means 
# using only the training set and then copy those into the test set, 
# thus preventing values from the test set from being included in the computation of the means. 
# This would have created some problems because it's very likely that some combinations 
# of pickup-dropoff locations would only appear in the test data (not the train data). 
# This means that there would be NaNs in the test data, and further steps would be required to address this. 
# In this case, the data leakage improved the R2 score by ~0.03.
# Imputing the fare amount for RatecodeID 2 after training the model and then calculating model 
# performance metrics on the post-imputed data is not best practice. It would be better 
# to separate the rides that did not have rate codes of 2, train the model on that data specifically, 
# and then add the RatecodeID 2 data (and its imputed rates) after. 
# This would prevent training the model on data that you don't need a model for, 
# and would likely result in a better final model. However, the steps were combined for simplicity.
# Models that predict values to be used in another downstream model are common in data science workflows.
# When models are deployed, the data cleaning, imputations, splits, predictions, etc. 
# are done using modeling pipelines. Pandas was used here to granularize and explain 
# the concepts of certain steps, but this process would be streamlined by machine learning engineers. 
# The ideas are the same, but the implementation would differ. 
# Once a modeling workflow has been validated, the entire process can be automated, 
# often with no need for pandas and no need to examine outputs at each step. 
# This entire process would be reduced to a page of code.

