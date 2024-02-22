# Data manipulation
import pandas as pd
from datetime import timedelta, date

# Stats 
import numpy as np
from scipy import stats

# Machine learning
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
# Save model
import joblib

#Folder Location
train = 'train.csv'
predict = 'predict.csv'
output = 'close-day-predict.csv'

# Training Features
train_columns = [
               'insert columns'
               ]

# Predict Features
predict_columns = [
                 'insert columns'
               ]


def read_data():
    # read the data with all default parameters
    df = pd.read_csv(train, sep=',', encoding='latin-1')
    predict_df = pd.read_csv(predict, sep=',', encoding='latin-1')
    # copy datafram to write back predictions
    df_out = predict_df.copy()
    df_out = df_out[['id']]
    return df,predict_df,df_out


def transform_data(df,predict_df):
    # using the chaining method concept in pandas
    return (df
         # choose features
          .pipe(drop_columns,train_columns)
         # transform features
          .pipe(data_cleanup)
         # scale features
          .pipe(scale_data)
         ),(
           predict_df
         # choose features
          .pipe(drop_columns,predict_columns)
         # transform features
          .pipe(data_cleanup)
         # scale features
          .pipe(scale_data)
        )


def data_cleanup(df):
    #label encoder for categorical features
    le = LabelEncoder()
    # drop na values
    df = df.dropna()
    # drop columns
    df = df.drop(columns=['id'])
    # convert float values to int
    for col in df.columns[0:]:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(0).astype(int)
    # convert objects values to int
    for col in df.columns[0:]:
        if df[col].dtype == 'object':
            le.fit(df[col])
            df[col] = le.transform(df[col])
    return df

def drop_columns(df,columns):
    # dataframe columns for user
    df = df[columns]
    return df

def scale_data(df):
    # Standard scaler to use on data value conversion
    scaler = StandardScaler()
    # list of columns to be transformed
    df_col = list(df.columns)
    # remove target feature in training dataset
    try:
        df_col.remove('target')
    except:
        None  
    # Scale features
    for col in df_col:
        df[col] = df[col].astype(float)
        df[[col]] = scaler.fit_transform(df[[col]])
    return df


def closed_days_predict(df,predict_df,df_out):
    # Model training
    X, y = df.drop('target',axis=1), df['target']
    # split training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
    # Regressor for closed day predictions (standard tuning)
    model = LinearRegression()
    # fit model 
    model.fit(X_train, y_train)
    # Save model
    joblib.dump(model , 'close_day_model')
    # opening the model
    y_pred = joblib.load('close_day_model')
    # model prediction on open opportunities dataset
    y_pred = y_pred.predict(predict_df)
    # merge closed day prediction onto open opportunities 
    predicted_df = pd.DataFrame(data=y_pred.round(), columns=['Prediction'],index=predict_df.index.copy())
    predicted_df['ClosedDate'] = pd.to_datetime(date.today()) + pd.to_timedelta(predicted_df['Prediction'], unit='D')

    output_df = pd.merge(df_out, predicted_df, how ='left', left_index=True, right_index=True)
    return output_df

def closed_day_buckets(df):
    # Closed day Segments
    if   df['Prediction'] in range(0,29) :
        return 'Closed in 30 Days'
    elif df['Prediction'] in range(30,59) :
        return 'Closed in 60 Days'
    elif df['Prediction'] in range(60,89) :
        return 'Closed in 90 Days'
    else:
        return 'Closed after 90 Days'

def write_data(df):
    # output data to csv
    df.to_csv(output,index=False) 

if __name__ == '__main__':
    # Read data
    df,predict_df,df_out = read_data()
    # Transform data
    df,predict_df = transform_data(df,predict_df)
    # Predict data
    output_df = closed_days_predict(df,predict_df,df_out)
    # Buckets
    output_df['PredictedBuckets'] = output_df.apply(closed_day_buckets, axis=1)
    # output predicted data
    write_data(output_df)
