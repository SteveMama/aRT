import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Data shape:", df.shape)
    print("Missing Values: ", df.isnull().sum())
    return df


def create_temporal_features(df):
    df['airing_data_aired_at_et'] = pd.to_datetime(df['airing_data_aired_at_et'])

    df['hour'] = df['airing_data_aired_at_et'].dt.hour
    df['day_of_week'] = df['airing_data_aired_at_et'].dt.dayofweek
    df['month'] = df['airing_data_aired_at_et'].dt.month

    def get_time_slot(hour):
        if 6 <=hour < 12:
            return 'morning'
        elif 12 <=hour < 17:
            return 'afternoon'
        elif 17 <- hour <23:
            return 'primetime'
        else:
            return 'late_night'

    df['time_slot'] = df['hour'].apply(get_time_slot)

    return df


def create_fin_features(df):

    df['cost_per_impression'] = df['airing_data_spend_estimated'] / df['audience_data_impressions']

    df['spend_category'] = pd.qcut(df['airing_data_spend_estimated'], q = 5, labels=['very_low',
                                                                                     'low',
                                                                                     'medium',
                                                                                     'high',
                                                                                     'very_high'])
    return df

def create_audience_feat(df):
    df['audience_category'] = pd.qcut(df['audience_data_impressions'],
                                      q = 5,
                                      labels=['very_small', 'small','medium','large','very_large'])

    return df


def preprocess(df):

    df['view_complete_data_rate_interruption'].fillna(df['view_complete_data_rate_interruption'].mean(), inplace = True)

    # IQR Method implementation

    Q1 = df['view_completion_data_rate_interruption'].quantile(0.25)
    Q3 = df['view_completion_data_rate_interruption'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df['view_completion_data_rate_interruption'] >= lower_bound) &
            (df['view_completion_data_rate_interruption'] <= upper_bound)]

    return df


def encode_categorical(df):
    le = LabelEncoder()

    categorical_cols = ['brand_data_name', 'time_slot', 'spend_category','audience_category']

    for col in categorical_cols:
        df[f'{col}_encoded'] = le.fit_transform(df[col])

    return df

def scale_num_feat(df):
    scaler = StandardScaler()

    num_cols = ['airing_data_spend_estimated', 'audience_data_impressions', 'cost_per_impression', 'hour','day_of_week','month']

    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

    return df_scaled

def preprocess_data(file_path):

    df = load_dataset(file_path)

    df = create_temporal_features(df)
    df = create_fin_features(df)
    df = create_audience_feat(df)

    df = preprocess(df)
    df = encode_categorical(df)
    df = scale_num_feat(df)

    features = ['hour', 'day_of_week', 'month',
                'airing_data_spend_estimated', 'audience_data_impressions',
                'cost_per_impression', 'brand_data_name_encoded',
                'time_slot_encoded', 'spend_category_encoded',
                'audience_category_encoded']

    target = 'view_completion_data_rate_interruption'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, df
