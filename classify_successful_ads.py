# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class TVCommercialClassifier:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.threshold = 0.5  # Define successful commercials as those with interruption rate ≤ 0.5

    def load_and_preprocess_data(self, file_path):

        df = pd.read_csv(file_path)
        print("Data shape:", df.shape)
        print("Missing Values:\n", df.isnull().sum())


        df['airing_data_aired_at_et'] = pd.to_datetime(df['airing_data_aired_at_et'])
        df['hour'] = df['airing_data_aired_at_et'].dt.hour
        df['day_of_week'] = df['airing_data_aired_at_et'].dt.dayofweek
        df['month'] = df['airing_data_aired_at_et'].dt.month


        def get_time_slot(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 23:
                return 'primetime'
            else:
                return 'late_night'

        df['time_slot'] = df['hour'].apply(get_time_slot)


        df['cost_per_impression'] = df['airing_data_spend_estimated'] / df['audience_data_impressions']
        df['spend_category'] = pd.qcut(df['airing_data_spend_estimated'],
                                       q=5,
                                       labels=['very_low', 'low', 'medium', 'high', 'very_high'])


        df['audience_category'] = pd.qcut(df['audience_data_impressions'],
                                          q=5,
                                          labels=['very_small', 'small', 'medium', 'large', 'very_large'])


        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)


        df['is_successful'] = (df['view_completion_data_rate_interruption'] <= self.threshold).astype(int)


        le = LabelEncoder()
        categorical_cols = ['brand_data_name', 'time_slot', 'spend_category', 'audience_category']
        for col in categorical_cols:
            df[f'{col}_encoded'] = le.fit_transform(df[col])


        scaler = StandardScaler()
        numerical_cols = ['airing_data_spend_estimated', 'audience_data_impressions',
                          'cost_per_impression', 'hour', 'day_of_week', 'month']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


        features = ['hour', 'day_of_week', 'month',
                    'airing_data_spend_estimated', 'audience_data_impressions',
                    'cost_per_impression', 'brand_data_name_encoded',
                    'time_slot_encoded', 'spend_category_encoded',
                    'audience_category_encoded']

        X = df[features]
        y = df['is_successful']

        return train_test_split(X, y, test_size=0.2, random_state=42), df

    def analyze_successful_commercials(self, df):
        successful_ads = df[df['is_successful'] == 1]

        print("\nSuccessful Commercial Characteristics:")
        print(f"Total successful commercials: {len(successful_ads)}")
        print(f"Success rate: {(len(successful_ads) / len(df) * 100):.2f}%")
        print(f"\nAverage cost per impression: {successful_ads['cost_per_impression'].mean():.2f}")
        print("\nMost common time slots:")
        print(successful_ads['time_slot'].value_counts())
        print(f"\nAverage audience size: {successful_ads['audience_data_impressions'].mean():.0f}")

        return successful_ads

    def initialize_models(self):
        self.models = {
            'XGBoost': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', XGBRegressor(
                    objective='binary:logistic',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ))
            ]),

            'LightGBM': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', LGBMRegressor(
                    objective='binary',
                    n_estimators=100,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42
                ))
            ]),

            'RandomForest': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ))
            ])
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        best_f1 = 0

        for name, pipeline in self.models.items():
            print(f"\nTraining {name}...")

            pipeline.fit(X_train, y_train)
            y_pred = (pipeline.predict(X_test) > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            self.results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'model': pipeline
            }

            print(f"{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': pipeline.named_steps['model'].feature_importances_
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                print("\nTop 5 Important Features:")
                print(feature_importance.head())

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = pipeline


def main():

    classifier = TVCommercialClassifier()


    (X_train, X_test, y_train, y_test), preprocessed_df = classifier.load_and_preprocess_data(
        "interruption_rate_20192020.csv"
    )
    successful_ads = classifier.analyze_successful_commercials(preprocessed_df)


    classifier.initialize_models()
    classifier.train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()