import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class UnsuccessfulAdAnalyzer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.threshold = 0.52  # Define unsuccessful ads as those with interruption rate > 0.5

    def load_and_preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        print("Data shape:", df.shape)
        print("Missing Values:\n", df.isnull().sum())

        df['airing_data_aired_at_et'] = pd.to_datetime(df['airing_data_aired_at_et'])
        df['hour'] = df['airing_data_aired_at_et'].dt.hour
        df['day_of_week'] = df['airing_data_aired_at_et'].dt.dayofweek
        df['month'] = df['airing_data_aired_at_et'].dt.month

        df['time_slot'] = pd.cut(df['hour'],
                                 bins=[0, 6, 12, 18, 24],
                                 labels=['night', 'morning', 'afternoon', 'evening'])

        df['cost_per_impression'] = df['airing_data_spend_estimated'] / df['audience_data_impressions']

        df['is_unsuccessful'] = (df['view_completion_data_rate_interruption'] > self.threshold).astype(int)

        features = ['hour', 'day_of_week', 'month', 'airing_data_spend_estimated',
                    'audience_data_impressions', 'cost_per_impression']

        X = df[features]
        y = df['is_unsuccessful']

        return train_test_split(X, y, test_size=0.2, random_state=42), df

    def analyze_unsuccessful_ads(self, df):
        unsuccessful_ads = df[df['is_unsuccessful'] == 1]

        print("\nUnsuccessful Ad Characteristics:")
        print(f"Total unsuccessful ads: {len(unsuccessful_ads)}")
        print(f"Unsuccessful rate: {(len(unsuccessful_ads) / len(df) * 100):.2f}%")
        print(f"\nAverage cost per impression: ${unsuccessful_ads['cost_per_impression'].mean():.4f}")
        print("\nMost common time slots:")
        print(unsuccessful_ads['time_slot'].value_counts(normalize=True))
        print(f"\nAverage audience size: {unsuccessful_ads['audience_data_impressions'].mean():.0f}")


        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.histplot(unsuccessful_ads['view_completion_data_rate_interruption'], kde=True)
        plt.title('Distribution of Interruption Rates for Unsuccessful Ads')

        plt.subplot(1, 3, 2)
        sns.boxplot(x='time_slot', y='view_completion_data_rate_interruption', data=unsuccessful_ads)
        plt.title('Interruption Rates by Time Slot')

        plt.subplot(1, 3, 3)
        sns.scatterplot(x='cost_per_impression', y='view_completion_data_rate_interruption', data=unsuccessful_ads)
        plt.title('Interruption Rate vs Cost per Impression')

        plt.tight_layout()
        plt.show()

    def initialize_models(self):
        self.models = {
            'RandomForest': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'XGBoost': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', XGBClassifier(n_estimators=100, random_state=42))
            ]),
            'LightGBM': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', LGBMClassifier(n_estimators=100, random_state=42))
            ])
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        for name, pipeline in self.models.items():
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            print(f"{name} Classification Report:")
            print(classification_report(y_test, y_pred))

            self.results[name] = {
                'model': pipeline,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': pipeline.named_steps['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                print("\nTop 5 Important Features:")
                print(feature_importance.head())


def main():
    analyzer = UnsuccessfulAdAnalyzer()
    (X_train, X_test, y_train, y_test), df = analyzer.load_and_preprocess_data("interruption_rate_20192020.csv")

    analyzer.analyze_unsuccessful_ads(df)
    analyzer.initialize_models()
    analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()