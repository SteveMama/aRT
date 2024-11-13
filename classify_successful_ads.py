import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class TVInterruptionRateAnalyzer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None

    def load_and_analyze_data(self, file_path):

        df = pd.read_csv(file_path)
        print("Data shape:", df.shape)
        print("Missing Values:\n", df.isnull().sum())


        df['airing_data_aired_at_et'] = pd.to_datetime(df['airing_data_aired_at_et'])
        df['hour'] = df['airing_data_aired_at_et'].dt.hour


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


        plt.figure(figsize=(15, 5))


        plt.subplot(1, 3, 1)
        sns.histplot(data=df, x='view_completion_data_rate_interruption',
                     bins=50, kde=True)
        plt.title('Distribution of Interruption Rate')


        plt.subplot(1, 3, 2)
        sns.boxplot(data=df, x='time_slot', y='view_completion_data_rate_interruption')
        plt.title('Interruption Rate by Time Slot')
        plt.xticks(rotation=45)


        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()


        plt.subplot(1, 3, 3)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        plt.tight_layout()
        plt.show()

        return df

    def preprocess_data(self, df):

        df['day_of_week'] = df['airing_data_aired_at_et'].dt.dayofweek
        df['month'] = df['airing_data_aired_at_et'].dt.month


        imputer = SimpleImputer(strategy='mean')
        numerical_cols = ['airing_data_spend_estimated', 'audience_data_impressions']
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])


        df['cost_per_impression'] = df['airing_data_spend_estimated'] / df['audience_data_impressions']


        le = LabelEncoder()
        categorical_cols = ['brand_data_name', 'time_slot']
        for col in categorical_cols:
            df[f'{col}_encoded'] = le.fit_transform(df[col])


        features = ['hour', 'day_of_week', 'month',
                    'airing_data_spend_estimated', 'audience_data_impressions',
                    'cost_per_impression', 'brand_data_name_encoded',
                    'time_slot_encoded']

        X = df[features]
        y = df['view_completion_data_rate_interruption']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def initialize_models(self):
        self.models = {
            'XGBoost': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ))
            ]),
            'LightGBM': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', LGBMRegressor(
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
        results_summary = []

        for name, pipeline in self.models.items():
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results_summary.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

            # Feature importance
            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': pipeline.named_steps['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                print("\nTop 5 Important Features:")
                print(feature_importance.head())


        results_df = pd.DataFrame(results_summary)
        with open('README.md', 'w') as f:
            f.write('# TV Commercial Interruption Rate Analysis\n\n')
            f.write('## Model Performance Metrics\n\n')
            f.write(results_df.to_markdown())

        return results_df


def main():
    analyzer = TVInterruptionRateAnalyzer()


    df = analyzer.load_and_analyze_data("interruption_rate_20192020.csv")


    X_train, X_test, y_train, y_test = analyzer.preprocess_data(df)


    analyzer.initialize_models()
    results = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nResults saved to README.md")
    print("\nFinal Results Summary:")
    print(results)


if __name__ == "__main__":
    main()