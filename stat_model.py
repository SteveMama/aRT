from eda import *
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class TVInterruptionRatePredictor:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None

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
            ]),

            'GradientBoosting': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('model', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ))
            ])
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        best_rmse = float('inf')

        for name, pipeline in self.models.items():
            print(f"\nTraining {name}...")

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store results
            self.results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'model': pipeline
            }

            print(f"{name} Results:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2 Score: {r2:.4f}")

            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': pipeline.named_steps['model'].feature_importances_
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                print("\nTop 5 Important Features:")
                print(feature_importance.head())

            if rmse < best_rmse:
                best_rmse = rmse
                self.best_model = pipeline

    def visualize_results(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        rmse_scores = [self.results[model]['RMSE'] for model in self.results.keys()]
        plt.bar(self.results.keys(), rmse_scores)
        plt.title('RMSE Comparison')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        r2_scores = [self.results[model]['R2'] for model in self.results.keys()]
        plt.bar(self.results.keys(), r2_scores)
        plt.title('R2 Score Comparison')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


def main():
    predictor = TVInterruptionRatePredictor()

    file_path = "interruption_rate_20192020.csv"
    X_train, X_test, y_train, y_test, preprocessed_df = preprocess_data(file_path)

    predictor.initialize_models()

    predictor.train_and_evaluate(X_train, X_test, y_train, y_test)

    predictor.visualize_results()


if __name__ == "__main__":
    main()