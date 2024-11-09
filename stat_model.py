import numpy as np
import pandas as pd
import matplotlib as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from eda import *


def create_modeling_pipeline():

    xgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ))
    ])

    lgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        ))
    ])


    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ))
    ])


    gb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ))
    ])

    models = {
        'XGBoost': xgb_pipeline,
        'LightGBM': lgb_pipeline,
        'RandomForest': rf_pipeline,
        'GradientBoosting': gb_pipeline
    }

    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")


        model.fit(X_train, y_train)


        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'model': model
        }

        print(f"{name} Results:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        # Feature importance for tree-based models has been calculated here
        if hasattr(model[-1], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model[-1].feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            print("\nTop 5 Important Features:")
            print(feature_importance.head())

    return results


def plot_results(results):
    plt.figure(figsize=(12, 6))

    # RMSE comparison
    plt.subplot(1, 2, 1)
    rmse_scores = [results[model]['RMSE'] for model in results.keys()]
    plt.bar(results.keys(), rmse_scores)
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)

    # R2 comparison
    plt.subplot(1, 2, 2)
    r2_scores = [results[model]['R2'] for model in results.keys()]
    plt.bar(results.keys(), r2_scores)
    plt.title('R2 Score Comparison')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    file_path = "interruption_rate_20192020.csv"
    X_train, X_test, y_train, y_test, preprocessed_df = preprocess_data(file_path)


    models = create_modeling_pipeline()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)


    plot_results(results)