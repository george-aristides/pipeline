# ml_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
    f1_score,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class MLModelPipeline:
    def __init__(
        self,
        df,
        target_column,
        task='classification',
        test_size=0.2,
        random_state=42,
        display_analytics=True,
        resampling_method='none',
        cross_validation=False,
        cv_folds=5,
    ):
        """
        Initializes the ML pipeline.

        Parameters:
        - df: pandas DataFrame containing the dataset.
        - target_column: The name of the target column.
        - task: 'classification' or 'regression'
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Controls the shuffling applied to the data before applying the split.
        - display_analytics: Whether to display analytics for each model.
        - resampling_method: 'none', 'oversample', 'undersample', 'smote'
        - cross_validation: Whether to perform cross-validation during training.
        - cv_folds: Number of folds for cross-validation.
        """
        self.df = df
        self.target_column = target_column
        self.task = task
        self.test_size = test_size
        self.random_state = random_state
        self.display_analytics = display_analytics
        self.resampling_method = resampling_method
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.models = self._initialize_models()
        self.model_results = {}
        self.best_model = None
        self.preprocessor = None

    def _initialize_models(self):
        if self.task == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'SVM': SVC(probability=True),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Extra Trees': ExtraTreesClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'AdaBoost': AdaBoostClassifier(),
            }
        elif self.task == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'SVR': SVR(),
                'KNN Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Extra Trees Regressor': ExtraTreesRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }
        else:
            raise ValueError("Invalid task. Choose 'classification' or 'regression'.")
        return models

    def preprocess_data(self):
        """Cleans and preprocesses the data."""
        # Check for features with 25% or more NaNs and drop them
        threshold = 0.25
        missing_percentage = self.df.isnull().mean()
        features_to_drop = missing_percentage[missing_percentage >= threshold].index
        self.df.drop(columns=features_to_drop, inplace=True)
        print(f"Dropped features with >=25% missing values: {features_to_drop.tolist()}")

        # Drop rows with null values
        self.df.dropna(inplace=True)

        # Separate features and target
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Define preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Fit and transform the training data, transform the test data
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        # Resampling if needed
        if self.resampling_method != 'none':
            if self.task != 'classification':
                print("Resampling methods are only applicable for classification tasks.")
            else:
                if self.resampling_method == 'oversample':
                    sampler = RandomOverSampler(random_state=self.random_state)
                elif self.resampling_method == 'undersample':
                    sampler = RandomUnderSampler(random_state=self.random_state)
                elif self.resampling_method == 'smote':
                    sampler = SMOTE(random_state=self.random_state)
                else:
                    print(f"Unknown resampling method: {self.resampling_method}. Proceeding without resampling.")
                    sampler = None
                if sampler:
                    X_train, y_train = sampler.fit_resample(X_train, y_train)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_models(self):
        """Trains all models and evaluates them."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            clf = Pipeline(steps=[('model', model)])
            clf.fit(self.X_train, self.y_train)
            y_pred_train = clf.predict(self.X_train)
            y_pred_test = clf.predict(self.X_test)

            if self.task == 'classification':
                train_score = accuracy_score(self.y_train, y_pred_train)
                test_score = accuracy_score(self.y_test, y_pred_test)
                if self.cross_validation:
                    cv_scores = cross_val_score(
                        clf,
                        self.X_train,
                        self.y_train,
                        cv=self.cv_folds,
                        scoring='accuracy',
                    )
            else:
                train_score = r2_score(self.y_train, y_pred_train)
                test_score = r2_score(self.y_test, y_pred_test)
                if self.cross_validation:
                    cv_scores = cross_val_score(
                        clf,
                        self.X_train,
                        self.y_train,
                        cv=self.cv_folds,
                        scoring='r2',
                    )

            self.model_results[name] = {
                'model': clf,
                'train_score': train_score,
                'test_score': test_score,
                'y_pred_test': y_pred_test,
            }

            if self.cross_validation:
                self.model_results[name]['cv_scores'] = cv_scores
                self.model_results[name]['cv_mean'] = cv_scores.mean()
                self.model_results[name]['cv_std'] = cv_scores.std()

            if self.display_analytics:
                print(f"\n{name} Results:")
                print(f"Training Score: {train_score:.4f}")
                print(f"Test Score: {test_score:.4f}")
                if self.cross_validation:
                    print(
                        f"Cross-Validation Score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
                    )
                if self.task == 'classification':
                    print("\nClassification Report:")
                    print(classification_report(self.y_test, y_pred_test))
                    print("Confusion Matrix:")
                    cm = confusion_matrix(self.y_test, y_pred_test)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'{name} Confusion Matrix')
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    plt.show()
                else:
                    print("\nRegression Metrics:")
                    mse = mean_squared_error(self.y_test, y_pred_test)
                    print(f"Mean Squared Error: {mse:.4f}")

    def select_best_model(self):
        """Selects the best model based on test score."""
        best_score = -np.inf
        best_model_name = None

        for name, results in self.model_results.items():
            if results['test_score'] > best_score:
                best_score = results['test_score']
                best_model_name = name

        self.best_model = self.model_results[best_model_name]['model']
        print(
            f"The best model is {best_model_name} with a test score of {best_score:.4f}."
        )

    def get_feature_importance(self):
        """Displays feature importance for models that support it."""
        if self.best_model is None:
            print("Please run select_best_model() first.")
            return

        model = self.best_model.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.preprocessor.get_feature_names_out()
            feature_importances = pd.Series(importances, index=feature_names)
            feature_importances.sort_values(ascending=False, inplace=True)
            plt.figure(figsize=(10, 6))
            feature_importances.head(20).plot(kind='bar')
            plt.title('Feature Importances')
            plt.show()
        elif hasattr(model, 'coef_'):
            importances = model.coef_.flatten()
            feature_names = self.preprocessor.get_feature_names_out()
            feature_importances = pd.Series(importances, index=feature_names)
            feature_importances.sort_values(ascending=False, inplace=True)
            plt.figure(figsize=(10, 6))
            feature_importances.head(20).plot(kind='bar')
            plt.title('Feature Coefficients')
            plt.show()
        else:
            print("The best model does not support feature importance.")

    def tune_best_model(self, param_grid=None):
        """Performs hyperparameter tuning on the best model."""
        if self.best_model is None:
            print("Please run select_best_model() first.")
            return

        model_name = [
            name
            for name, result in self.model_results.items()
            if result['model'] == self.best_model
        ][0]
        base_model = self.models[model_name]

        if param_grid is None:
            # Define default parameter grids for some models
            if isinstance(base_model, (LogisticRegression, LinearRegression)):
                param_grid = {'model__C': [0.01, 0.1, 1, 10, 100]}
            elif isinstance(
                base_model, (RandomForestClassifier, RandomForestRegressor)
            ):
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20, 30],
                }
            else:
                print(
                    "No default parameter grid defined for this model. Please provide param_grid."
                )
                return

        print(f"Tuning hyperparameters for {model_name}...")
        clf = Pipeline(steps=[('model', base_model)])
        grid_search = GridSearchCV(
            clf,
            param_grid,
            cv=self.cv_folds,
            scoring='accuracy' if self.task == 'classification' else 'r2',
        )
        grid_search.fit(self.X_train, self.y_train)
        self.best_model = grid_search.best_estimator_
        self.model_results[model_name]['best_params'] = grid_search.best_params_
        self.model_results[model_name]['best_score'] = grid_search.best_score_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    def tune_threshold(self):
        """Tunes the classification threshold for the best model."""
        if self.best_model is None:
            print("Please run select_best_model() first.")
            return

        if self.task != 'classification':
            print("Threshold tuning is only applicable for classification tasks.")
            return

        model = self.best_model
        if not hasattr(model.named_steps['model'], 'predict_proba'):
            print(
                "The best model does not support probability predictions for threshold tuning."
            )
            return

        y_scores = model.predict_proba(self.X_test)[:, 1]
        thresholds = np.arange(0.0, 1.01, 0.01)
        f1_scores = []
        for thresh in thresholds:
            y_pred_thresh = (y_scores >= thresh).astype(int)
            f1 = f1_score(self.y_test, y_pred_thresh)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal threshold based on F1 score: {optimal_threshold:.2f}")

        # Update best model predictions with optimal threshold
        model_name = [
            name
            for name, result in self.model_results.items()
            if result['model'] == self.best_model
        ][0]
        self.model_results[model_name]['optimal_threshold'] = optimal_threshold
        self.model_results[model_name]['f1_score'] = f1_scores[optimal_idx]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores)
        plt.title('Threshold vs F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.show()

    def save_results(self, filename='model_results.json'):
        """Saves the model results to a JSON file."""
        import json

        def default(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.Series):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return o.to_dict()
            if isinstance(o, Pipeline):
                return str(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, bytes):
                return o.decode('utf-8')
            return str(o)

        with open(filename, 'w') as f:
            json.dump(self.model_results, f, default=default, indent=4)
        print(f"Model results saved to {filename}")

    def run_pipeline(self):
        """Runs the full pipeline."""
        self.preprocess_data()
        self.train_models()
        self.select_best_model()
