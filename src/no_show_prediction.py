import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight
from src import config
from src.plots import PlotGenerator

class NoShowPredictionModel:
    def __init__(self, df, features, target, plotter=None):
        self.df = df
        self.features = features
        self.target = target
        self.plotter = plotter or PlotGenerator()
        self.X = df[features].copy()
        self.y = df[target]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_weight_dict = None
        self.scale_pos_weight = None
        self.base_models = {}
        self.tuned_models = {}
        self.results_df = None
        self.best_model_name = None
        self.best_model = None

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=self.y
        )

    def analyze_class_imbalance(self):
        class_distribution = self.y_train.value_counts(normalize=True)
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        self.class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        self.scale_pos_weight = class_weights[0] / class_weights[1]
        return class_distribution, self.class_weight_dict, self.scale_pos_weight

    def train_baseline_models(self):
        self.base_models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', random_state=config.RANDOM_STATE, max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                class_weight='balanced', random_state=config.RANDOM_STATE, n_estimators=100
            ),
            'XGBoost': XGBClassifier(
                scale_pos_weight=self.scale_pos_weight, random_state=config.RANDOM_STATE,
                eval_metric='logloss', use_label_encoder=False
            )
        }
        results = {}
        for name, model in self.base_models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            results[name] = {
                'F1': f1_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'ROC_AUC': roc_auc_score(self.y_test, y_proba)
            }
        return pd.DataFrame(results).T

    def tune_hyperparameters(self):
        # Use config.HYPERPARAMETERS for all grids
        hp = config.HYPERPARAMETERS
        print("Starting hyperparameter tuning for Logistic Regression...")
        lr_param_grid = hp.get('logistic_regression', {})
        lr_n_iter = hp.get('lr_n_iter', 50)
        lr_random = RandomizedSearchCV(
            LogisticRegression(random_state=config.RANDOM_STATE),
            lr_param_grid, n_iter=lr_n_iter, cv=5, scoring='f1',
            n_jobs=-1, random_state=config.RANDOM_STATE, verbose=1
        )
        lr_random.fit(self.X_train, self.y_train)
        print(f"Best Logistic Regression F1: {lr_random.best_score_:.3f}")
        print(f"Best Logistic Regression params: {lr_random.best_params_}")
        lr_tuned = lr_random.best_estimator_

        print("Starting hyperparameter tuning for Random Forest...")
        rf_param_grid = hp.get('random_forest', {})
        rf_n_iter = hp.get('rf_n_iter', 50)
        rf_random = RandomizedSearchCV(
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            rf_param_grid, n_iter=rf_n_iter, cv=5, scoring='f1',
            n_jobs=-1, random_state=config.RANDOM_STATE, verbose=1
        )
        rf_random.fit(self.X_train, self.y_train)
        print(f"Best Random Forest F1: {rf_random.best_score_:.3f}")
        print(f"Best Random Forest params: {rf_random.best_params_}")
        rf_tuned = rf_random.best_estimator_

        print("Starting hyperparameter tuning for XGBoost...")
        xgb_param_grid = hp.get('xgboost', {})
        xgb_n_iter = hp.get('xgb_n_iter', 75)
        # Inject scale_pos_weight if present in grid
        if 'scale_pos_weight' in xgb_param_grid:
            xgb_param_grid['scale_pos_weight'] = [self.scale_pos_weight]
        xgb_random = RandomizedSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                          random_state=config.RANDOM_STATE, tree_method='hist'),
            xgb_param_grid, n_iter=xgb_n_iter, cv=5, scoring='f1',
            n_jobs=1, random_state=config.RANDOM_STATE, verbose=1
        )
        xgb_random.fit(self.X_train, self.y_train)
        print(f"Best XGBoost F1: {xgb_random.best_score_:.3f}")
        print(f"Best XGBoost params: {xgb_random.best_params_}")
        xgb_tuned = xgb_random.best_estimator_

        self.tuned_models = {
            'Logistic Regression (Tuned)': lr_tuned,
            'Random Forest (Tuned)': rf_tuned,
            'XGBoost (Tuned)': xgb_tuned
        }
        return lr_random, rf_random, xgb_random

    def smote_threshold_optimization(self, best_model):
        print("Starting SMOTE + Threshold Optimization...")
        smote_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=config.RANDOM_STATE, k_neighbors=3)),
            ('classifier', best_model)
        ])
        smote_pipeline.fit(self.X_train, self.y_train)
        smote_proba = smote_pipeline.predict_proba(self.X_test)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_results = []
        for threshold in thresholds:
            y_pred_thresh = (smote_proba >= threshold).astype(int)
            f1 = f1_score(self.y_test, y_pred_thresh)
            precision = precision_score(self.y_test, y_pred_thresh)
            recall = recall_score(self.y_test, y_pred_thresh)
            threshold_results.append({
                'Threshold': threshold,
                'F1': f1,
                'Precision': precision,
                'Recall': recall
            })
            print(f"Threshold: {threshold:.2f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
        threshold_df = pd.DataFrame(threshold_results)
        optimal_idx = threshold_df['F1'].idxmax()
        optimal_threshold = threshold_df.loc[optimal_idx, 'Threshold']
        print(f"Optimal threshold: {optimal_threshold:.2f}")
        print(f"Optimal F1 score: {threshold_df.loc[optimal_idx, 'F1']:.3f}")
        class OptimizedSMOTEModel:
            def __init__(self, pipeline, threshold):
                self.pipeline = pipeline
                self.threshold = threshold
            def predict(self, X):
                proba = self.pipeline.predict_proba(X)[:, 1]
                return (proba >= self.threshold).astype(int)
            def predict_proba(self, X):
                return self.pipeline.predict_proba(X)
        optimized_smote_model = OptimizedSMOTEModel(smote_pipeline, optimal_threshold)
        self.tuned_models[f'{type(best_model).__name__} (SMOTE + Threshold)'] = optimized_smote_model
        return optimized_smote_model, threshold_df

    def evaluate_models(self):
        all_models = {**self.base_models, **self.tuned_models}
        results = {}
        for name, model in all_models.items():
            try:
                y_pred = model.predict(self.X_test)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                elif hasattr(model, 'pipeline'):
                    y_proba = model.pipeline.predict_proba(self.X_test)[:, 1]
                else:
                    y_proba = None
                results[name] = {
                    'F1': f1_score(self.y_test, y_pred),
                    'Precision': precision_score(self.y_test, y_pred),
                    'Recall': recall_score(self.y_test, y_pred),
                    'Accuracy': accuracy_score(self.y_test, y_pred),
                    'ROC_AUC': roc_auc_score(self.y_test, y_proba) if y_proba is not None else 0.0
                }
            except Exception as e:
                results[name] = {'Error': str(e)}
        self.results_df = pd.DataFrame(results).T
        self.results_df = self.results_df[self.results_df.columns[~self.results_df.columns.isin(['Error'])]]
        self.results_df = self.results_df.sort_values('F1', ascending=False)
        self.best_model_name = self.results_df.index[0]
        self.best_model = all_models[self.best_model_name]
        return self.results_df

    def plot_performance(self, show_tuning_impact=True):
        if self.results_df is not None:
            self.plotter.plot_prediction_model_performance(self.results_df, show_tuning_impact)

    def evaluate_model_metrics(self, no_show_cost=150, intervention_cost=25, intervention_success_rate=0.25):
        total_patients = len(self.y_test)
        actual_no_shows = sum(self.y_test)
        best_results = self.results_df.iloc[0]
        precision = best_results['Precision']
        recall = best_results['Recall']
        f1 = best_results['F1']
        predicted_no_shows = int(actual_no_shows / precision) if precision > 0 else 0
        true_positives = int(predicted_no_shows * precision)
        prevented_no_shows = int(true_positives * intervention_success_rate)
        baseline_cost = actual_no_shows * no_show_cost
        intervention_cost_total = predicted_no_shows * intervention_cost
        prevented_cost = prevented_no_shows * no_show_cost
        net_benefit = prevented_cost - intervention_cost_total
        roi = (net_benefit / intervention_cost_total * 100) if intervention_cost_total > 0 else 0
        return {
            'total_patients': total_patients,
            'actual_no_shows': actual_no_shows,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_no_shows': predicted_no_shows,
            'true_positives': true_positives,
            'prevented_no_shows': prevented_no_shows,
            'baseline_cost': baseline_cost,
            'intervention_cost_total': intervention_cost_total,
            'prevented_cost': prevented_cost,
            'net_benefit': net_benefit,
            'roi': roi
        }
