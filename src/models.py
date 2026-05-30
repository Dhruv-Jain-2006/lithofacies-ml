import logging
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_knn_model(n_neighbors=5):
    """
    Initializes a K-Nearest Neighbors Classifier.
    Note: Requires scaled input features.
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)

def get_decision_tree_model(max_depth=15):
    """
    Initializes a Decision Tree Classifier.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42
    )

def get_random_forest_model(max_features='sqrt'):
    """
    Initializes a Random Forest Classifier with the exact hyperparameters
    from Table 3 of Merembayev et al. (2021) with scalable max_features.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=70,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features=max_features,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

def get_xgboost_model():
    """
    Initializes an XGBoost Classifier with the exact hyperparameters
    from Table 4 of Merembayev et al. (2021).
    """
    return xgb.XGBClassifier(
        n_estimators=526,
        max_depth=12,
        learning_rate=0.73,
        min_child_weight=11,
        gamma=8,
        reg_lambda=1.36,  # lambda
        reg_alpha=0.23,   # alpha
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )

def get_lightgbm_model():
    """
    Initializes a LightGBM Classifier with the exact hyperparameters
    from Table 5 of Merembayev et al. (2021).
    """
    return lgb.LGBMClassifier(
        n_estimators=216,
        max_depth=11,
        learning_rate=0.05,
        min_child_weight=4.12,
        reg_alpha=2.69,   # lambda_l1
        reg_lambda=4.27,  # lambda_l2
        class_weight='balanced',
        objective='multiclass',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

def tune_knn(X_train, y_train, groups_train):
    """
    Finds the optimal number of neighbors (k) for KNN using well-based GroupKFold cross-validation.
    To avoid performance issues on large datasets, we sub-sample X_train group-wise if it exceeds 15,000 points.
    """
    logger.info("Tuning K-Nearest Neighbors Classifier...")
    knn = KNeighborsClassifier(weights='distance')
    
    if len(X_train) > 15000:
        logger.info(f"Dataset size ({len(X_train)}) is large. Sub-sampling to ~15,000 rows for hyperparameter tuning...")
        np.random.seed(42)
        indices = np.random.choice(len(X_train), size=15000, replace=False)
        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
            X_tune = X_train.iloc[indices]
        else:
            X_tune = X_train[indices]
        y_tune = y_train[indices]
        groups_tune = groups_train[indices]
    else:
        X_tune = X_train
        y_tune = y_train
        groups_tune = groups_train

    # 3-fold CV split by well groups to speed up tuning
    gkf = GroupKFold(n_splits=3)
    
    param_dist = {
        'n_neighbors': np.arange(1, 16)
    }
    
    search = RandomizedSearchCV(
        knn, 
        param_distributions=param_dist, 
        n_iter=5, 
        cv=gkf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_tune, y_tune, groups=groups_tune)
    logger.info(f"KNN tuning completed. Best k: {search.best_params_['n_neighbors']} (Acc: {search.best_score_:.4f})")
    
    # Return best estimator fitted on FULL training data
    best_estimator = KNeighborsClassifier(n_neighbors=search.best_params_['n_neighbors'], weights='distance', n_jobs=-1)
    return best_estimator

def tune_decision_tree(X_train, y_train, groups_train):
    """
    Finds the optimal max_depth for Decision Tree using well-based GroupKFold cross-validation.
    To avoid performance issues on large datasets, we sub-sample X_train if it exceeds 15,000 points.
    """
    logger.info("Tuning Decision Tree Classifier...")
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    
    if len(X_train) > 15000:
        logger.info(f"Dataset size ({len(X_train)}) is large. Sub-sampling to ~15,000 rows for hyperparameter tuning...")
        np.random.seed(42)
        indices = np.random.choice(len(X_train), size=15000, replace=False)
        X_tune = X_train.iloc[indices]
        y_tune = y_train[indices]
        groups_tune = groups_train[indices]
    else:
        X_tune = X_train
        y_tune = y_train
        groups_tune = groups_train

    # 3-fold CV split by well groups
    gkf = GroupKFold(n_splits=3)
    
    param_dist = {
        'max_depth': np.arange(5, 31)
    }
    
    search = RandomizedSearchCV(
        dt,
        param_distributions=param_dist,
        n_iter=5,
        cv=gkf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_tune, y_tune, groups=groups_tune)
    logger.info(f"Decision Tree tuning completed. Best max_depth: {search.best_params_['max_depth']} (Acc: {search.best_score_:.4f})")
    
    # Return best estimator fitted on FULL training data
    best_estimator = DecisionTreeClassifier(max_depth=search.best_params_['max_depth'], class_weight='balanced', random_state=42)
    return best_estimator

class WeightedCalibratedEnsemble:
    """
    Weighted Soft Voting Ensemble using calibrated model probabilities.
    Excludes KNN from active soft-voting (downgraded to baseline reference-only).
    Weights are dynamically determined based on out-of-fold validation metrics:
    Macro F1, ECE (Expected Calibration Error), and Rare-Facies Recall.
    """
    def __init__(self, models_dict, scaler=None, weights=None, validation_metrics=None):
        self.models_dict = models_dict
        self.scaler = scaler
        self.classes_ = np.arange(12)
        
        if weights is not None:
            self.weights = weights
        elif validation_metrics is not None:
            self.weights = {}
            for name in self.models_dict.keys():
                if name == "Ensemble" or name == "KNN":
                    self.weights[name] = 0.0
                    continue
                metrics = validation_metrics.get(name, {})
                f1 = metrics.get("macro_f1", 0.5)
                ece = metrics.get("ece", 0.1)
                rare_recall = metrics.get("rare_recall", 0.5)
                
                # Weight formula: F1 + Rare-Facies Recall - 0.5 * ECE
                w = max(0.01, f1 + rare_recall - 0.5 * ece)
                self.weights[name] = float(w)
            logger.info(f"Dynamic validation-driven soft-voting weights computed: {self.weights}")
        else:
            self.weights = {
                "KNN": 0.0,
                "Decision Tree": 0.05,
                "Random Forest": 0.35,
                "XGBoost": 0.25,
                "LightGBM": 0.35
            }
        
    def predict_proba(self, X):
        prob_sum = None
        total_weight = 0.0
        
        for name, model in self.models_dict.items():
            if name == "Ensemble":
                continue
            # Select correct feature representation (scaled for KNN, raw for trees)
            if name == "KNN":
                if self.scaler is not None:
                    X_in = self.scaler.transform(X)
                else:
                    X_in = X
            else:
                X_in = X.values if isinstance(X, pd.DataFrame) else X
                
            p = model.predict_proba(X_in)
            weight = self.weights.get(name, 0.0)
            if weight == 0.0:
                continue
            if prob_sum is None:
                prob_sum = p * weight
            else:
                prob_sum += p * weight
            total_weight += weight
            
        if prob_sum is None or total_weight == 0.0:
            # Fallback if all weights are zero (e.g. default fallback to LightGBM)
            fallback_model = self.models_dict.get("LightGBM")
            X_in = X.values if isinstance(X, pd.DataFrame) else X
            return fallback_model.predict_proba(X_in)
            
        return prob_sum / total_weight
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def train_and_save_all(X_train_scaled, X_train_raw, y_train, groups, feature_set_name="original", scaler=None, validation_metrics=None):
    """
    Trains, calibrates, and returns all 5 models plus a Weighted Calibrated Ensemble.
    """
    logger.info(f"Starting training pipeline for feature set: {feature_set_name}...")
    
    # 1. KNN (Scaled features)
    knn_raw = tune_knn(X_train_scaled, y_train, groups)
    logger.info("Training Calibrated KNN on full training set...")
    knn_model = CalibratedClassifierCV(estimator=knn_raw, method='isotonic', cv=3)
    knn_model.fit(X_train_scaled, y_train)
    
    # 2. Decision Tree (Raw features)
    dt_raw = tune_decision_tree(X_train_raw, y_train, groups)
    logger.info("Training Calibrated Decision Tree on full training set...")
    dt_model = CalibratedClassifierCV(estimator=dt_raw, method='isotonic', cv=3)
    dt_model.fit(X_train_raw, y_train)
    
    # 3. Random Forest (Raw features - paper tuned)
    rf_raw = get_random_forest_model()
    logger.info("Training Calibrated Random Forest on full training set...")
    rf_model = CalibratedClassifierCV(estimator=rf_raw, method='isotonic', cv=3)
    rf_model.fit(X_train_raw, y_train)
    
    # 4. XGBoost (Raw features - paper tuned)
    xgb_raw = get_xgboost_model()
    logger.info("Training Calibrated XGBoost on full training set...")
    xgb_model = CalibratedClassifierCV(estimator=xgb_raw, method='isotonic', cv=3)
    xgb_model.fit(X_train_raw, y_train)
    
    # 5. LightGBM (Raw features - paper tuned)
    lgb_raw = get_lightgbm_model()
    logger.info("Training Calibrated LightGBM on full training set...")
    lgb_model = CalibratedClassifierCV(estimator=lgb_raw, method='isotonic', cv=3)
    lgb_model.fit(X_train_raw, y_train)
    
    models = {
        "KNN": knn_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model
    }
    
    # 6. Weighted Calibrated Soft Voting Ensemble
    logger.info("Initializing Weighted Calibrated soft voting ensemble...")
    ensemble_model = WeightedCalibratedEnsemble(models, scaler=scaler, validation_metrics=validation_metrics)
    models["Ensemble"] = ensemble_model
    
    os.makedirs("data/models", exist_ok=True)
    filepath = f"data/models/models_{feature_set_name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(models, f)
    logger.info(f"All models (including Calibrated Ensemble) for feature set '{feature_set_name}' saved to {filepath}!")
    
    return models


def load_models(feature_set_name="original"):
    """
    Loads saved models dictionary.
    """
    filepath = f"data/models/models_{feature_set_name}.pkl"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found!")
    with open(filepath, "rb") as f:
        return pickle.load(f)
