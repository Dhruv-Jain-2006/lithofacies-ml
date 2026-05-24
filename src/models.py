import logging
import pickle
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
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

def get_random_forest_model():
    """
    Initializes a Random Forest Classifier with the exact hyperparameters
    from Table 3 of Merembayev et al. (2021).
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=70,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features=10,
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
    Finds the optimal number of neighbors (k) for KNN using 5-fold well-based GroupKFold cross-validation.
    """
    logger.info("Tuning K-Nearest Neighbors Classifier...")
    knn = KNeighborsClassifier(weights='distance')
    
    # 5-fold CV split by well groups
    gkf = GroupKFold(n_splits=5)
    
    param_dist = {
        'n_neighbors': np.arange(1, 16)
    }
    
    # Run randomized search
    search = RandomizedSearchCV(
        knn, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=gkf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train, groups=groups_train)
    logger.info(f"KNN tuning completed. Best k: {search.best_params_['n_neighbors']} (Acc: {search.best_score_:.4f})")
    
    return search.best_estimator_

def tune_decision_tree(X_train, y_train, groups_train):
    """
    Finds the optimal max_depth for Decision Tree using 5-fold well-based GroupKFold cross-validation.
    """
    logger.info("Tuning Decision Tree Classifier...")
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    
    # 5-fold CV split by well groups
    gkf = GroupKFold(n_splits=5)
    
    param_dist = {
        'max_depth': np.arange(5, 31)
    }
    
    search = RandomizedSearchCV(
        dt,
        param_distributions=param_dist,
        n_iter=10,
        cv=gkf,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train, groups=groups_train)
    logger.info(f"Decision Tree tuning completed. Best max_depth: {search.best_params_['max_depth']} (Acc: {search.best_score_:.4f})")
    
    return search.best_estimator_

def train_and_save_all(X_train_scaled, X_train_raw, y_train, groups, feature_set_name="original"):
    """
    Trains and returns all 5 models. For KNN, fits on scaled data.
    """
    logger.info(f"Starting training pipeline for feature set: {feature_set_name}...")
    
    # 1. KNN (Scaled features)
    knn_model = tune_knn(X_train_scaled, y_train, groups)
    logger.info("Training KNN on full training set...")
    knn_model.fit(X_train_scaled, y_train)
    
    # 2. Decision Tree (Raw features)
    dt_model = tune_decision_tree(X_train_raw, y_train, groups)
    logger.info("Training Decision Tree on full training set...")
    dt_model.fit(X_train_raw, y_train)
    
    # 3. Random Forest (Raw features - paper tuned)
    rf_model = get_random_forest_model()
    logger.info("Training Random Forest on full training set...")
    rf_model.fit(X_train_raw, y_train)
    
    # 4. XGBoost (Raw features - paper tuned)
    xgb_model = get_xgboost_model()
    logger.info("Training XGBoost on full training set...")
    xgb_model.fit(X_train_raw, y_train)
    
    # 5. LightGBM (Raw features - paper tuned)
    lgb_model = get_lightgbm_model()
    logger.info("Training LightGBM on full training set...")
    lgb_model.fit(X_train_raw, y_train)
    
    models = {
        "KNN": knn_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model
    }
    
    os.makedirs("data/models", exist_ok=True)
    filepath = f"data/models/models_{feature_set_name}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(models, f)
    logger.info(f"All 5 models for feature set '{feature_set_name}' saved to {filepath}!")
    
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
