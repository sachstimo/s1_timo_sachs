import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import json

# Load data
def load_data():
    df = pd.read_csv('./DATA/Telecom_Churn.csv')
    y = df['Churn']
    X = df.drop(['Churn', 'State', 'Area code'], axis=1)

    # Binary columns transformation
    bin_cols = ['International plan', 'Voice mail plan']
    X[bin_cols] = X[bin_cols].map(lambda x: 1 if x == 'Yes' else 0)

    # Feature engineering
    X['Total minutes'] = X[['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']].sum(axis=1)
    X['Total charge'] = X[['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']].sum(axis=1)
    X['Total calls'] = X[['Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls']].sum(axis=1)

    X['Minutes per call'] = X['Total minutes'] / X['Total calls']
    X['Service call share'] = X['Customer service calls'] / X['Total calls']

    # Log transformations
    X['Total minutes log'] = np.log1p(X['Total minutes'])
    X['Total charge log'] = np.log1p(X['Total charge'])
    X['Total calls log'] = np.log1p(X['Total calls'])

    # Drop original columns
    drop_cols = [
        'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge',
        'Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes',
        'Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls',
        'Voice mail plan', 'Total minutes', 'Total charge', 'Total calls'
    ]
    X.drop(drop_cols, axis=1, inplace=True)

    return X, y, df[['State', 'Area code']]

# Define profit score
def profit_score(y_real, y_pred, tp_profit, fp_loss, fn_loss, tn_profit):
    TP = np.sum((y_real == 1) & (y_pred == 1))
    FP = np.sum((y_real == 0) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == 0))
    TN = np.sum((y_real == 0) & (y_pred == 0))
    return (TP * tp_profit) - (FP * fp_loss) - (FN * fn_loss) + (TN * tn_profit)

# Train and save model
def train_and_save_model(X, y, tp_profit, fp_loss, fn_loss, tn_profit):
    profit_scorer = make_scorer(profit_score, greater_is_better=True, tp_profit=tp_profit, fp_loss=fp_loss, fn_loss=fn_loss, tn_profit=tn_profit)

    # The paramater grid is loaded from a JSON file with the parameters set to the best values found in the previous notebook
    param_grids = json.load(open('./OUTPUT/hyperparams.json', 'r'))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Cross-validation and pipeline
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(eval_metric='logloss'))
    ])

    grid = GridSearchCV(estimator=pipeline, param_grid=param_grids, cv=skf, scoring=profit_scorer, verbose=2, error_score='raise')

    grid.fit(X_train, y_train)
    
    # Save the best model
    joblib.dump(grid.best_estimator_, './OUTPUT/xgb_model_trained.pkl')
    return grid.best_estimator_