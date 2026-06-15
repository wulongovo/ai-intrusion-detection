# -*- coding: utf-8 -*-
"""
训练多个入侵检测模型: 随机森林, 梯度提升, MLP
"""
import sys, os, joblib, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

LABEL_MAP = {0: 'normal', 1: 'dos', 2: 'probe', 3: 'r2l', 4: 'u2r'}

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    X_train = train.drop('label', axis=1)
    y_train = train['label']
    X_test = test.drop('label', axis=1)
    y_test = test['label']
    return X_train, y_train, X_test, y_test

def train_random_forest(X_train, y_train, X_test, y_test):
    print("="*50)
    print("训练 随机森林 模型...")
    start = time.time()
    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    print(f"准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
    print(classification_report(y_test, y_pred, target_names=LABEL_MAP.values()))
    return model, {'accuracy': acc, 'time': elapsed}

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    print("="*50)
    print("训练 梯度提升 模型...")
    start = time.time()
    model = GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    print(f"准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
    print(classification_report(y_test, y_pred, target_names=LABEL_MAP.values()))
    return model, {'accuracy': acc, 'time': elapsed}

def train_mlp(X_train, y_train, X_test, y_test):
    print("="*50)
    print("训练 MLP神经网络 模型...")
    start = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, activation='relu', random_state=42, early_stopping=True)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start
    print(f"准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
    print(classification_report(y_test, y_pred, target_names=LABEL_MAP.values()))
    return model, scaler, {'accuracy': acc, 'time': elapsed}

def main():
    os.makedirs('models', exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}, 特征数: {X_train.shape[1]}")
    results = {}
    rf_model, rf_m = train_random_forest(X_train, y_train, X_test, y_test)
    joblib.dump(rf_model, 'models/random_forest.pkl')
    results['random_forest'] = rf_m
    gb_model, gb_m = train_gradient_boosting(X_train, y_train, X_test, y_test)
    joblib.dump(gb_model, 'models/gradient_boosting.pkl')
    results['gradient_boosting'] = gb_m
    mlp_model, scaler, mlp_m = train_mlp(X_train, y_train, X_test, y_test)
    joblib.dump(mlp_model, 'models/mlp.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    results['mlp'] = mlp_m
    joblib.dump(list(X_train.columns), 'models/feature_names.pkl')
    with open('models/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*50)
    print("所有模型训练完成!")
    for name, m in results.items():
        print(f"  {name}: 准确率={m['accuracy']:.4f}, 耗时={m['time']:.1f}s")

if __name__ == '__main__':
    main()
