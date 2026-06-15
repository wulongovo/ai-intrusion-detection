# -*- coding: utf-8 -*-
"""
NSL-KDD 真实数据集入侵检测模型训练
数据来源: Canadian Institute for Cybersecurity
"""
import sys, os, joblib, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# NSL-KDD: 41 features + label + difficulty = 43 columns
FEATURE_COLS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
]
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

ALL_COLS = FEATURE_COLS + ['attack_label', 'difficulty']

ATTACK_CATEGORIES = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos',
    'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l',
    'warezclient': 'r2l', 'warezmaster': 'r2l', 'snmpgetattack': 'r2l',
    'named': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l',
    'sendmail': 'r2l', 'httptunnel': 'r2l', 'worm': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
    'rootkit': 'u2r', 'xterm': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
}
CATEGORY_MAP = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
LABEL_CN = {0: '正常流量', 1: 'DOS拒绝服务攻击', 2: '探测扫描攻击', 3: '远程权限攻击', 4: '提权攻击'}


def load_data():
    train_path = os.path.join('data', 'KDDTrain+.txt')
    test_path = os.path.join('data', 'KDDTest+.txt')
    if not os.path.exists(train_path):
        print("错误: 未找到 KDDTrain+.txt"); sys.exit(1)

    train_df = pd.read_csv(train_path, header=None, names=ALL_COLS)
    print(f"训练集: {len(train_df)} 条")

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path, header=None, names=ALL_COLS)
        print(f"测试集: {len(test_df)} 条")
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['attack_label'])
        print(f"自动分割: 训练{len(train_df)} 测试{len(test_df)}")
    return train_df, test_df


def preprocess(train_df, test_df):
    # Map attack labels to 5 categories
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Drop rows with NaN in attack_label, fill NaN in features
    train_df = train_df.dropna(subset=['attack_label'])
    test_df = test_df.dropna(subset=['attack_label'])
    for col in FEATURE_COLS:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)

    train_df['label'] = train_df['attack_label'].map(ATTACK_CATEGORIES).map(CATEGORY_MAP).fillna(0).astype(int)
    test_df['label'] = test_df['attack_label'].map(ATTACK_CATEGORIES).map(CATEGORY_MAP).fillna(0).astype(int)

    # Encode categorical features (fit on train, transform test)
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le

    return train_df, test_df, encoders


def train_and_evaluate(X_train, y_train, X_test, y_test):
    results = {}

    # 随机森林
    print("=" * 50)
    print("训练 随机森林...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - t0
    print(f"准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
    print(classification_report(y_test, y_pred, target_names=LABEL_CN.values(), zero_division=0, digits=4))
    results['random_forest'] = {'accuracy': acc, 'time': elapsed}

    # 梯度提升
    print("=" * 50)
    print("训练 梯度提升...")
    t0 = time.time()
    gb = HistGradientBoostingClassifier(max_iter=150, max_depth=8, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - t0
    print(f"准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
    print(classification_report(y_test, y_pred, target_names=LABEL_CN.values(), zero_division=0, digits=4))
    results['gradient_boosting'] = {'accuracy': acc, 'time': elapsed}

    # MLP
    print("=" * 50)
    print("训练 MLP神经网络...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, activation='relu', random_state=42, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    y_pred = mlp.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - t0
    print(f"准确率: {acc:.4f}  耗时: {elapsed:.1f}s")
    print(classification_report(y_test, y_pred, target_names=LABEL_CN.values(), zero_division=0, digits=4))
    results['mlp'] = {'accuracy': acc, 'time': elapsed}

    return rf, gb, mlp, scaler, results


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("=" * 50)
    print("NSL-KDD 真实数据集 - 入侵检测模型训练")
    print("=" * 50)

    train_df, test_df = load_data()
    train_df, test_df, encoders = preprocess(train_df, test_df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['label']
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['label']

    print(f"\n训练集类别分布:")
    for k, v in y_train.value_counts().sort_index().items():
        print(f"  {LABEL_CN.get(k, k)}: {v}")
    print(f"\n测试集类别分布:")
    for k, v in y_test.value_counts().sort_index().items():
        print(f"  {LABEL_CN.get(k, k)}: {v}")

    rf, gb, mlp, scaler, results = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Save everything
    joblib.dump(rf, 'models/random_forest.pkl')
    joblib.dump(gb, 'models/gradient_boosting.pkl')
    joblib.dump(mlp, 'models/mlp.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(FEATURE_COLS, 'models/feature_names.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')

    with open('models/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    dataset_info = {
        'name': 'NSL-KDD',
        'train_samples': int(len(train_df)),
        'test_samples': int(len(test_df)),
        'features': len(FEATURE_COLS),
        'categories': {LABEL_CN[k]: int(v) for k, v in y_train.value_counts().sort_index().items()},
        'attack_types': list(set(ATTACK_CATEGORIES.values())),
    }
    with open('models/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"数据集: NSL-KDD | 训练集: {len(train_df)} | 测试集: {len(test_df)} | 特征: {len(FEATURE_COLS)}")
    for name, m in results.items():
        print(f"  {name}: 准确率={m['accuracy']:.4f}, 耗时={m['time']:.1f}s")

if __name__ == '__main__':
    main()
