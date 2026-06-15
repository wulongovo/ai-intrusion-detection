# -*- coding: utf-8 -*-
"""
生成模拟网络流量数据集（基于KDD Cup 99特征简化版）
包含正常流量和4类攻击: DOS, Probe, R2L, U2R
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES = [
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
    'dst_host_srv_rerror_rate'
]

ATTACK_MAP = {0: 'normal', 1: 'dos', 2: 'probe', 3: 'r2l', 4: 'u2r'}

def generate_normal(n=5000):
    np.random.seed(42)
    data = {
        'duration': np.random.exponential(100, n),
        'protocol_type': np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
        'service': np.random.randint(0, 20, n),
        'flag': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        'src_bytes': np.random.exponential(500, n),
        'dst_bytes': np.random.exponential(1000, n),
        'land': np.zeros(n),
        'wrong_fragment': np.random.choice([0, 1], n, p=[0.98, 0.02]),
        'urgent': np.zeros(n),
        'hot': np.random.exponential(1, n),
        'num_failed_logins': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'logged_in': np.ones(n),
        'num_compromised': np.zeros(n),
        'root_shell': np.zeros(n),
        'su_attempted': np.zeros(n),
        'num_root': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        'num_file_creations': np.random.exponential(0.5, n),
        'num_shells': np.zeros(n),
        'num_access_files': np.random.exponential(0.3, n),
        'count': np.random.poisson(50, n),
        'srv_count': np.random.poisson(30, n),
        'serror_rate': np.random.beta(1, 50, n),
        'srv_serror_rate': np.random.beta(1, 50, n),
        'rerror_rate': np.random.beta(1, 30, n),
        'srv_rerror_rate': np.random.beta(1, 30, n),
        'same_srv_rate': np.random.beta(20, 2, n),
        'diff_srv_rate': np.random.beta(1, 20, n),
        'srv_diff_host_rate': np.random.beta(1, 10, n),
        'dst_host_count': np.random.poisson(100, n),
        'dst_host_srv_count': np.random.poisson(50, n),
        'dst_host_same_srv_rate': np.random.beta(10, 2, n),
        'dst_host_diff_srv_rate': np.random.beta(1, 10, n),
        'dst_host_serror_rate': np.random.beta(1, 50, n),
        'dst_host_srv_serror_rate': np.random.beta(1, 50, n),
        'dst_host_rerror_rate': np.random.beta(1, 30, n),
        'dst_host_srv_rerror_rate': np.random.beta(1, 30, n),
    }
    return pd.DataFrame(data)

def generate_dos(n=3000):
    np.random.seed(123)
    data = generate_normal(n)
    data['count'] = np.random.poisson(300, n)
    data['srv_count'] = np.random.poisson(250, n)
    data['serror_rate'] = np.random.beta(20, 2, n)
    data['srv_serror_rate'] = np.random.beta(20, 2, n)
    data['same_srv_rate'] = np.random.beta(25, 1, n)
    data['dst_host_count'] = np.random.poisson(255, n)
    data['dst_host_srv_count'] = np.random.poisson(200, n)
    data['dst_host_serror_rate'] = np.random.beta(20, 2, n)
    data['duration'] = np.random.exponential(1, n)
    data['logged_in'] = np.zeros(n)
    return data

def generate_probe(n=2000):
    np.random.seed(456)
    data = generate_normal(n)
    data['count'] = np.random.poisson(100, n)
    data['srv_count'] = np.random.poisson(5, n)
    data['diff_srv_rate'] = np.random.beta(10, 2, n)
    data['srv_diff_host_rate'] = np.random.beta(8, 2, n)
    data['dst_host_diff_srv_rate'] = np.random.beta(8, 2, n)
    data['duration'] = np.random.exponential(10, n)
    data['src_bytes'] = np.random.exponential(50, n)
    return data

def generate_r2l(n=1500):
    np.random.seed(789)
    data = generate_normal(n)
    data['num_failed_logins'] = np.random.choice([2, 3, 4, 5], n)
    data['logged_in'] = np.random.choice([0, 1], n, p=[0.6, 0.4])
    data['hot'] = np.random.exponential(5, n)
    data['duration'] = np.random.exponential(500, n)
    data['src_bytes'] = np.random.exponential(2000, n)
    data['num_access_files'] = np.random.randint(0, 4, n)
    return data

def generate_u2r(n=500):
    np.random.seed(101)
    data = generate_normal(n)
    data['root_shell'] = np.ones(n)
    data['su_attempted'] = np.random.choice([0, 1], n, p=[0.3, 0.7])
    data['num_root'] = np.random.randint(1, 10, n)
    data['num_shells'] = np.random.randint(1, 3, n)
    data['num_file_creations'] = np.random.randint(2, 8, n)
    data['num_compromised'] = np.random.randint(1, 5, n)
    data['hot'] = np.random.exponential(8, n)
    return data

def main():
    print("生成模拟网络流量数据集...")
    normal = generate_normal(5000); normal['label'] = 0
    dos = generate_dos(3000); dos['label'] = 1
    probe = generate_probe(2000); probe['label'] = 2
    r2l = generate_r2l(1500); r2l['label'] = 3
    u2r = generate_u2r(500); u2r['label'] = 4
    df = pd.concat([normal, dos, probe, r2l, u2r], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv('data/network_traffic.csv', index=False)
    X = df.drop('label', axis=1); y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    print(f"总数据量: {len(df)}")
    print(f"训练集: {len(train)}, 测试集: {len(test)}")
    print(f"类别分布:\n{df['label'].value_counts().sort_index().rename(ATTACK_MAP)}")
    print("数据已保存到 data/ 目录")

if __name__ == '__main__':
    main()
