# -*- coding: utf-8 -*-
"""
入侵检测预测器 - 支持NSL-KDD编码器
"""
import os, joblib, json
import numpy as np
import pandas as pd

LABEL_MAP = {0: 'normal', 1: 'dos', 2: 'probe', 3: 'r2l', 4: 'u2r'}
LABEL_CN = {
    'normal': '正常流量', 'dos': 'DOS拒绝服务攻击',
    'probe': '探测扫描攻击', 'r2l': '远程权限攻击', 'u2r': '提权攻击'
}
RISK_LEVEL = {
    'normal': ('安全', 'green'), 'dos': ('高危', 'red'),
    'probe': ('中危', 'orange'), 'r2l': ('高危', 'red'), 'u2r': ('严重', 'darkred'),
}

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


class IntrusionDetector:
    def __init__(self, model_dir='models'):
        self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        self.models = {
            'random_forest': joblib.load(os.path.join(model_dir, 'random_forest.pkl')),
            'gradient_boosting': joblib.load(os.path.join(model_dir, 'gradient_boosting.pkl')),
            'mlp': joblib.load(os.path.join(model_dir, 'mlp.pkl')),
        }
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        # Load label encoders if available (NSL-KDD)
        enc_path = os.path.join(model_dir, 'encoders.pkl')
        self.encoders = joblib.load(enc_path) if os.path.exists(enc_path) else None

    def _encode_features(self, df):
        """Encode categorical features if encoders exist"""
        df = df.copy()
        if self.encoders:
            for col in CATEGORICAL_COLS:
                if col in df.columns and col in self.encoders:
                    le = self.encoders[col]
                    # Handle unseen categories gracefully
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
        # Ensure numeric
        for col in self.feature_names:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df[self.feature_names]

    def predict(self, data, model_name='random_forest'):
        model = self.models[model_name]
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data, columns=self.feature_names)

        df = self._encode_features(df)

        if model_name == 'mlp':
            df = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)

        preds = model.predict(df)
        try:\n            probas = model.predict_proba(df)\n        except Exception:\n            probas = None

        results = []
        for i, pred in enumerate(preds):
            label = LABEL_MAP.get(int(pred), 'normal')
            risk, color = RISK_LEVEL.get(label, ('未知', 'gray'))
            result = {'prediction': label, 'label_cn': LABEL_CN.get(label, label),
                      'risk_level': risk, 'risk_color': color}
            if probas is not None:
                result['confidence'] = float(probas[i].max())
                result['probabilities'] = {LABEL_MAP.get(j, str(j)): float(p) for j, p in enumerate(probas[i])}
            results.append(result)
        return results

    def get_metrics(self):
        with open(os.path.join('models', 'metrics.json'), 'r') as f:
            return json.load(f)
