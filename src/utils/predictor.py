# -*- coding: utf-8 -*-
"""入侵检测预测器"""
import os,joblib,json
import numpy as np,pandas as pd
LABEL_MAP={0:'normal',1:'dos',2:'probe',3:'r2l',4:'u2r'}
LABEL_CN={'normal':'正常流量','dos':'DOS拒绝服务攻击','probe':'探测扫描攻击','r2l':'远程权限攻击','u2r':'提权攻击'}
RISK_LEVEL={'normal':('安全','green'),'dos':('高危','red'),'probe':('中危','orange'),'r2l':('高危','red'),'u2r':('严重','darkred')}
CAT_COLS=['protocol_type','service','flag']
class IntrusionDetector:
    def __init__(self,model_dir='models'):
        self.feature_names=joblib.load(os.path.join(model_dir,'feature_names.pkl'))
        self.models={'random_forest':joblib.load(os.path.join(model_dir,'random_forest.pkl')),'gradient_boosting':joblib.load(os.path.join(model_dir,'gradient_boosting.pkl')),'mlp':joblib.load(os.path.join(model_dir,'mlp.pkl'))}
        self.scaler=joblib.load(os.path.join(model_dir,'scaler.pkl'))
        ep=os.path.join(model_dir,'encoders.pkl')
        self.encoders=joblib.load(ep) if os.path.exists(ep) else None
    def _encode(self,df):
        df=df.copy()
        if self.encoders:
            for c in CAT_COLS:
                if c in df.columns and c in self.encoders:
                    le=self.encoders[c]
                    df[c]=df[c].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    df[c]=le.transform(df[c])
        for c in self.feature_names:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
        return df[self.feature_names]
    def predict(self,data,model_name='random_forest'):
        model=self.models[model_name]
        if isinstance(data,dict): df=pd.DataFrame([data])
        elif isinstance(data,pd.DataFrame): df=data.copy()
        else: df=pd.DataFrame(data,columns=self.feature_names)
        df=self._encode(df)
        if model_name=='mlp': df=pd.DataFrame(self.scaler.transform(df),columns=self.feature_names)
        preds=model.predict(df)
        try: probas=model.predict_proba(df)
        except: probas=None
        results=[]
        for i,pred in enumerate(preds):
            label=LABEL_MAP.get(int(pred),'normal'); risk,color=RISK_LEVEL.get(label,('未知','gray'))
            r={'prediction':label,'label_cn':LABEL_CN.get(label,label),'risk_level':risk,'risk_color':color}
            if probas is not None: r['confidence']=float(probas[i].max()); r['probabilities']={LABEL_MAP.get(j,str(j)):float(x) for j,x in enumerate(probas[i])}
            results.append(r)
        return results
