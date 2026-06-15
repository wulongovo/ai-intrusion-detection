# -*- coding: utf-8 -*-
"""NSL-KDD 真实数据集训练"""
import sys,os,json,time,joblib
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd,numpy as np
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report,accuracy_score
import warnings; warnings.filterwarnings('ignore')

FEATURE_COLS=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
CAT_COLS=['protocol_type','service','flag']
ALL_COLS=FEATURE_COLS+['attack_label','difficulty']
ATTACK_CAT={'normal':'normal','back':'dos','land':'dos','neptune':'dos','pod':'dos','smurf':'dos','teardrop':'dos','mailbomb':'dos','apache2':'dos','processtable':'dos','udpstorm':'dos','ipsweep':'probe','nmap':'probe','portsweep':'probe','satan':'probe','mscan':'probe','saint':'probe','ftp_write':'r2l','guess_passwd':'r2l','imap':'r2l','multihop':'r2l','phf':'r2l','spy':'r2l','warezclient':'r2l','warezmaster':'r2l','snmpgetattack':'r2l','named':'r2l','xlock':'r2l','xsnoop':'r2l','sendmail':'r2l','httptunnel':'r2l','worm':'r2l','buffer_overflow':'u2r','loadmodule':'u2r','perl':'u2r','rootkit':'u2r','xterm':'u2r','ps':'u2r','sqlattack':'u2r'}
CAT_MAP={'normal':0,'dos':1,'probe':2,'r2l':3,'u2r':4}
LABEL_CN={0:'正常流量',1:'DOS拒绝服务攻击',2:'探测扫描攻击',3:'远程权限攻击',4:'提权攻击'}

def main():
    os.makedirs('models',exist_ok=True)
    train_df=pd.read_csv(os.path.join('data','KDDTrain+.txt'),header=None,names=ALL_COLS)
    test_path=os.path.join('data','KDDTest+.txt')
    test_df=pd.read_csv(test_path,header=None,names=ALL_COLS) if os.path.exists(test_path) else train_df.sample(frac=0.2,random_state=42)
    for df in [train_df,test_df]:
        df.dropna(subset=['attack_label'],inplace=True)
        for c in FEATURE_COLS: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
        df['label']=df['attack_label'].map(ATTACK_CAT).map(CAT_MAP).fillna(0).astype(int)
    encoders={}
    for c in CAT_COLS:
        le=LabelEncoder(); le.fit(pd.concat([train_df[c],test_df[c]]))
        train_df[c]=le.transform(train_df[c]); test_df[c]=le.transform(test_df[c]); encoders[c]=le
    Xtr,ytr=train_df[FEATURE_COLS],train_df['label']
    Xte,yte=test_df[FEATURE_COLS],test_df['label']
    print(f"训练集:{len(train_df)} 测试集:{len(test_df)} 特征:{len(FEATURE_COLS)}")
    results={}
    for name,model in [('random_forest',RandomForestClassifier(n_estimators=200,max_depth=30,random_state=42,n_jobs=-1)),('gradient_boosting',HistGradientBoostingClassifier(max_iter=150,max_depth=8,random_state=42)),('mlp',MLPClassifier(hidden_layer_sizes=(256,128,64),max_iter=300,random_state=42,early_stopping=True))]:
        print(f"\n训练 {name}...")
        t0=time.time()
        if name=='mlp':
            sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
            model.fit(Xtr_s,ytr); acc=accuracy_score(yte,model.predict(Xte_s)); joblib.dump(sc,'models/scaler.pkl')
        else:
            model.fit(Xtr,ytr); acc=accuracy_score(yte,model.predict(Xte))
        elapsed=time.time()-t0
        results[name]={'accuracy':acc,'time':elapsed}
        print(f"准确率:{acc:.4f} 耗时:{elapsed:.1f}s")
        joblib.dump(model,f'models/{name}.pkl')
    joblib.dump(FEATURE_COLS,'models/feature_names.pkl')
    joblib.dump(encoders,'models/encoders.pkl')
    with open('models/metrics.json','w') as f: json.dump(results,f,indent=2)
    ds={'name':'NSL-KDD','train_samples':int(len(train_df)),'test_samples':int(len(test_df)),'features':len(FEATURE_COLS),'categories':{LABEL_CN[k]:int(v) for k,v in ytr.value_counts().sort_index().items()}}
    with open('models/dataset_info.json','w') as f: json.dump(ds,f,indent=2,ensure_ascii=False)
    print("\n训练完成!")
    for n,m in results.items(): print(f"  {n}: {m['accuracy']:.4f} ({m['time']:.1f}s)")
if __name__=='__main__': main()
