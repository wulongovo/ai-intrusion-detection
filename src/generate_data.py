# -*- coding: utf-8 -*-
"""生成模拟网络流量数据集"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
ATTACK_MAP = {0:'normal',1:'dos',2:'probe',3:'r2l',4:'u2r'}
def gen_normal(n=5000):
    np.random.seed(42)
    return pd.DataFrame({
        'duration':np.random.exponential(100,n),'protocol_type':np.random.choice([0,1,2],n,p=[0.7,0.2,0.1]),
        'service':np.random.randint(0,20,n),'flag':np.random.choice([0,1],n,p=[0.9,0.1]),
        'src_bytes':np.random.exponential(500,n),'dst_bytes':np.random.exponential(1000,n),
        'land':np.zeros(n),'wrong_fragment':np.random.choice([0,1],n,p=[0.98,0.02]),
        'urgent':np.zeros(n),'hot':np.random.exponential(1,n),
        'num_failed_logins':np.random.choice([0,1],n,p=[0.95,0.05]),'logged_in':np.ones(n),
        'num_compromised':np.zeros(n),'root_shell':np.zeros(n),'su_attempted':np.zeros(n),
        'num_root':np.random.choice([0,1],n,p=[0.9,0.1]),
        'num_file_creations':np.random.exponential(0.5,n),'num_shells':np.zeros(n),
        'num_access_files':np.random.exponential(0.3,n),
        'count':np.random.poisson(50,n),'srv_count':np.random.poisson(30,n),
        'serror_rate':np.random.beta(1,50,n),'srv_serror_rate':np.random.beta(1,50,n),
        'rerror_rate':np.random.beta(1,30,n),'srv_rerror_rate':np.random.beta(1,30,n),
        'same_srv_rate':np.random.beta(20,2,n),'diff_srv_rate':np.random.beta(1,20,n),
        'srv_diff_host_rate':np.random.beta(1,10,n),
        'dst_host_count':np.random.poisson(100,n),'dst_host_srv_count':np.random.poisson(50,n),
        'dst_host_same_srv_rate':np.random.beta(10,2,n),'dst_host_diff_srv_rate':np.random.beta(1,10,n),
        'dst_host_serror_rate':np.random.beta(1,50,n),'dst_host_srv_serror_rate':np.random.beta(1,50,n),
        'dst_host_rerror_rate':np.random.beta(1,30,n),'dst_host_srv_rerror_rate':np.random.beta(1,30,n),
    })
def main():
    normal=gen_normal(5000); normal['label']=0
    dos=gen_normal(3000); dos['label']=1; dos['count']=np.random.poisson(300,3000); dos['serror_rate']=np.random.beta(20,2,3000)
    probe=gen_normal(2000); probe['label']=2; probe['diff_srv_rate']=np.random.beta(10,2,2000)
    r2l=gen_normal(1500); r2l['label']=3; r2l['num_failed_logins']=np.random.choice([2,3,4,5],1500)
    u2r=gen_normal(500); u2r['label']=4; u2r['root_shell']=np.ones(500); u2r['num_root']=np.random.randint(1,10,500)
    df=pd.concat([normal,dos,probe,r2l,u2r],ignore_index=True).sample(frac=1,random_state=42).reset_index(drop=True)
    df.to_csv('data/network_traffic.csv',index=False)
    X,y=df.drop('label',axis=1),df['label']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    pd.concat([Xtr,ytr],axis=1).to_csv('data/train.csv',index=False)
    pd.concat([Xte,yte],axis=1).to_csv('data/test.csv',index=False)
    print(f"数据生成完成: {len(df)}条, 训练{len(Xtr)}, 测试{len(Xte)}")
if __name__=='__main__': main()
