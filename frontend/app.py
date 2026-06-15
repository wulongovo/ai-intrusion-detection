# -*- coding: utf-8 -*-
"""AI入侵检测系统 - Streamlit前端 (含服务器日志检测)"""
import sys,os,time,json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd,numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.predictor import IntrusionDetector,LABEL_MAP,LABEL_CN
from src.utils.server_log_analyzer import ServerLogAnalyzer

st.set_page_config(page_title="AI入侵检测系统",page_icon="🛡️",layout="wide")
st.markdown("<style>.risk-safe{color:green;font-weight:bold;font-size:20px}.risk-warning{color:orange;font-weight:bold;font-size:20px}.risk-danger{color:red;font-weight:bold;font-size:20px}.risk-critical{color:darkred;font-weight:bold;font-size:20px}</style>",unsafe_allow_html=True)

@st.cache_resource
def load_detector(): return IntrusionDetector(model_dir='models')
def get_dataset_info():
    try:
        with open('models/dataset_info.json','r') as f: return json.load(f)
    except: return None

PRESETS={"正常Web浏览":{'duration':50,'protocol_type':'tcp','service':'http','flag':'S1','src_bytes':800,'dst_bytes':2000,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,'num_failed_logins':0,'logged_in':1,'num_compromised':0,'root_shell':0,'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,'num_access_files':0,'count':30,'srv_count':25,'serror_rate':0.01,'srv_serror_rate':0.01,'rerror_rate':0.02,'srv_rerror_rate':0.02,'same_srv_rate':0.95,'diff_srv_rate':0.03,'srv_diff_host_rate':0.1,'dst_host_count':100,'dst_host_srv_count':60,'dst_host_same_srv_rate':0.85,'dst_host_diff_srv_rate':0.05,'dst_host_serror_rate':0.01,'dst_host_srv_serror_rate':0.01,'dst_host_rerror_rate':0.02,'dst_host_srv_rerror_rate':0.02},"Neptune DOS":{'duration':0,'protocol_type':'tcp','service':'private','flag':'S0','src_bytes':0,'dst_bytes':0,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,'num_failed_logins':0,'logged_in':0,'num_compromised':0,'root_shell':0,'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,'num_access_files':0,'count':511,'srv_count':511,'serror_rate':1.0,'srv_serror_rate':1.0,'rerror_rate':0.0,'srv_rerror_rate':0.0,'same_srv_rate':1.0,'diff_srv_rate':0.0,'srv_diff_host_rate':0.0,'dst_host_count':255,'dst_host_srv_count':255,'dst_host_same_srv_rate':1.0,'dst_host_diff_srv_rate':0.0,'dst_host_serror_rate':1.0,'dst_host_srv_serror_rate':1.0,'dst_host_rerror_rate':0.0,'dst_host_srv_rerror_rate':0.0},"端口扫描":{'duration':0,'protocol_type':'tcp','service':'private','flag':'REJ','src_bytes':0,'dst_bytes':0,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,'num_failed_logins':0,'logged_in':0,'num_compromised':0,'root_shell':0,'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,'num_access_files':0,'count':229,'srv_count':10,'serror_rate':0.0,'srv_serror_rate':0.0,'rerror_rate':1.0,'srv_rerror_rate':1.0,'same_srv_rate':0.04,'diff_srv_rate':0.06,'srv_diff_host_rate':0.0,'dst_host_count':255,'dst_host_srv_count':1,'dst_host_same_srv_rate':0.0,'dst_host_diff_srv_rate':0.06,'dst_host_serror_rate':0.0,'dst_host_srv_serror_rate':0.0,'dst_host_rerror_rate':1.0,'dst_host_srv_rerror_rate':1.0},"Root提权":{'duration':300,'protocol_type':'tcp','service':'telnet','flag':'SF','src_bytes':5000,'dst_bytes':10000,'land':0,'wrong_fragment':0,'urgent':1,'hot':8,'num_failed_logins':0,'logged_in':1,'num_compromised':3,'root_shell':1,'su_attempted':1,'num_root':8,'num_file_creations':5,'num_shells':1,'num_access_files':3,'count':10,'srv_count':8,'serror_rate':0.0,'srv_serror_rate':0.0,'rerror_rate':0.0,'srv_rerror_rate':0.0,'same_srv_rate':0.95,'diff_srv_rate':0.02,'srv_diff_host_rate':0.02,'dst_host_count':50,'dst_host_srv_count':25,'dst_host_same_srv_rate':0.9,'dst_host_diff_srv_rate':0.02,'dst_host_serror_rate':0.0,'dst_host_srv_serror_rate':0.0,'dst_host_rerror_rate':0.0,'dst_host_srv_rerror_rate':0.0}}

def show_dataset_overview():
    st.header("📊 数据集概览")
    info=get_dataset_info()
    if info:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("数据集",info.get('name','Unknown')); c2.metric("训练样本",f"{info.get('train_samples',0):,}")
        c3.metric("测试样本",f"{info.get('test_samples',0):,}"); c4.metric("特征数",info.get('features',0))
        cats=info.get('categories',{})
        if cats: st.subheader("类别分布"); st.bar_chart(pd.Series(cats))
    else: st.info("请先运行 python src/training/train_nsl_kdd.py")

def show_realtime_detection():
    st.header("🔍 实时流量检测")
    try: detector=load_detector()
    except Exception as e: st.error(f"模型加载失败: {e}"); return
    model=st.selectbox("模型",['random_forest','gradient_boosting','mlp'],format_func=lambda x:{'random_forest':'🌲 随机森林','gradient_boosting':'🚀 梯度提升','mlp':'🧠 MLP'}[x])
    scenario=st.selectbox("场景",["自定义输入"]+list(PRESETS.keys()))
    defaults=PRESETS.get(scenario,{})
    with st.form("detect"):
        cols=st.columns(4); user_input={}
        for i,feat in enumerate(detector.feature_names):
            with cols[i%4]:
                d=defaults.get(feat,0.0)
                if isinstance(d,str): user_input[feat]=st.text_input(feat,value=d,key=feat)
                else: user_input[feat]=st.number_input(feat,value=float(d),step=0.01,format="%.4f",key=feat)
        submitted=st.form_submit_button("🔍 检测",use_container_width=True)
    if submitted:
        with st.spinner("分析中..."): result=detector.predict(user_input,model)[0]
        st.markdown("---"); st.subheader("检测结果")
        c1,c2,c3=st.columns(3)
        with c1:
            rm={'green':'risk-safe','orange':'risk-warning','red':'risk-danger','darkred':'risk-critical'}
            st.markdown(f'<p class="{rm.get(result["risk_color"],"risk-safe")}">结果: {result["label_cn"]}</p>',unsafe_allow_html=True)
        with c2: st.metric("风险等级",result['risk_level'])
        with c3:
            if 'confidence' in result: st.metric("置信度",f"{result['confidence']:.1%}")
        if 'probabilities' in result:
            pdf=pd.DataFrame({'类别':[LABEL_CN.get(k,k) for k in result['probabilities']],'概率':list(result['probabilities'].values())})
            st.bar_chart(pdf.set_index('类别'))

def show_server_log_detection():
    st.header("🖥️ 服务器日志入侵检测")
    st.markdown("上传服务器日志，自动检测SSH暴力破解、SQL注入、端口扫描等入侵行为")
    analyzer=ServerLogAnalyzer()
    log_type=st.selectbox("日志类型",["SSH登录日志 (auth.log / secure)","Web访问日志 (nginx / apache access.log)","防火墙日志 (iptables / firewalld)"])
    st.markdown("---")
    upload_method=st.radio("输入方式",["上传文件","粘贴日志内容"],horizontal=True)
    log_content=None
    if upload_method=="上传文件":
        uploaded=st.file_uploader("选择日志文件",type=['log','txt'],key='log_upload')
        if uploaded: log_content=uploaded.read().decode('utf-8',errors='ignore'); st.info(f"已读取 {len(log_content)} 字符")
    else:
        log_content=st.text_area("粘贴日志内容",height=200,placeholder="将服务器日志粘贴到这里...")
    if log_content and st.button("🔍 开始分析",use_container_width=True):
        import tempfile
        tmp=tempfile.NamedTemporaryFile(mode='w',suffix='.log',delete=False,encoding='utf-8')
        tmp.write(log_content); tmp.close()
        with st.spinner("AI分析中..."):
            if "SSH" in log_type or "登录" in log_type: alerts,summary=analyzer.parse_auth_log(tmp.name)
            elif "Web" in log_type or "访问" in log_type: alerts,summary=analyzer.parse_access_log(tmp.name)
            else: alerts,summary=analyzer.parse_firewall_log(tmp.name)
        os.unlink(tmp.name)
        if "error" in summary: st.error(summary["error"]); return
        st.markdown("---"); st.subheader("📊 分析概览")
        if "SSH" in log_type or "登录" in log_type:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("失败登录",summary.get('ssh_failed',0)); c2.metric("成功登录",summary.get('ssh_success',0))
            c3.metric("攻击IP数",summary.get('attacking_ips',0)); c4.metric("告警数",summary.get('alert_count',0))
            if summary.get('top_attackers'): st.subheader("🔴 攻击IP排行"); st.dataframe(pd.DataFrame(summary['top_attackers'],columns=['IP','尝试次数']),use_container_width=True)
            if summary.get('top_failed_users'): st.subheader("🔴 被试用户名"); st.dataframe(pd.DataFrame(summary['top_failed_users'],columns=['用户名','失败次数']),use_container_width=True)
        elif "Web" in log_type or "访问" in log_type:
            c1,c2,c3=st.columns(3); c1.metric("总请求",f"{summary.get('total_requests',0):,}"); c2.metric("独立IP",summary.get('unique_ips',0)); c3.metric("告警",summary.get('alert_count',0))
            if summary.get('attack_types'): st.subheader("🔴 攻击类型"); st.bar_chart(pd.DataFrame(summary['attack_types'],columns=['类型','次数']).set_index('类型'))
            if summary.get('top_ips'): st.subheader("🔴 高频IP"); st.dataframe(pd.DataFrame(summary['top_ips'],columns=['IP','请求数']),use_container_width=True)
        else:
            c1,c2,c3=st.columns(3); c1.metric("总拦截",summary.get('total_blocked',0)); c2.metric("独立IP",summary.get('unique_ips',0)); c3.metric("告警",summary.get('alert_count',0))
            if summary.get('top_blocked'): st.subheader("🔴 被封IP"); st.dataframe(pd.DataFrame(summary['top_blocked'],columns=['IP','拦截次数']),use_container_width=True)
        if alerts:
            st.markdown("---"); st.subheader(f"🚨 告警详情 (共{len(alerts)}条)")
            risk_order={'严重':0,'高危':1,'中危':2,'低危':3}
            for a in sorted(alerts,key=lambda x:risk_order.get(x.get('risk',''),9))[:50]:
                r=a.get('risk','未知'); t=a.get('type',''); d=a.get('detail','')
                if r=='严重': st.error(f"🔴 [{r}] {t}: {d}")
                elif r=='高危': st.error(f"🟠 [{r}] {t}: {d}")
                elif r=='中危': st.warning(f"🟡 [{r}] {t}: {d}")
                else: st.info(f"⚪ [{r}] {t}: {d}")
            if len(alerts)>50: st.info(f"还有{len(alerts)-50}条未显示")
            rpt=json.dumps(analyzer.generate_report(),ensure_ascii=False,indent=2)
            st.download_button("📥 下载报告",rpt.encode('utf-8'),"intrusion_report.json","application/json")
        else: st.success("✅ 未检测到入侵行为")
    with st.expander("📖 如何获取服务器日志"):
        st.markdown("""**SSH日志:**
```bash
scp root@服务器IP:/var/log/auth.log ./auth.log
```
**Nginx日志:**
```bash
scp root@服务器IP:/var/log/nginx/access.log ./access.log
```
**防火墙日志:**
```bash
scp root@服务器IP:/var/log/messages ./firewall.log
```""")

def show_batch_detection():
    st.header("📁 批量流量检测")
    try: detector=load_detector()
    except Exception as e: st.error(f"模型加载失败: {e}"); return
    model=st.selectbox("模型",['random_forest','gradient_boosting','mlp'],format_func=lambda x:{'random_forest':'🌲 RF','gradient_boosting':'🚀 GB','mlp':'🧠 MLP'}[x],key='bm')
    uploaded=st.file_uploader("上传CSV",type=['csv'])
    if uploaded:
        df=pd.read_csv(uploaded); st.dataframe(df.head())
        if st.button("批量检测",use_container_width=True):
            with st.spinner("分析中..."): results=detector.predict(df,model)
            rdf=pd.DataFrame(results); total=len(results); atk=sum(1 for r in results if r['prediction']!='normal')
            c1,c2,c3,c4=st.columns(4); c1.metric("总流量",total); c2.metric("攻击",atk); c3.metric("正常",total-atk); c4.metric("攻击率",f"{atk/total:.1%}")
            st.bar_chart(rdf['label_cn'].value_counts())
            ddf=df.copy(); ddf['检测结果']=rdf['label_cn']; ddf['风险']=rdf['risk_level']
            st.dataframe(ddf)
            st.download_button("下载结果",ddf.to_csv(index=False).encode('utf-8-sig'),"results.csv","text/csv")

def show_model_evaluation():
    st.header("📈 模型评估")
    if not os.path.exists('models/metrics.json'): st.warning("请先训练模型"); return
    with open('models/metrics.json') as f: metrics=json.load(f)
    mn={'random_forest':'🌲 随机森林','gradient_boosting':'🚀 梯度提升','mlp':'🧠 MLP'}
    cols=st.columns(3)
    for i,(n,m) in enumerate(metrics.items()):
        with cols[i]: st.markdown(f"### {mn.get(n,n)}"); st.metric("准确率",f"{m['accuracy']:.2%}"); st.metric("耗时",f"{m['time']:.1f}s")
    fig,ax=plt.subplots(figsize=(8,4))
    names=[mn.get(n,n) for n in metrics.keys()]; accs=[m['accuracy'] for m in metrics.values()]
    bars=ax.barh(names,accs,color=['#2ecc71','#3498db','#e74c3c']); ax.set_xlim(0.6,1.0)
    for bar,acc in zip(bars,accs): ax.text(bar.get_width()-0.05,bar.get_y()+bar.get_height()/2,f'{acc:.2%}',ha='center',va='center',color='white',fontweight='bold')
    st.pyplot(fig)

def show_about():
    st.header("ℹ️ 系统说明")
    st.markdown("""
    ### AI入侵检测系统

    **功能模块:**
    | 模块 | 说明 |
    |------|------|
    | 📊 数据集概览 | NSL-KDD数据分布 |
    | 🔍 实时检测 | 手动输入特征测试 |
    | 🖥️ 服务器日志检测 | **上传日志自动检测入侵** |
    | 📁 批量检测 | CSV批量分析 |
    | 📈 模型评估 | 三模型对比 |

    **服务器日志检测能力:**
    | 攻击类型 | 检测方法 | 风险 |
    |----------|----------|------|
    | SSH暴力破解 | 同一IP失败≥5次 | 高危 |
    | SQL注入 | union select等 | 高危 |
    | XSS攻击 | script标签等 | 高危 |
    | 目录遍历 | ../等 | 高危 |
    | CC攻击 | 单IP≥1000次 | 高危 |
    | 端口扫描 | 多端口探测 | 高危 |
    | Root直接登录 | auth.log检测 | 中危 |
    | 提权尝试 | sudo危险命令 | 严重 |
    """)

def main():
    st.title("🛡️ AI入侵检测系统")
    info=get_dataset_info(); ds=info.get('name','') if info else ''
    st.markdown(f"{'真实' if 'KDD' in ds else '模拟'}网络流量检测 | 5类攻击识别 | 服务器日志分析")
    st.sidebar.title("功能导航")
    page=st.sidebar.radio("选择功能",["📊 数据集概览","🔍 实时检测","🖥️ 服务器日志检测","📁 批量检测","📈 模型评估","ℹ️ 系统说明"])
    if page=="📊 数据集概览": show_dataset_overview()
    elif page=="🔍 实时检测": show_realtime_detection()
    elif page=="🖥️ 服务器日志检测": show_server_log_detection()
    elif page=="📁 批量检测": show_batch_detection()
    elif page=="📈 模型评估": show_model_evaluation()
    elif page=="ℹ️ 系统说明": show_about()

if __name__=='__main__': main()
