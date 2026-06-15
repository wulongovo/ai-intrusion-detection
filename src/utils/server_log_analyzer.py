# -*- coding: utf-8 -*-
"""服务器日志入侵检测模块 - SSH/Web/防火墙"""
import re,os
from datetime import datetime
from collections import Counter,defaultdict
class ServerLogAnalyzer:
    def __init__(self):
        self.alerts=[]
        self.stats=defaultdict(int)
    def parse_auth_log(self,log_path):
        alerts=[]; ip_att=defaultdict(list); fail_usr=defaultdict(int); success=[]
        p_fail=re.compile(r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sshd\[\d+\]:\s+Failed password for (?:invalid user )?(\S+) from (\S+) port (\d+)')
        p_ok=re.compile(r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sshd\[\d+\]:\s+Accepted (?:password|publickey) for (\S+) from (\S+) port (\d+)')
        p_inv=re.compile(r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sshd\[\d+\]:\s+Invalid user (\S+) from (\S+)')
        p_sudo=re.compile(r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sudo:\s+(\S+)\s+:\s+.*COMMAND=(.+)')
        if not os.path.exists(log_path): return [],{"error":f"文件不存在: {log_path}"}
        with open(log_path,'r',encoding='utf-8',errors='ignore') as f:
            for line in f:
                m=p_fail.search(line)
                if m: ts,user,ip,port=m.groups(); ip_att[ip].append(ts); fail_usr[user]+=1; self.stats['ssh_failed']+=1; continue
                m=p_inv.search(line)
                if m: ts,user,ip=m.groups(); ip_att[ip].append(ts); fail_usr[user]+=1; self.stats['ssh_invalid']+=1; continue
                m=p_ok.search(line)
                if m: ts,user,ip,port=m.groups(); success.append({'time':ts,'user':user,'ip':ip}); self.stats['ssh_ok']+=1; continue
                m=p_sudo.search(line)
                if m:
                    ts,user,cmd=m.groups()
                    if any(d in cmd for d in ['passwd','chmod 777','visudo','/etc/shadow','useradd']):
                        alerts.append({'time':ts,'type':'提权尝试','risk':'严重','detail':f'{user} 执行: {cmd.strip()[:80]}'})
        for ip,times in ip_att.items():
            if len(times)>=5: alerts.append({'type':'SSH暴力破解','risk':'高危','detail':f'IP {ip} 尝试{len(times)}次','ip':ip,'count':len(times)})
        for u,c in fail_usr.items():
            if c>=10: alerts.append({'type':'高频失败用户','risk':'中危','detail':f'用户 {u} 失败{c}次'})
        for lg in success:
            if lg['user']=='root': alerts.append({'type':'Root直接登录','risk':'中危','detail':f"Root从{lg['ip']}登录({lg['time']})"})
        summary={'ssh_failed':self.stats['ssh_failed'],'ssh_success':self.stats['ssh_ok'],'attacking_ips':len(ip_att),'alert_count':len(alerts),'top_attackers':Counter({ip:len(t) for ip,t in ip_att.items()}).most_common(10),'top_failed_users':Counter(fail_usr).most_common(10)}
        self.alerts.extend(alerts); return alerts,summary
    def parse_access_log(self,log_path):
        alerts=[]; ip_req=defaultdict(int); status_codes=Counter()
        atk={'SQL注入':re.compile(r"(union\s+select|or\s+1=1|drop\s+table|insert\s+into|delete\s+from|exec\s*\(|concat\(|benchmark\(|sleep\(|load_file|into\s+outfile)",re.I),'XSS攻击':re.compile(r"(<script|javascript:|onerror=|onload=|alert\(|document\.cookie|eval\()",re.I),'目录遍历':re.compile(r"(\.\./|\.\.\\|/etc/passwd|/etc/shadow|/proc/self|win\.ini)",re.I),'命令注入':re.compile(r"(;\s*ls|;\s*cat|;\s*id|;\s*whoami|\|\s*ls|\|\s*cat|`id`|`whoami`)",re.I),'WebShell':re.compile(r"(\.php\?cmd=|eval\(|system\(|passthru|shell_exec|exec\(|base64_decode)",re.I),'扫描器':re.compile(r"(nikto|sqlmap|nmap|masscan|dirbuster|gobuster|wpscan|burpsuite|acunetix|nessus)",re.I),'敏感文件':re.compile(r"(/\.env|/wp-config|/config\.php|/admin|/phpmyadmin|/\.git|/backup|/dump\.sql)",re.I)}
        pat=re.compile(r'(\S+)\s+-\s+(\S+)\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"')
        if not os.path.exists(log_path): return [],{"error":f"文件不存在: {log_path}"}
        with open(log_path,'r',encoding='utf-8',errors='ignore') as f:
            for line in f:
                m=pat.match(line.strip())
                if not m: continue
                ip,user,ts,req,st,sz,ref,ua=m.groups(); ip_req[ip]+=1; status_codes[st]+=1
                for atype,ap in atk.items():
                    if ap.search(req) or ap.search(ua):
                        alerts.append({'time':ts,'type':atype,'risk':'高危','ip':ip,'detail':f'{atype}来自{ip}: {req[:100]}'})
                        self.stats[f'web_{atype}']+=1; break
        for ip,c in ip_req.items():
            if c>=1000: alerts.append({'type':'CC攻击','risk':'高危','ip':ip,'count':c,'detail':f'IP {ip} 请求{c}次'})
        summary={'total_requests':sum(ip_req.values()),'unique_ips':len(ip_req),'status_codes':dict(status_codes.most_common(10)),'top_ips':Counter(ip_req).most_common(10),'alert_count':len(alerts),'attack_types':Counter(a['type'] for a in alerts).most_common()}
        self.alerts.extend(alerts); return alerts,summary
    def parse_firewall_log(self,log_path):
        alerts=[]; ip_blk=defaultdict(int); port_scan=defaultdict(set)
        pat=re.compile(r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+kernel:.*SRC=(\S+)\s+DST=(\S+)\s+.*DPT=(\d+)')
        if not os.path.exists(log_path): return [],{"error":f"文件不存在: {log_path}"}
        with open(log_path,'r',encoding='utf-8',errors='ignore') as f:
            for line in f:
                m=pat.search(line)
                if m: ts,src,dst,port=m.groups(); ip_blk[src]+=1; port_scan[src].add(port)
        for ip,ports in port_scan.items():
            if len(ports)>=20: alerts.append({'type':'端口扫描','risk':'高危','ip':ip,'detail':f'IP {ip} 扫描{len(ports)}个端口'})
        for ip,c in ip_blk.items():
            if c>=100: alerts.append({'type':'高频拦截IP','risk':'中危','ip':ip,'detail':f'IP {ip} 被拦{c}次'})
        summary={'total_blocked':sum(ip_blk.values()),'unique_ips':len(ip_blk),'top_blocked':Counter(ip_blk).most_common(10),'alert_count':len(alerts)}
        self.alerts.extend(alerts); return alerts,summary
    def generate_report(self):
        return {'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'total_alerts':len(self.alerts),'risk_summary':dict(Counter(a.get('risk','未知') for a in self.alerts)),'type_summary':dict(Counter(a.get('type','未知') for a in self.alerts)),'alerts':self.alerts[:100]}
