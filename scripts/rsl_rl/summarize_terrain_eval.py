"""Summarize one complete paired traversal evaluation (any supported model set)."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ame-terrain-eval-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args=parser.parse_args()
p=args.directory.resolve()
records=json.loads((p/'trials.json').read_text())
protocol=json.loads((p/'protocol.json').read_text())
pairing=json.loads((p/'pairing.json').read_text())
manifest=json.loads((p/'terrain_manifest.json').read_text())
models=list(protocol['checkpoint_metadata'])
terrains=list(dict.fromkeys(x['terrain'] for x in manifest))
difficulties=protocol['difficulties']
width=protocol.get('half_width',1.0)
per_model=protocol['num_envs']*len(protocol['seeds'])
expected=per_model*len(models)
assert len(records)==expected,(len(records),expected)
assert len(pairing)==len(models)*len(protocol['seeds'])
assert all(r['max_initial_state_delta']<=1e-5 and r['max_initial_observation_delta']<=1e-4 for r in pairing)
keys=[(r['model'],r['seed'],r['env_id']) for r in records]
assert len(set(keys))==len(keys)

def stats(rows):
    wins=[r for r in rows if r['success']]
    return dict(n=len(rows),successes=len(wins),success_rate=len(wins)/len(rows),
                outcomes=dict(Counter(r['outcome'] for r in rows)),
                mean_success_seconds=float(np.mean([r['seconds'] for r in wins])) if wins else None,
                mean_velocity_error_m_s=float(np.mean([r['mean_velocity_error_m_s'] for r in rows])),
                mean_success_velocity_error_m_s=float(np.mean([r['mean_velocity_error_m_s'] for r in wins])) if wins else None)

summary={}
for m in models:
    rows=[r for r in records if r['model']==m]
    summary[m]={'all':stats(rows),'terrains':{t:stats([r for r in rows if r['terrain']==t]) for t in terrains},
                'difficulties':{str(d):stats([r for r in rows if r['difficulty']==d]) for d in difficulties},
                'cells':{f'{t}:{d}':stats([r for r in rows if r['terrain']==t and r['difficulty']==d]) for t in terrains for d in difficulties},
                'seeds':{str(seed):stats([r for r in rows if r['seed']==seed]) for seed in protocol['seeds']}}
paired={}
for a,b in combinations(models,2):
    aa={(r['seed'],r['env_id']):r['success'] for r in records if r['model']==a}
    bb={(r['seed'],r['env_id']):r['success'] for r in records if r['model']==b}
    assert aa.keys()==bb.keys()
    paired[a+' vs '+b]=dict(a_only=sum(aa[k] and not bb[k] for k in aa),b_only=sum(bb[k] and not aa[k] for k in aa),
                            both_success=sum(aa[k] and bb[k] for k in aa),both_fail=sum(not aa[k] and not bb[k] for k in aa))
(p/'summary.json').write_text(json.dumps(dict(results=summary,paired=paired),ensure_ascii=False,indent=2))
ncols=min(3,len(models));nrows=math.ceil(len(models)/ncols)
fig,axes=plt.subplots(nrows,ncols,figsize=(4.7*ncols,4.8*nrows),layout='constrained',squeeze=False)
used=[]
for ax,m in zip(axes.flat,models):
    used.append(ax)
    values=np.array([[summary[m]['cells'][f'{t}:{d}']['success_rate']*100 for d in difficulties] for t in terrains])
    im=ax.imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_title(m);ax.set_xticks(range(len(difficulties)),[str(d) for d in difficulties]);ax.set_xlabel('Terrain difficulty parameter')
    ax.set_yticks(range(len(terrains)),terrains)
    for i in range(len(terrains)):
        for j in range(len(difficulties)):
            ax.text(j,i,f'{values[i,j]:.1f}%',ha='center',va='center',fontsize=10,color='white' if values[i,j]>60 else '#122334')
for ax in list(axes.flat)[len(models):]:ax.set_visible(False)
fig.colorbar(im,ax=used,label='Success rate (%)',shrink=.75)
fig.suptitle(f'Paired finetune traversal | 3.5 m forward within 10 s | lateral limit +/-{width:g} m\nSame geometry and initial states; clean observations, no pushes',fontsize=13)
fig.savefig(p/'success_rates.png',dpi=170)
labels={'pyramid_stairs':'正向金字塔楼梯','pyramid_stairs_inv':'反向金字塔楼梯','stakes1':'双列桩 stakes1','stakes2':'交错桩 stakes2','stakes3':'交错桩 stakes3','hf_gaps':'沟壑','stonebridge':'石桥','rails':'栏杆'}
def rate(x):return f"{100*x['success_rate']:.2f}%（{x['successes']}/{x['n']}）"
def group_table(title, groups, accessor):
    text=f'\n## {title}\n\n| 分组 | '+' | '.join(models)+' |\n|---|'+'---:|'*len(models)+'\n'
    for key,label in groups:
        text+='| '+label+' | '+' | '.join(rate(accessor(summary[m],key)) for m in models)+' |\n'
    return text

report=f'''# 配对地形通过率评估：{len(models)} 组检查点

评估日期：2026-09-22。执行环境：本地 NVIDIA RTX 5080、Isaac Sim GPU 物理仿真。当前通道横向限制为 **±{width:g} m**，共 {expected} 次尝试。本文为独立推理评测，不以训练回报推算成功率。

## 1. 测试对象

| 模型 | 检查点 | 保存的 iter | 结构 |
|---|---|---:|---|
'''
structures={'AME-LSIO':'LSIO + 稠密注意力；全局关闭，非 AME2','AME1':'普通 AME，无全局分支','AME2':'普通 AME，带全局 MLP/max-pool 分支','GLAD':'LSIO + 全局池化 + Top-K=32；联合训练','GLAD-CleanStopGrad':'同 GLAD；训练时 Critic 干净地形编码后 detach'}
for m in models:
    ck=protocol['checkpoint_metadata'][m]
    report+=f"| {m} | `{ck['path']}` | {ck['iter']} | {structures[m]} |\n"
report+='''
所有权重严格加载，检查点模式及 LSIO/GLAD 架构元数据通过校验。AME1/AME2 缺少完整训练配置与配套日志，因此这些具体模型的比较不等于等预算的纯网络消融。CriticCleanStopGrad 使用 Critic 干净地形；它不是此前复用 Actor 特征的 CriticStopGrad。推理仅调用 Actor，不计算价值函数；GLAD 使用确定性硬 Top-K。

## 2. 协议

'''
report+=f'''- 地形为 finetune 的 {len(terrains)} 类，难度参数 {difficulties}；参数不保证所有地形实际难度单调，例如 rails 高度与厚度同时变化。
- 每个地形×参数组合有 {protocol['variants']} 个实例槽位，每槽位 {protocol['replicas']} 个初始样本，初始状态种子为 {protocol['seeds']}。每模型 {per_model} 次；确定性地形槽位可能重复，不视为独立随机几何。
- 模型共享地形网格，每批初始机器人状态及所有初始观测逐元素配对；只记录每环境第一次尝试，在自动重置之前保存终止结果。
- 固定 vx=1 m/s、vy=0、目标航向 0，采用原航向反馈。关闭观测噪声、外推及物理参数随机化；使用 eval 模式动作均值。
- 从中央平台出发，10 秒内到达 x≥3.5 m，同时脚部有 >20 N 支撑、重力投影 z<−0.5；要求 |y|≤{width:g} m、−1≤x≤3.9 m，并且没有非法身体接触。非法接触或越界与成功同一步发生时，失败优先；超时不算成功。
- 这是从中央平台到边缘的 3.5 m 单向路线，不是完整 8 m 穿越。总体结果对地形和参数等权，未按训练地形占比加权。

## 3. 总体结果

| 模型 | 通过率 | 成功样本平均用时 | 全部尝试平均速度误差 |
|---|---:|---:|---:|
'''
for m in models:
    x=summary[m]['all'];duration=f"{x['mean_success_seconds']:.3f} s" if x['mean_success_seconds'] is not None else '无成功样本'
    report+=f"| {m} | {rate(x)} | {duration} | {x['mean_velocity_error_m_s']:.3f} m/s |\n"
report+='\n速度误差是逐控制步二维机体速度误差模长的均值；失败会截短统计，成功用时也有样本选择影响，均需结合通过率解释。\n'
report+=group_table('4. 分地形结果',[(t,labels[t]) for t in terrains],lambda s,k:s['terrains'][k])
report+='\n![地形与难度通过率](success_rates.png)\n'
report+=group_table('5. 分难度参数结果',[(str(d),str(d)) for d in difficulties],lambda s,k:s['difficulties'][k])
report+='\n## 6. 失败类型\n\n| 模型 | 非法接触 | 越界 | 超时 |\n|---|---:|---:|---:|\n'
for m in models:
    o=summary[m]['all']['outcomes'];report+='| '+m+' | '+' | '.join(str(o.get(k,0)) for k in ['illegal_contact','out_of_bounds','timeout'])+' |\n'
report+='\n非法接触沿用项目躯干、肩、髋、膝、肘、腰、骨盆等身体部位的接触终止规则；横向越界不等于跌倒。\n'
report+=group_table('7. 初始状态种子敏感性',[(str(seed),str(seed)) for seed in protocol['seeds']],lambda s,k:s['seeds'][k])
report+='\n| 配对 A / B | 仅 A 成功 | 仅 B 成功 | 两者成功 | 两者失败 |\n|---|---:|---:|---:|---:|\n'
for name,x in paired.items():report+='| '+name+' | '+' | '.join(str(x[k]) for k in ['a_only','b_only','both_success','both_fail'])+' |\n'
report+='''
## 8. 结论边界与复现

这些是初始状态种子，不是多种子重新训练。多个尝试共享几何，不能以普通二项置信区间代表跨地形泛化；接近的样本比例也不证明稳定优劣。带噪声、外推、高速、侧向/反向路线尚未评估。两 GLAD 模型同结构，差异来自各自训练所得 Actor；当前消融还包含冻结 Critic 本体投影与禁止 Critic 写 BN 状态，不能把差距全部归因于共享价值梯度。

- [逐次记录](trials.json)、[聚合统计](summary.json)、[协议及检查点 SHA256](protocol.json)。
- [初始条件配对](pairing.json)、[地形网格哈希](terrain_manifest.json)、[环境配置](environment.yaml)。

'''
report+=f'''```bash
cd /home/kqw/AME_Locomotion
LD_PRELOAD=/home/kqw/miniconda3/envs/lab/lib/libstdc++.so.6 \\
/home/kqw/miniconda3/envs/lab/bin/python scripts/rsl_rl/evaluate_terrain.py \\
  --headless --device cuda:0 --half_width {width:g} \\
  --models {' '.join(models)} --output output/terrain_eval_rerun
/home/kqw/miniconda3/envs/lab/bin/python scripts/rsl_rl/summarize_terrain_eval.py output/terrain_eval_rerun
```
'''
(p/'evaluation_report.md').write_text(report)
import markdown,base64
body=markdown.markdown(report,extensions=['tables','fenced_code'])
body=body.replace('src="success_rates.png"','src="data:image/png;base64,'+base64.b64encode((p/'success_rates.png').read_bytes()).decode()+'"')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AME / GLAD 地形通过率评估</title><style>body{font:16px/1.85 system-ui,"Noto Sans CJK SC",sans-serif;color:#193044;background:#eef2f5;margin:0}main{max-width:1240px;margin:32px auto;background:white;padding:45px 50px}h1{font-size:30px}h2{margin-top:38px;border-bottom:2px solid #d6e7e0;padding-bottom:9px;font-size:23px}table{border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}th,td{border:1px solid #d6dee7;padding:9px;overflow-wrap:anywhere}th{background:#e8f2ed}img{width:100%;height:auto}code{overflow-wrap:anywhere;font-size:.87em}pre{background:#eef2f5;padding:16px;white-space:pre-wrap}a{color:#167363}@media(max-width:800px){main{margin:0;padding:20px}table{font-size:11px}td,th{padding:5px}}@media print{body{background:white;font-size:10pt}main{margin:0;padding:0}h2{break-after:avoid}tr,img{break-inside:avoid}}</style><main>'''+body+'</main></html>'
(p/'evaluation_report.html').write_text(html)
for m in models:print(m,summary[m]['all'])
print('Report:',p/'evaluation_report.html')
