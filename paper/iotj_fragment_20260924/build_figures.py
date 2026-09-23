from pathlib import Path
import gzip,json,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;R=P.parents[1];E=R/'evidence/fragment_repair_20260921'
(P/'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
colors=['#686868','#D88D29','#176B91','#75549A','#50996B']
names={'whole':'Whole retry','selective':'Selective retry','block':'Block ACK','prefix':'Block prefix','guard':'Completion guard'}
results={s:json.loads((E/s/'results.json').read_text()) for s in ('fixed','guard','shared')}
fig,axes=plt.subplots(1,2,figsize=(7,2.6))
for j,(metric,label) in enumerate([('delivery_ratio','Timely delivery (%)'),('packets_per_j','Delivered packets / charged J')]):
    for k,p in enumerate(('whole','selective','block')):
        vals=[results[s]['aggregate'][p][metric]*(100 if j==0 else 1) for s in ('fixed','guard','shared')]
        axes[j].bar(np.arange(3)+(k-1)*.24,vals,.22,label=names[p],color=colors[k])
    axes[j].set_xticks(range(3),['Single source\ncohort 1','Single source\ncohort 2','Shared head']);axes[j].set_ylabel(label);axes[j].grid(axis='y',alpha=.2)
axes[0].legend(loc='upper left',fontsize=7);fig.tight_layout();fig.savefig(P/'figures/aggregate.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(3.45,2.5))
rows=[]
for s,c,b in [('fixed','block','whole'),('fixed','block','selective'),('shared','block','whole'),('shared','block','selective'),('fixed','prefix','block'),('guard','guard','block')]:
    rows.append((s,c,b,results[s]['iterations'][c]['comparisons'][b]))
for j,metric in enumerate(('delivery','efficiency')):
    v=np.array([r[3][metric]['relative_gain']*100 for r in rows]);ci=np.array([r[3][metric]['paired_95_interval'] for r in rows])*100
    ax.errorbar(v,np.arange(len(rows))+(j-.5)*.19,xerr=[v-ci[:,0],ci[:,1]-v],fmt='o' if j==0 else 's',ms=3,capsize=2,color=colors[j+1],label='Delivery' if j==0 else 'Packets/J')
ax.axvline(0,color='gray',lw=.7);ax.set_yticks(range(len(rows)),['C1: block / whole','C1: block / selective','SH: block / whole','SH: block / selective','C1: prefix / block','C2: guard / block'],fontsize=7);ax.invert_yaxis();ax.set_xlabel('Relative change (%), descriptive 95% CI');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(P/'figures/effects.pdf');plt.close(fig)

scenarios=json.loads((E/'shared/SCENARIOS.json').read_text());raw=[]
for path in sorted((E/'shared').glob('seed_*.jsonl.gz')):
    with gzip.open(path,'rt') as f:
        for line in f:
            z=json.loads(line);raw.append({**z,**scenarios[z['scenario']]})
summary={};fig,axes=plt.subplots(2,3,figsize=(7,3.5))
for ax,field in zip(axes.flat,('sources','capacity_uj','harvest_uj','ttl','head_change','loss')):
    vals=sorted(set(z[field] for z in raw));summary[field]={}
    for k,p in enumerate(('whole','selective','block')):
        ys=[]
        for v in vals:
            ms=[z['metrics'] for z in raw if z[field]==v and z['policy']==p]
            a={key:float(np.mean([m[key] for m in ms])) for key in ms[0]}
            a['delivery_ratio']=a['delivered']/a['generated'];a['packets_per_j']=a['delivered']/(a['energy_uj']*1e-6)
            summary[field].setdefault(str(v),{})[p]=a;ys.append(100*a['delivery_ratio'])
        ax.plot(range(len(vals)),ys,'o-',ms=3,color=colors[k],label=names[p])
    labels=[str(v) for v in vals]
    if field=='capacity_uj': labels=[str(v//1000) for v in vals]
    if field=='head_change': labels=['Fixed','Switch']
    if field=='loss': labels=['Burst','Independent']
    ax.set_xticks(range(len(vals)),labels,fontsize=8);ax.set_xlabel({'sources':'Number of sources','capacity_uj':'Capacity (mJ)','harvest_uj':'Harvest (uJ/frame)','ttl':'Deadline (frames)','head_change':'Head schedule','loss':'Data erasure law'}[field]);ax.set_ylabel('Delivery (%)');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(P/'figures/factors.pdf');plt.close(fig)
(P/'derived_results.json').write_text(json.dumps(summary,indent=2))

fig,axes=plt.subplots(1,2,figsize=(7,2.5))
policies=('whole','selective','block');agg=results['shared']['aggregate'];bottom=np.zeros(3)
for k,(field,label) in enumerate([('background_uj','Background'),('control_uj','Sync/report'),('member_uj','Member data'),('forward_uj','Forward data'),('feedback_uj','Feedback/padding')]):
    y=np.array([agg[p][field]/1000 for p in policies]);axes[0].bar(range(3),y,bottom=bottom,color=colors[k],label=label);bottom+=y
axes[0].set_ylabel('Mean charged energy (mJ)');axes[0].legend(fontsize=6,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);bottom=np.zeros(3)
for k,(field,label) in enumerate([('delivered','Delivered'),('stale','Expired'),('overflow','Overflow'),('death','Death'),('pending_undelivered','Pending')]):
    y=np.array([100*agg[p][field]/agg[p]['generated'] for p in policies]);axes[1].bar(range(3),y,bottom=bottom,color=colors[k],label=label);bottom+=y
axes[1].set_ylabel('Generated-datagram outcome (%)');axes[1].legend(fontsize=6,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
for ax in axes:ax.set_xticks(range(3),['Whole','Selective','Block']);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(P/'figures/accounting.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(7,2.5));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
boxes=[(.1,1.7,2.1,.9,'IoT sources\nqueue cap: 2 each\nretained custody'),(3.2,1.7,2.4,.9,'Exogenous active head\nshared battery\n2 live reassemblies'),(7,1.7,2.1,.9,'Gateway / sink\nunique delivery\nreassembly bitmap'),(.1,.1,9,.8,'Paid sync and reports  →  EDF admission  →  member fragments + ACK\n→  forwarded fragments + ACK  →  fixed-boundary receipt')]
for x,y,w,h,label in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor='#eef4f7',edgecolor='#176B91',lw=.8));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
for x1,x2 in [(2.3,3.1),(5.7,6.9)]:
    ax.annotate('',xy=(x2,2.35),xytext=(x1,2.35),arrowprops=dict(arrowstyle='->',lw=1));ax.annotate('',xy=(x1,1.95),xytext=(x2,1.95),arrowprops=dict(arrowstyle='->',lw=1,color='#D88D29'))
ax.text(4.6,1.23,'Both battery-powered participants fund the full scheduled window',ha='center',fontsize=8)
fig.savefig(P/'figures/architecture.pdf');plt.close(fig)
print(json.dumps({f:{v:round(a['block']['delivery_ratio']*100,3) for v,a in values.items()} for f,values in summary.items()},indent=2))
