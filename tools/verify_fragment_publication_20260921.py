"""Verify published hashes, V1, raw ledgers, paired statistics and unchanged gates."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.fragment_study import sha,analyze,verify_v1,STUDIES


def main():
    public=ROOT/'evidence/fragment_repair_20260921'
    for p,h in json.loads((public/'PUBLICATION_MANIFEST.json').read_text()).items():
        assert sha(ROOT/p)==h,p
    seeds=set(); studies={}
    for short,name in STUDIES.items():
        out=public/short;c=json.loads((ROOT/'config'/f'{name}.json').read_text())
        cohort=set(range(c['seed_start'],c['seed_start']+c['seed_count']))
        assert not cohort&seeds and not cohort&set(range(3900,3920));seeds|=cohort
        for p,h in json.loads((out/'SOURCE_FREEZE.json').read_text()).items():assert sha(ROOT/p)==h,p
        for p,h in json.loads((out/'MANIFEST.json').read_text()).items():assert sha(out/p)==h,p
        result=analyze(out,c,len(json.loads((out/'SCENARIOS.json').read_text())))
        saved=json.loads((out/'results.json').read_text())
        for k in result:assert result[k]==saved[k],(short,k)
        studies[short]=dict(episodes=result['raw_rows_recomputed'],gates={p:z['status'] for p,z in result['iterations'].items()})
    assert json.loads((public/'INVALIDATED_PREDECESSOR.json').read_text())['status']=='INVALID_FOR_PERFORMANCE_CLAIMS'
    print(json.dumps(dict(verified=True,studies=studies,distinct_development_seeds=len(seeds),v1_payloads=verify_v1(),neural_training=False),indent=2))


if __name__=='__main__': main()
