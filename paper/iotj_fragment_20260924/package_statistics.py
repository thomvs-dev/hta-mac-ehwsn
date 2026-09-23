from pathlib import Path
import hashlib,json,shutil,zipfile
P=Path(__file__).resolve().parent
O=P.parents[1]/'output/pdf'
stem='iotj_fragment_recovery_20260924_statistics'
pdf=O/(stem+'.pdf'); archive=O/(stem+'_source.zip')
shutil.copyfile(P/'main.pdf',pdf)
names=['main.tex','README.md','PLAN_AND_STYLE.md','EDITORIAL_REVIEW.md','STATISTICAL_REVIEW.md',
       'build_figures.py','derived_results.json','validate_manuscript.py','VALIDATION.json',
       'analyze_statistics.py','write_statistical_report.py','verify_statistics.py',
       'statistical_analysis.json','statistical_tests.csv','seed_endpoints.csv','STATISTICS_VALIDATION.json']
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in names:z.write(P/name,name)
    for f in sorted((P/'figures').glob('*.pdf')):z.write(f,'figures/'+f.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert z.read('main.tex')==(P/'main.tex').read_bytes()
    assert z.read('statistical_analysis.json')==(P/'statistical_analysis.json').read_bytes()
manifest={'files':{f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in [pdf,archive]},
          'validation':json.loads((P/'VALIDATION.json').read_text()),
          'statistical_validation':json.loads((P/'STATISTICS_VALIDATION.json').read_text()),
          'earlier_pdf_preserved':True,'frozen_v1_payloads_verified':38,
          'scope':'Post hoc paired statistical supplement; unchanged experiments and gates; no external performance ranking'}
(O/(stem+'_manifest.json')).write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
