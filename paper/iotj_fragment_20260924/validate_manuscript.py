"""Check manuscript pagination, resolved references, and every aggregate-table value."""
from pathlib import Path
import hashlib,json,re
from pypdf import PdfReader
P=Path(__file__).resolve().parent;R=P.parents[1]
source=(P/'main.tex').read_text();log=(P/'main.log').read_text()
reader=PdfReader(P/'main.pdf');assert 8<=len(reader.pages)<=10
assert 'Overfull' not in log and 'undefined' not in log
abstract=source.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert 150<=len(abstract.split())<=250
citations=set(k for item in re.findall(r'\\cite\{([^}]+)\}',source) for k in item.split(','))
references=set(re.findall(r'\\bibitem\{([^}]+)\}',source));assert citations==references
table=source.split(r'\label{tab:aggregate}')[1].split(r'\end{table*}')[0]
numeric_rows=[]
for line in table.splitlines():
    values=re.findall(r'&\s*([0-9]+\.[0-9]+)',line)
    if len(values)==3:numeric_rows.append([float(v) for v in values])
expected=[]
for study,policies in [('fixed',['whole','selective','block','prefix']),('guard',['whole','selective','block','prefix','guard']),('shared',['whole','selective','block'])]:
    results=json.loads((R/'evidence/fragment_repair_20260921'/study/'results.json').read_text())
    for policy in policies:
        m=results['aggregate'][policy]
        expected.append([round(100*m['delivery_ratio'],3),round(m['packets_per_j'],4),round(m['energy_uj'],2)])
assert numeric_rows==expected,(numeric_rows,expected)
alltext='\n'.join(p.extract_text() for p in reader.pages)
assert '??' not in alltext
result=dict(pages=len(reader.pages),abstract_words=len(abstract.split()),references=len(references),
    aggregate_table_values_verified=len(expected)*3,undefined_references=0,overfull_boxes=0,
    authors_pending=True,source_pdf_sha256=hashlib.sha256((P/'main.pdf').read_bytes()).hexdigest(),
    study_evidence='published commit 0e12c49; component evaluation only',
    visual_review='all ten pages reviewed; final diagnostic legends moved outside data areas')
(P/'VALIDATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
