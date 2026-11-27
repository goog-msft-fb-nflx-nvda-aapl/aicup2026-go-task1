"""Check the versioned guide/report mirror against the shared reproduction contract."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report-text', type=Path, help='Optional text extracted from the updated report PDF')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
contract = json.loads((root / 'config/reproduction_contract.json').read_text())
weights = json.loads((root / contract['weights_manifest']).read_text())
checkpoint = next(a for a in weights['assets'] if a['name'] == 'idft_cx40_lm4_noaug_all.pt')
assert checkpoint['sha256'] == contract['accepted_cx40_sha256']
documents = {name: (root / name).read_text() for name in ['REPRODUCE.md', 'REPORT_REPRODUCTION.md']}
if args.report_text:
    documents['rendered_report'] = args.report_text.read_text()
for name, text in documents.items():
    compact = re.sub(r'\s+', '', text)
    for command in contract['commands'].values():
        assert re.sub(r'\s+', '', command) in compact, (name, command)
    for field in ['version', 'accepted_cx40_sha256', 'accepted_csv_sha256']:
        # The accepted prediction digest is printed by the report builder after section 7.
        assert contract[field] in compact, (name, field)
assert not contract['retraining_required']
print('GUIDE_REPORT_COMMANDS_AND_SELECTED_MODEL_CONSISTENT', len(contract['commands']))
print('Inference acceptance status:', contract['raw_inference_status'])
