"""Export the original CSV corpus without inventing dates, article URLs or ratings."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('nzz', 'spiegel', 'taz', 'focus')

def read_source(path, source):
    records, seen = [], set()
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            headline = ' '.join((row.get('Teaser') or '').split())
            if not headline or headline in seen:
                continue
            seen.add(headline)
            records.append({
                'id': hashlib.sha256(f'{source}:{headline}'.encode()).hexdigest()[:16],
                'source': source, 'headline': headline,
                'capturedAt': row.get('Date') or None,
                'sourceFile': path.name, 'articleUrl': None,
            })
    return records

def export(destination=ROOT / 'web/public/data/archive.json'):
    records = [record for source in SOURCES for record in read_source(ROOT / f'{source}_scrape.csv', source)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'version': 1, 'records': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(records)} unique archived headlines to {destination}')

if __name__ == '__main__':
    export()
