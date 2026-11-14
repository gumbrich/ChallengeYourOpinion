"""Headline collection for the original research corpus.

Selectors are historical and may need maintenance. Run explicitly with --once or
--interval; importing this module never starts network access or a scheduler.
"""
import argparse
import csv
import logging
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'nzz': ('https://www.nzz.ch', 'span.teaser__title-name', None),
    'spiegel': ('https://www.spiegel.de', 'article[aria-label]', 'aria-label'),
    'taz': ('https://taz.de', 'h3', None),
    'focus': ('https://www.focus.de', 'a[title]', 'title'),
}


def append_headlines(path, headlines, captured_at=None):
    """Normalize and deduplicate without changing historic dates or losing rows."""
    rows = []
    if path.exists():
        with path.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
    known = {' '.join((row.get('Teaser') or '').split()) for row in rows}
    added = 0
    for headline in headlines:
        headline = ' '.join((headline or '').split())
        if headline and headline not in known:
            rows.append({'Teaser': headline, 'Date': captured_at or date.today().isoformat()})
            known.add(headline)
            added += 1
    # Atomic replacement keeps an interrupted collection from truncating the corpus.
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Teaser', 'Date'])
        writer.writeheader()
        writer.writerows({'Teaser': row.get('Teaser', ''), 'Date': row.get('Date', '')} for row in rows)
    temporary.replace(path)
    return added


def scrape_source(source, output_dir=ROOT):
    import requests
    from bs4 import BeautifulSoup
    url, selector, attribute = SOURCES[source]
    response = requests.get(url, timeout=20, headers={'User-Agent': 'ChallengeYourOpinion-Research/1.0'})
    response.raise_for_status()
    elements = BeautifulSoup(response.content, 'html.parser').select(selector)
    if not elements:
        logging.warning('%s: no matches; inspect the selector. Existing data was not changed.', source)
        return 0
    headlines = [element.get(attribute) if attribute else element.get_text(' ', strip=True) for element in elements]
    output_dir.mkdir(parents=True, exist_ok=True)
    return append_headlines(output_dir / f'{source}_scrape.csv', headlines)


def scrape_nzz(): return scrape_source('nzz')
def scrape_spiegel(): return scrape_source('spiegel')
def scrape_taz(): return scrape_source('taz')
def scrape_focus(): return scrape_source('focus')


def scrape_web(sources=('nzz', 'spiegel', 'taz')):
    for source in sources:
        try:
            logging.info('%s: %s new headlines', source, scrape_source(source))
        except Exception:
            logging.exception('%s: collection failed; continuing with other sources', source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--interval', type=int, help='Poll every N seconds (minimum 60).')
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=['nzz', 'spiegel', 'taz'])
    args = parser.parse_args()
    if args.interval is not None and args.interval < 60:
        parser.error('--interval must be at least 60 seconds')
    logging.basicConfig(level=logging.INFO)
    while True:
        scrape_web(args.sources)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
