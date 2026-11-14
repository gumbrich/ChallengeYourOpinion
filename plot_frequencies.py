"""Plot frequencies from the local corpus without unexpected network requests."""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def distribution(source):
    stopwords = set((ROOT / 'stopwords.txt').read_text(encoding='utf-8').splitlines())
    words = Counter()
    with (ROOT / f'{source}_scrape.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            words.update(word for word in re.findall(r'\w+', (row.get('Teaser') or '').lower()) if word not in stopwords)
    return words


def distr_nzz(): return distribution('nzz')
def distr_spiegel(): return distribution('spiegel')
def distr_taz(): return distribution('taz')


def main():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for source, axis in zip(('nzz', 'spiegel', 'taz'), axes):
        pairs = distribution(source).most_common(25)
        axis.barh([word for word, _ in reversed(pairs)], [count for _, count in reversed(pairs)])
        axis.set_title(source.upper())
    fig.tight_layout()
    fig.savefig(ROOT / 'word_frequencies.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
