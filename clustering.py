"""Cluster complete German headlines (not individual flattened tokens)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    stopwords = (ROOT / 'stopwords.txt').read_text(encoding='utf-8').splitlines()
    frames = [pd.read_csv(ROOT / f'{source}_scrape.csv').assign(source=source) for source in ('nzz', 'spiegel', 'taz')]
    headlines = pd.concat(frames, ignore_index=True)['Teaser'].dropna().str.strip().drop_duplicates()
    headlines = headlines[headlines.str.len() > 0]
    if len(headlines) < 2:
        raise ValueError('At least two headlines are required.')
    vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000)
    matrix = vectorizer.fit_transform(headlines)
    word_features = vectorizer.get_feature_names_out()
    maximum = min(15, len(headlines))
    inertia = []
    for count in range(1, maximum + 1):
        model = KMeans(n_clusters=count, n_init=10, max_iter=500, random_state=0).fit(matrix)
        inertia.append(model.inertia_)
    plt.plot(range(1, maximum + 1), inertia)
    plt.title('Elbow method · complete headlines')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster sum of squares')
    plt.savefig(ROOT / f'cluster_elbow_c{maximum}.png')
    plt.close()
    model = KMeans(n_clusters=min(4, len(headlines)), n_init=40, random_state=0).fit(matrix)
    for number, centroid in enumerate(model.cluster_centers_.argsort()[:, -1:-6:-1]):
        print(f'{number}: ' + ', '.join(word_features[index] for index in centroid))


if __name__ == '__main__':
    main()
