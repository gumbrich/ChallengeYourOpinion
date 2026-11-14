import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Search, ArrowUpRight, Database, Info } from 'lucide-react';
import { newsRepository } from '../data/repository';
import { sources } from '../data/sources';
import { useResource } from '../hooks';
import { Empty, ErrorState, Loading } from '../components/UI';
const loadArchive = (signal: AbortSignal) => newsRepository.getArchive(signal);
export default function Archive({ directory = false }: { directory?: boolean }) {
  const { data, error, retry } = useResource(loadArchive);
  const [params, setParams] = useSearchParams();
  const query = params.get('q') || '';
  const source = params.get('quelle') || 'alle';
  const [limit, setLimit] = useState(30);
  function filter(key: string, value: string) {
    setLimit(30);
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (value && value !== 'alle') next.set(key, value);
        else next.delete(key);
        return next;
      },
      { replace: true },
    );
  }
  const records =
    data?.filter(
      (r) =>
        (source === 'alle' || r.source === source) &&
        r.headline.toLocaleLowerCase('de').includes(query.toLocaleLowerCase('de').trim()),
    ) || [];
  return (
    <div className="container page">
      <section className="page-heading">
        <div className="eyebrow">
          {directory ? 'HERKUNFT SICHTBAR MACHEN' : 'AUS DEM URSPRÜNGLICHEN PROJEKT'}
        </div>
        <h1>
          {directory ? (
            <>
              Jede Quelle hat
              <br />
              einen <em>Kontext.</em>
            </>
          ) : (
            <>
              Ein Blick zurück.
              <br />
              Mit offenen Daten.
            </>
          )}
        </h1>
        <p>
          {directory
            ? 'Keine pauschalen Wahrheitsscores. Hier siehst du, welche Medien im historischen Korpus vorkommen – und welche Informationen fehlen.'
            : 'Die echten Schlagzeilen aus dem Forschungskorpus von 2021. Unverändert in ihrer Aussage, mit bereinigten Leerzeichen und entfernten Dubletten.'}
        </p>
      </section>
      <div className="archive-notice">
        <Database size={21} />
        <div>
          <strong>Historischer Datensatz, kein Live-Feed</strong>
          <p>
            Ein Datum bezeichnet die Erfassung, nicht die Veröffentlichung. Undatierte Einträge
            bleiben undatiert. Der ursprüngliche Scraper erfasste auch Werbung und Navigation; diese
            Daten wurden nicht redaktionell geprüft.
          </p>
        </div>
      </div>
      {error ? (
        <ErrorState retry={retry} />
      ) : !data ? (
        <Loading />
      ) : directory ? (
        <>
          <div className="source-grid">
            {sources.map((s) => (
              <article className="source-card" key={s.id}>
                <div className="source-card-top">
                  <span className="outlet-mark" style={{ background: s.color }}>
                    {s.short}
                  </span>
                  <span className="small muted">{s.country}</span>
                </div>
                <h2>{s.name}</h2>
                <p>{s.description}</p>
                <dl>
                  <div>
                    <dt>Archivierte Einträge</dt>
                    <dd>{data.filter((r) => r.source === s.id).length.toLocaleString('de-DE')}</dd>
                  </div>
                  <div>
                    <dt>Politische Einordnung</dt>
                    <dd>Nicht bewertet</dd>
                  </div>
                  <div>
                    <dt>Eigentum & Finanzierung</dt>
                    <dd>Nicht recherchiert</dd>
                  </div>
                  <div>
                    <dt>Artikelbelege & Faktenprüfung</dt>
                    <dd>Nicht vorhanden</dd>
                  </div>
                </dl>
                <div className="source-card-actions">
                  <Link className="text-link" to={`/archiv?quelle=${s.id}`}>
                    Archiv ansehen <ArrowUpRight size={16} />
                  </Link>
                  <a
                    className="small"
                    href={s.url}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`${s.name}: Website öffnen (neuer Tab)`}
                  >
                    Website ↗
                  </a>
                </div>
              </article>
            ))}
          </div>
          <section className="method-note">
            <Info size={23} />
            <div>
              <h2>Verschiedene Quellen sind noch kein vollständiges Bild.</h2>
              <p>
                Der Korpus ist eine kleine, historische Auswahl. Die bloße Aufnahme einer Quelle ist
                keine Empfehlung. Für einen belastbaren Live-Vergleich fehlen vollständige Artikel,
                verifizierte Herkunft und dokumentierte Einordnungen.
              </p>
              <Link className="text-link" to="/methodik">
                Unsere Grenzen verstehen →
              </Link>
            </div>
          </section>
        </>
      ) : (
        <>
          <div className="archive-filters">
            <div className="search-field">
              <Search size={18} />
              <label className="sr-only" htmlFor="archive-search">
                Archiv durchsuchen
              </label>
              <input
                id="archive-search"
                type="search"
                placeholder="Schlagzeilen durchsuchen …"
                value={query}
                onChange={(e) => filter('q', e.target.value)}
              />
            </div>
            <label className="source-select">
              Quelle
              <select value={source} onChange={(e) => filter('quelle', e.target.value)}>
                <option value="alle">Alle Quellen</option>
                {sources.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="section-heading">
            <h2>Schlagzeilen im Originalwortlaut</h2>
            <span className="small muted" role="status">
              {records.length.toLocaleString('de-DE')} Einträge
            </span>
          </div>
          {!records.length ? (
            <Empty
              title="Keine Schlagzeile gefunden."
              action={
                <button
                  className="button"
                  onClick={() => {
                    setParams({});
                    setLimit(30);
                  }}
                >
                  Filter zurücksetzen
                </button>
              }
            >
              Versuche einen anderen Suchbegriff oder eine andere Quelle.
            </Empty>
          ) : (
            <>
              <ul className="archive-list">
                {records.slice(0, limit).map((r) => {
                  const s = sources.find((s) => s.id === r.source)!;
                  return (
                    <li key={r.id}>
                      <span className="archive-mark" style={{ color: s.color }}>
                        {s.short}
                      </span>
                      <div>
                        <div className="eyebrow muted">
                          {s.name}{' '}
                          <span>
                            ·{' '}
                            {r.capturedAt
                              ? `Erfasst am ${new Intl.DateTimeFormat('de-DE', { timeZone: 'UTC' }).format(new Date(r.capturedAt))}`
                              : 'Erfassungsdatum unbekannt'}
                          </span>
                        </div>
                        <h3>{r.headline}</h3>
                        <span className="small muted">
                          Original-URL nicht erfasst · {r.sourceFile}
                        </span>
                      </div>
                    </li>
                  );
                })}
              </ul>
              <div className="load-more">
                <p className="small muted">
                  {Math.min(limit, records.length)} von {records.length.toLocaleString('de-DE')}{' '}
                  Einträgen
                </p>
                {limit < records.length && (
                  <button className="button secondary" onClick={() => setLimit((v) => v + 30)}>
                    Weitere 30 Einträge anzeigen
                  </button>
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
