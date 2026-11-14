import { ArrowDown, ArrowRight, Check, Compass, Search, X } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import { useState } from 'react';
import { z } from 'zod';
import { useLocalState } from '../hooks';
import { filterStories } from '../data/repository';
import type { Story } from '../data/schema';
import { Empty, StoryCard } from '../components/UI';
const bool = z.boolean();
export default function Feed({
  stories,
  saved,
  toggle,
  savedOnly = false,
}: {
  stories: Story[];
  saved: string[];
  toggle: (id: string) => void;
  savedOnly?: boolean;
}) {
  const [params, setParams] = useSearchParams();
  const query = params.get('q') || '';
  const category = params.get('thema') || 'Alle Themen';
  const [onboarded, setOnboarded] = useLocalState('weitblick:onboarded', false, bool);
  const [guide, setGuide] = useState(false);
  const categories = ['Alle Themen', ...new Set(stories.map((s) => s.category))];
  const filtered = filterStories(stories, query, category, savedOnly ? saved : undefined);
  const isDefault = !savedOnly && saved.length === 0 && !query && category === 'Alle Themen';
  const setFilter = (key: string, value: string) =>
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (value && value !== 'Alle Themen') next.set(key, value);
        else next.delete(key);
        return next;
      },
      { replace: true },
    );
  return (
    <div className="container page feed-page">
      {!savedOnly ? (
        <>
          <section className="intro">
            <div>
              <div className="eyebrow overline">
                <span className="status-dot" /> NACHRICHTEN BESSER EINORDNEN
              </div>
              <h1>
                Versteh mehr
                <br />
                als <em>eine Seite.</em>
                <span className="coral-dot" aria-hidden="true">
                  ●
                </span>
              </h1>
            </div>
            <div className="intro-copy">
              <p>
                Dieselbe Welt. Unterschiedliche Blickwinkel. <br />
                Entdecke, was andere sehen – und was deiner
                <br className="desktop-break" /> eigenen Perspektive vielleicht entgeht.
              </p>
              <a href="#themen" className="text-link">
                Den Blick weiten <ArrowDown size={17} />
              </a>
            </div>
          </section>
          <div className="demo-notice">
            <span>
              <strong>Ein offener Blick. Ein offener Prototyp.</strong> Die Themen sind redaktionell
              verfasste Beispiele, keine aktuellen Nachrichten.
            </span>
            <Link to="/archiv">
              Originalarchiv entdecken <ArrowRight size={15} />
            </Link>
          </div>
        </>
      ) : (
        <section className="page-heading">
          <div className="eyebrow">DEIN LESEZEICHEN</div>
          <h1>Für einen zweiten Blick.</h1>
          <p>Deine gemerkten Themen. Nur in diesem Browser gespeichert.</p>
        </section>
      )}
      <section id="themen" className="feed-section">
        <div className="section-heading">
          <h2>{savedOnly ? 'Deine Merkliste' : 'Themen, die mehr als eine Antwort haben'}</h2>
          <span className="small muted">
            {filtered.length} {filtered.length === 1 ? 'Thema' : 'Themen'} · Demo-Ausgabe
          </span>
        </div>
        <div className="filter-row">
          <div className="category-list" aria-label="Themen filtern">
            {categories.map((c) => (
              <button
                key={c}
                aria-pressed={category === c}
                className={category === c ? 'active' : ''}
                onClick={() => setFilter('thema', c)}
              >
                {c}
              </button>
            ))}
          </div>
          <div className="search-field">
            <Search size={17} />
            <label className="sr-only" htmlFor="topic-search">
              Themen durchsuchen
            </label>
            <input
              id="topic-search"
              type="search"
              value={query}
              placeholder="Thema suchen"
              onChange={(e) => setFilter('q', e.target.value)}
            />
          </div>
        </div>
        <p className="sr-only" role="status">
          {filtered.length} Themen gefunden
        </p>
        {filtered.length === 0 ? (
          <Empty
            title={
              savedOnly && saved.length === 0 && !query
                ? 'Noch nichts auf deiner Merkliste.'
                : 'Hier fehlt noch ein Blickwinkel.'
            }
            action={
              savedOnly && saved.length === 0 && !query ? (
                <Link className="button" to="/">
                  Themen entdecken <ArrowRight size={16} />
                </Link>
              ) : (
                <button className="button" onClick={() => setParams({})}>
                  Filter zurücksetzen
                </button>
              )
            }
          >
            {savedOnly && saved.length === 0 && !query
              ? 'Mit dem Lesezeichen an einem Thema kannst du es für später merken.'
              : 'Für diese Suche gibt es keine Themen. Versuche einen anderen Begriff oder entferne die Filter.'}
          </Empty>
        ) : (
          <>
            {isDefault ? (
              <div className="lead-grid">
                <StoryCard
                  story={filtered[0]}
                  featured
                  saved={saved.includes(filtered[0].id)}
                  toggle={toggle}
                />
                <aside className="challenge-card">
                  <div className="challenge-label">
                    <Compass size={18} /> DER PERSPEKTIVWECHSEL
                  </div>
                  <div className="orbit-art" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                    <b>↗</b>
                  </div>
                  <span className="small">Ein guter Gedanke beginnt mit einer Frage.</span>
                  <h2>Was, wenn die andere Seite einen Punkt hat?</h2>
                  <p>
                    Keine Punkte. Kein Richtig oder Falsch. Nimm dir einen Moment für das stärkste
                    Argument der anderen Seite.
                  </p>
                  <Link to={`/thema/${filtered[0].slug}#reflexion`} className="button light">
                    Perspektivwechsel starten <ArrowRight size={17} />
                  </Link>
                  <span className="challenge-foot">3 Schritte · nur für dich</span>
                </aside>
              </div>
            ) : null}
            <div className="stories-grid">
              {(isDefault ? filtered.slice(1) : filtered).map((s) => (
                <StoryCard key={s.id} story={s} saved={saved.includes(s.id)} toggle={toggle} />
              ))}
            </div>
          </>
        )}
      </section>
      {!savedOnly && !onboarded && (
        <section className="onboarding">
          <div className="onboard-icon">
            <Compass size={26} />
          </div>
          <div>
            <div className="eyebrow">DEIN ERSTER PERSPEKTIVWECHSEL</div>
            <h2>
              Du musst deine Meinung nicht ändern.
              <br />
              Nur wissen, was sie herausfordert.
            </h2>
            <p>Thema wählen. Blickwinkel vergleichen. Eigene Gedanken prüfen.</p>
            {guide && (
              <ol className="guide">
                <li>
                  <strong>Erst verstehen.</strong> Lies den Kontext und die offenen Fragen.
                </li>
                <li>
                  <strong>Fair vergleichen.</strong> Prüfe Argumente, Belege und blinde Flecken.
                </li>
                <li>
                  <strong>Selbst entscheiden.</strong> Halte fest, was dich überzeugt – und was
                  nicht.
                </li>
              </ol>
            )}
            <button className="text-link" onClick={() => setGuide(!guide)} aria-expanded={guide}>
              {guide ? 'Anleitung schließen' : 'So funktioniert WEITBLICK'} <ArrowRight size={16} />
            </button>
          </div>
          <button
            className="dismiss"
            aria-label="Einführung ausblenden"
            onClick={() => setOnboarded(true)}
          >
            <X size={18} />
          </button>
        </section>
      )}
      {!savedOnly && (
        <div className="principles">
          <span>
            <Check size={17} /> Perspektiven statt Schubladen
          </span>
          <span>
            <Check size={17} /> Herkunft und Grenzen offenlegen
          </span>
          <span>
            <Check size={17} /> Deine Meinung bleibt deine
          </span>
        </div>
      )}
    </div>
  );
}
