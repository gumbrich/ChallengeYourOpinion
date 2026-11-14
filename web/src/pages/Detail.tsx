import { useEffect, useState } from 'react';
import { Link, useParams, useLocation } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Info, Link2, Check } from 'lucide-react';
import type { Story } from '../data/schema';
import { focusLabels } from '../data/schema';
import { Empty, SaveButton, StoryCard } from '../components/UI';
import Reflection from '../components/Reflection';
export default function Detail({
  stories,
  saved,
  toggle,
}: {
  stories: Story[];
  saved: string[];
  toggle: (id: string) => void;
}) {
  const { slug } = useParams();
  const location = useLocation();
  const story = stories.find((s) => s.slug === slug);
  const [selected, setSelected] = useState<string[]>([]);
  const [shared, setShared] = useState('');
  useEffect(() => {
    if (story) {
      setSelected(story.perspectives.map((p) => p.id));
      setShared('');
    }
  }, [story]);
  useEffect(() => {
    if (['#reflexion', '#vergleich'].includes(location.hash)) {
      document.getElementById(location.hash.slice(1))?.scrollIntoView();
    }
  }, [location.hash, story]);
  if (!story)
    return (
      <div className="container page">
        <Empty
          title="Dieses Thema ist nicht dabei."
          action={
            <Link className="button" to="/">
              Alle Themen entdecken
            </Link>
          }
        >
          Vielleicht hat sich der Link geändert. Im Überblick findest du alle verfügbaren Themen.
        </Empty>
      </div>
    );
  const chosen = story.perspectives.filter((p) => selected.includes(p.id));
  async function share() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setShared('Link kopiert.');
    } catch {
      setShared('Kopieren nicht möglich. Du kannst den Link aus der Adressleiste teilen.');
    }
  }
  return (
    <div className="container page detail-page">
      <Link to="/" className="back-link">
        <ArrowLeft size={16} /> Zur Themenübersicht
      </Link>
      <header className="detail-header">
        <div className="eyebrow">
          {story.category} <span className="muted">/ {story.readMinutes} MIN. LESEZEIT / DEMO</span>
        </div>
        <h1>{story.title}</h1>
        <p className="detail-deck">{story.summary}</p>
        <div className="detail-actions">
          <a href="#vergleich" className="button">
            Perspektiven vergleichen <ArrowRight size={17} />
          </a>
          <SaveButton story={story} saved={saved.includes(story.id)} toggle={toggle} />
          <button className="share-button" onClick={share}>
            <Link2 size={17} /> Link teilen
          </button>
          <span role="status" className="small">
            {shared}
          </span>
        </div>
      </header>
      <img className="detail-image" src={story.image} alt="" />
      <div className="context-grid">
        <div>
          <div className="eyebrow">WORUM ES GEHT</div>
          <h2>Ein gemeinsamer Ausgangspunkt.</h2>
          {story.context.map((c) => (
            <p key={c}>{c}</p>
          ))}
        </div>
        <aside className="context-note">
          <Info size={21} />
          <h3>Ein Beispiel, keine Nachricht.</h3>
          <p>
            Alle Texte und drei Beispielquellen sind eigens für diesen Prototyp verfasst. Sie
            stammen nicht von realen Medien. Die Schwerpunkte sind illustrative Zuordnungen.
          </p>
          <Link className="text-link" to="/methodik">
            Wie wir vergleichen <ArrowRight size={15} />
          </Link>
        </aside>
      </div>
      <section className="comparison" id="vergleich">
        <div className="section-heading">
          <div>
            <div className="eyebrow">DIE GLEICHE FRAGE. ANDERE SCHWERPUNKTE.</div>
            <h2>Was rückt in den Vordergrund?</h2>
          </div>
          <span className="small muted">
            {chosen.length} von {story.perspectives.length} Blickwinkeln
          </span>
        </div>
        <p className="section-description">
          Vergleiche Argumente statt Etiketten. Ein Schwerpunkt ist keine politische Einordnung und
          kein Urteil über Glaubwürdigkeit.
        </p>
        <fieldset className="comparison-controls">
          <legend>Blickwinkel im Vergleich</legend>
          {story.perspectives.map((p) => (
            <label key={p.id}>
              <input
                type="checkbox"
                checked={selected.includes(p.id)}
                onChange={() =>
                  setSelected((v) =>
                    v.includes(p.id) ? v.filter((id) => id !== p.id) : [...v, p.id],
                  )
                }
              />
              <span className={`focus-dot ${p.focus}`} />
              {focusLabels[p.focus]}
            </label>
          ))}
        </fieldset>
        {chosen.length === 0 ? (
          <Empty title="Wähle einen Blickwinkel aus.">
            Mit mindestens zwei Perspektiven kannst du Argumente direkt vergleichen.
          </Empty>
        ) : (
          <div className="comparison-grid">
            {chosen.map((p) => (
              <article className={`perspective-card ${p.focus}`} key={p.id}>
                <div className="perspective-source">
                  <span className={`source-monogram ${p.focus}`}>{p.source[0]}</span>
                  <span>
                    {p.source}
                    <small>Fiktive Quelle · Beispieltext</small>
                  </span>
                </div>
                <span className={`focus-tag ${p.focus}`}>{focusLabels[p.focus]}</span>
                <h3>{p.headline}</h3>
                <p>{p.argument}</p>
                <div className="evidence-block">
                  <h4>Welche Belege fehlen?</h4>
                  <p>{p.evidence}</p>
                </div>
                <div className="blindspot">
                  <h4>Was dieser Blick offenlässt</h4>
                  <p>{p.blindSpot}</p>
                </div>
                <span className="assessment">
                  <Info size={14} /> Keine externe Prüfung
                </span>
              </article>
            ))}
          </div>
        )}
        {chosen.length === 1 && (
          <p className="small">Wähle einen weiteren Blickwinkel für den direkten Vergleich.</p>
        )}
        <p className="comparison-caption">
          Die drei Farben stehen für thematische Schwerpunkte. Sie messen weder Wahrheit noch
          politische Richtung. In diesem Demo sind die Perspektiven gleich gewichtet.
        </p>
      </section>
      <section className="open-questions">
        <span className="eyebrow">GUTE FRAGEN BLEIBEN</span>
        <h2>Was wir noch wissen müssten.</h2>
        {story.unknowns.map((q) => (
          <p key={q}>
            <span aria-hidden="true">↗</span>
            {q}
          </p>
        ))}
        <div className="small muted">
          <Check size={16} /> Unsicherheit sichtbar zu machen gehört zum Vergleich.
        </div>
      </section>
      <Reflection key={story.id} story={story} />
      <section className="related">
        <div className="section-heading">
          <h2>Weiter über den Tellerrand.</h2>
          <Link className="text-link" to="/">
            Alle Themen <ArrowRight size={16} />
          </Link>
        </div>
        <div className="stories-grid">
          {stories
            .filter((s) => s.id !== story.id)
            .slice(0, 3)
            .map((s) => (
              <StoryCard key={s.id} story={s} saved={saved.includes(s.id)} toggle={toggle} />
            ))}
        </div>
      </section>
    </div>
  );
}
