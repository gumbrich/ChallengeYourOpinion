import { ArrowRight, ArrowUpRight, Bookmark, Check, RefreshCw, SearchX } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { Story } from '../data/schema';
export function Arrow({ external = false }: { external?: boolean }) {
  return external ? (
    <ArrowUpRight size={17} aria-hidden="true" />
  ) : (
    <ArrowRight size={18} aria-hidden="true" />
  );
}
export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty">
      <SearchX size={30} />
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Loading() {
  return (
    <div role="status" className="loading">
      <span className="eyebrow">Perspektiven werden geladen …</span>
      <div className="skeleton-grid">
        {[1, 2, 3].map((i) => (
          <div className="skeleton" key={i} />
        ))}
      </div>
    </div>
  );
}
export function ErrorState({ retry }: { retry: () => void }) {
  return (
    <Empty
      title="Die Inhalte sind gerade nicht erreichbar."
      action={
        <button className="button" onClick={retry}>
          <RefreshCw size={17} /> Erneut versuchen
        </button>
      }
    >
      Bitte prüfe deine Verbindung. Deine gespeicherten Themen bleiben erhalten.
    </Empty>
  );
}
export function SaveButton({
  story,
  saved,
  toggle,
}: {
  story: Story;
  saved: boolean;
  toggle: (id: string) => void;
}) {
  return (
    <button
      className={`save-button ${saved ? 'is-saved' : ''}`}
      aria-pressed={saved}
      aria-label={`${story.title}: ${saved ? 'aus Merkliste entfernen' : 'merken'}`}
      onClick={() => toggle(story.id)}
    >
      {saved ? <Check size={18} /> : <Bookmark size={18} />}
    </button>
  );
}
export function PerspectiveBar({ story }: { story: Story }) {
  const counts = ['gesellschaft', 'wirtschaft', 'umsetzung'].map(
    (f) => story.perspectives.filter((p) => p.focus === f).length,
  );
  return (
    <div className="perspective-meta">
      <div className="perspective-bar" aria-hidden="true">
        {counts.map((n, i) => (
          <span
            key={i}
            className={`segment segment-${i}`}
            style={{ flex: n, display: n ? 'block' : 'none' }}
          />
        ))}
      </div>
      <span>
        {story.perspectives.length} Perspektiven <span aria-hidden="true">·</span> Beispielvergleich
      </span>
    </div>
  );
}
export function StoryCard({
  story,
  saved,
  toggle,
  featured = false,
}: {
  story: Story;
  saved: boolean;
  toggle: (id: string) => void;
  featured?: boolean;
}) {
  return (
    <article className={`story-card ${featured ? 'featured' : ''}`}>
      <Link to={`/thema/${story.slug}`} className="story-image" tabIndex={-1} aria-hidden="true">
        <img src={story.image} alt="" loading={featured ? 'eager' : 'lazy'} />
        <span className="image-label">{featured ? 'IM FOKUS' : 'PERSPEKTIVEN'}</span>
      </Link>
      <div className="story-body">
        <div className="story-topline">
          <span className="eyebrow">
            {story.category} <span className="muted">/ {story.readMinutes} Min. Lesezeit</span>
          </span>
          <SaveButton story={story} saved={saved} toggle={toggle} />
        </div>
        <h2>
          <Link to={`/thema/${story.slug}`}>{story.title}</Link>
        </h2>
        <p>{story.summary}</p>
        <div className="story-bottom">
          <PerspectiveBar story={story} />
          <Link
            to={`/thema/${story.slug}`}
            className="round-link"
            aria-label={`Perspektiven vergleichen: ${story.title}`}
          >
            <Arrow />
          </Link>
        </div>
      </div>
    </article>
  );
}
