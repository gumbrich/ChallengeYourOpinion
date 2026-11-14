import { Bookmark, Compass, Menu, X, ArrowUpRight } from 'lucide-react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
export function Layout({
  children,
  savedCount,
}: {
  children: React.ReactNode;
  savedCount: number;
}) {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const main = useRef<HTMLElement>(null);
  const first = useRef(true);
  const menuButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    setOpen(false);
    if (first.current) {
      first.current = false;
      return;
    }
    main.current?.focus({ preventScroll: true });
    const anchor = document.getElementById(location.hash.slice(1));
    if (anchor) anchor.scrollIntoView();
    else window.scrollTo({ top: 0, behavior: 'instant' });
  }, [location.pathname, location.hash]);
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
        menuButton.current?.focus();
      }
    };
    document.addEventListener('keydown', close);
    return () => document.removeEventListener('keydown', close);
  }, [open]);
  return (
    <>
      <a className="skip-link" href="#main">
        Zum Inhalt springen
      </a>
      <div className="edition-bar">
        <div className="container">
          <span>Ein Thema. Mehr Perspektiven.</span>
          <Link to="/methodik">
            Unabhängig denken beginnt mit Hinsehen <ArrowUpRight size={13} />
          </Link>
        </div>
      </div>
      <header className="site-header">
        <div className="container header-inner">
          <Link className="brand" to="/" aria-label="WEITBLICK Startseite">
            <span className="brand-icon" aria-hidden="true">
              w
            </span>
            <span>
              WEITBLICK<small>CHALLENGE YOUR OPINION</small>
            </span>
          </Link>
          <button
            className="menu-button"
            ref={menuButton}
            aria-label={open ? 'Menü schließen' : 'Menü öffnen'}
            aria-expanded={open}
            aria-controls="main-nav"
            onClick={() => setOpen(!open)}
          >
            {open ? <X /> : <Menu />}
          </button>
          <nav id="main-nav" className={open ? 'nav-open' : ''} aria-label="Hauptnavigation">
            <NavLink to="/" end>
              Entdecken
            </NavLink>
            <NavLink to="/quellen">Quellen</NavLink>
            <NavLink to="/archiv">Archiv</NavLink>
            <NavLink to="/methodik">Unser Ansatz</NavLink>
            <NavLink to="/merkliste" className="saved-nav">
              <Bookmark size={17} /> Merkliste <span className="count">{savedCount}</span>
            </NavLink>
          </nav>
        </div>
      </header>
      <main id="main" ref={main} tabIndex={-1}>
        {children}
      </main>
      <footer className="site-footer">
        <div className="container footer-top">
          <div>
            <Link to="/" className="brand footer-brand">
              <Compass size={28} /> WEITBLICK
            </Link>
            <p>Für eine Meinung, die weiter denkt.</p>
            <span className="small muted">Ein Projekt von Challenge Your Opinion.</span>
          </div>
          <div className="footer-links">
            <Link to="/methodik">Methodik & Grenzen</Link>
            <Link to="/quellen">Unsere Quellen</Link>
            <Link to="/datenschutz">Datenschutz</Link>
            <a
              href="https://github.com/gumbrich/ChallengeYourOpinion"
              target="_blank"
              rel="noreferrer"
            >
              Projekt auf GitHub <ArrowUpRight size={14} />
            </a>
          </div>
        </div>
        <div className="container footer-bottom">
          <span>Offener Prototyp · Keine aktuelle Nachrichtenredaktion</span>
          <span>Neugierig bleiben. Fair vergleichen.</span>
        </div>
      </footer>
    </>
  );
}
