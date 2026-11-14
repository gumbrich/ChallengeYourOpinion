import { Component, useEffect } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { z } from 'zod';
import { Layout } from './components/Layout';
import { Empty, ErrorState, Loading } from './components/UI';
import { useLocalState, useResource } from './hooks';
import { newsRepository } from './data/repository';
import Feed from './pages/Feed';
import Detail from './pages/Detail';
import Archive from './pages/Archive';
import About from './pages/About';
const loadStories = (signal: AbortSignal) => newsRepository.getStories(signal);
const savedSchema = z.array(z.string());
class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(_error: Error, _info: ErrorInfo) {
    /* Do not transmit private reflections or error state. */
  }
  render() {
    return this.state.failed ? (
      <div className="container page">
        <Empty
          title="Hier ist etwas schiefgegangen."
          action={
            <button className="button" onClick={() => window.location.reload()}>
              Seite neu laden
            </button>
          }
        >
          Bitte lade die Seite erneut. Deine gespeicherten Themen bleiben erhalten.
        </Empty>
      </div>
    ) : (
      this.props.children
    );
  }
}
export default function App() {
  const { data: stories, error, retry } = useResource(loadStories);
  const [saved, setSaved, saveFailed] = useLocalState('weitblick:saved', [], savedSchema);
  const location = useLocation();
  useEffect(() => {
    const titles: Record<string, string> = {
      '/': 'Versteh mehr als eine Seite.',
      '/quellen': 'Quellen & Kontext',
      '/archiv': 'Das Originalarchiv',
      '/methodik': 'Unser Ansatz',
      '/datenschutz': 'Datenschutz',
      '/merkliste': 'Deine Merkliste',
    };
    const story = stories?.find((s) => location.pathname === `/thema/${s.slug}`);
    const title = `${story?.title || titles[location.pathname] || 'Seite nicht gefunden'} | WEITBLICK`;
    const description =
      story?.summary ||
      'Ein Thema. Mehr Perspektiven. WEITBLICK macht unterschiedliche Blickwinkel sichtbar – transparent und ohne feste politische Schubladen.';
    document.title = title;
    document.querySelector('meta[name="description"]')?.setAttribute('content', description);
    document.querySelector('meta[property="og:title"]')?.setAttribute('content', title);
    document.querySelector('meta[property="og:description"]')?.setAttribute('content', description);
  }, [location.pathname, stories]);
  const toggle = (id: string) =>
    setSaved((previous) =>
      previous.includes(id) ? previous.filter((v) => v !== id) : [...previous, id],
    );
  const feed = (savedOnly = false) =>
    error ? (
      <div className="container page">
        <ErrorState retry={retry} />
      </div>
    ) : stories ? (
      <Feed stories={stories} saved={saved} toggle={toggle} savedOnly={savedOnly} />
    ) : (
      <div className="container page">
        <Loading />
      </div>
    );
  return (
    <Layout savedCount={saved.length}>
      <ErrorBoundary>
        {saveFailed && (
          <div className="container storage-warning" role="status">
            Dein Browser verhindert das Speichern. Änderungen bleiben nur in dieser Sitzung
            erhalten.
          </div>
        )}
        <Routes>
          <Route path="/" element={feed()} />
          <Route path="/merkliste" element={feed(true)} />
          <Route
            path="/thema/:slug"
            element={
              error ? (
                <div className="container page">
                  <ErrorState retry={retry} />
                </div>
              ) : stories ? (
                <Detail stories={stories} saved={saved} toggle={toggle} />
              ) : (
                <div className="container page">
                  <Loading />
                </div>
              )
            }
          />
          <Route path="/quellen" element={<Archive directory />} />
          <Route path="/archiv" element={<Archive />} />
          <Route path="/methodik" element={<About />} />
          <Route path="/datenschutz" element={<About privacy />} />
          <Route
            path="*"
            element={
              <div className="container page">
                <Empty
                  title="Hier geht der Blick ins Leere."
                  action={
                    <Link className="button" to="/">
                      Zur Startseite
                    </Link>
                  }
                >
                  Diese Seite gibt es nicht. In der Themenübersicht findest du den nächsten
                  Blickwinkel.
                </Empty>
              </div>
            }
          />
        </Routes>
      </ErrorBoundary>
    </Layout>
  );
}
