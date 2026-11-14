import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import App from './App';
import { feedSchema, archiveSchema } from './data/schema';
import { filterStories, newsRepository } from './data/repository';
import raw from '../public/data/stories.json';
import archive from '../public/data/archive.json';
const stories = feedSchema.parse(raw).stories;
function mount(path = '/') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (path: string) => ({
      ok: true,
      json: async () => (path.includes('archive') ? archive : raw),
    })),
  );
  vi.stubGlobal('scrollTo', vi.fn());
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
describe('data contract', () => {
  it('validates all seeds and preserves genuine archive provenance', () => {
    expect(stories).toHaveLength(6);
    const data = archiveSchema.parse(archive);
    expect(data.records.length).toBe(3698);
    expect(data.records.some((r) => r.capturedAt === null)).toBe(true);
    expect(data.records.every((r) => r.articleUrl === null)).toBe(true);
    expect(new Set(data.records.map((r) => r.id)).size).toBe(data.records.length);
  });
  it('combines search terms, category and saved filters', () => {
    expect(
      filterStories(stories, 'ticket klima', 'Politik', ['mobilitaet']).map((s) => s.id),
    ).toEqual(['mobilitaet']);
    expect(filterStories(stories, 'ticket', 'Klima')).toEqual([]);
    expect(filterStories(stories, '', 'Alle Themen', [])).toEqual([]);
  });
  it('rejects server failures and malformed feeds', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: false, status: 503 } as Response);
    await expect(newsRepository.getStories()).rejects.toThrow('503');
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ stories: [{}] }),
    } as Response);
    await expect(newsRepository.getStories()).rejects.toThrow();
  });
});
describe('reader flows', () => {
  it('closes the mobile menu with Escape and restores focus', async () => {
    const user = userEvent.setup();
    mount();
    await user.click(screen.getByRole('button', { name: 'Menü öffnen' }));
    expect(screen.getByRole('button', { name: 'Menü schließen' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    await user.keyboard('{Escape}');
    expect(screen.getByRole('button', { name: 'Menü öffnen' })).toHaveFocus();
  });
  it('uses a topic-specific title on direct navigation', async () => {
    mount(`/thema/${stories[0].slug}`);
    await screen.findByRole('heading', { name: stories[0].title, level: 1 });
    await waitFor(() => expect(document.title).toBe(`${stories[0].title} | WEITBLICK`));
  });

  it('searches the feed and restores the empty state', async () => {
    const user = userEvent.setup();
    mount();
    const search = await screen.findByRole('searchbox', { name: 'Themen durchsuchen' });
    await user.type(search, 'nichtvorhanden');
    expect(screen.getByText('Hier fehlt noch ein Blickwinkel.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Filter zurücksetzen' }));
    expect(screen.getByRole('link', { name: stories[0].title })).toBeInTheDocument();
  });
  it('persists a bookmark and removes it from the reading list', async () => {
    const user = userEvent.setup();
    mount();
    await user.click(await screen.findByRole('button', { name: `${stories[0].title}: merken` }));
    expect(JSON.parse(localStorage.getItem('weitblick:saved')!)).toEqual(['mobilitaet']);
    await user.click(screen.getByRole('link', { name: /Merkliste 1/ }));
    await user.click(
      screen.getByRole('button', { name: `${stories[0].title}: aus Merkliste entfernen` }),
    );
    expect(screen.getByText('Noch nichts auf deiner Merkliste.')).toBeInTheDocument();
  });
  it('does not crash on corrupted saved data', async () => {
    localStorage.setItem('weitblick:saved', '{"not":"an array"}');
    mount('/merkliste');
    expect(await screen.findByText('Noch nichts auf deiner Merkliste.')).toBeInTheDocument();
  });
  it('finishes, persists and clears a private reflection', async () => {
    const user = userEvent.setup();
    mount(`/thema/${stories[0].slug}`);
    await user.click(await screen.findByRole('radio', { name: 'Eher ja' }));
    await user.click(screen.getByRole('button', { name: /Weiter zum Gegenargument/ }));
    await user.type(
      screen.getByRole('textbox', { name: 'Dein Gegenargument und deine Gedanken' }),
      'Ohne Bus hilft ein günstiges Ticket wenig.',
    );
    await user.click(screen.getByRole('button', { name: /Noch einmal hinsehen/ }));
    await user.click(screen.getByRole('radio', { name: 'Ich brauche weitere Informationen.' }));
    await user.click(screen.getByRole('button', { name: /Reflexion abschließen/ }));
    expect(screen.getByText('Ein Gedanke weiter.')).toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem('weitblick:reflection:mobilitaet')!).completed).toBe(
      true,
    );
    await user.click(screen.getByRole('button', { name: 'Meine Antworten löschen' }));
    expect(JSON.parse(localStorage.getItem('weitblick:reflection:mobilitaet')!).note).toBe('');
  });
  it('allows selecting and clearing perspectives', async () => {
    const user = userEvent.setup();
    mount(`/thema/${stories[0].slug}`);
    const boxes = await screen.findAllByRole('checkbox');
    for (const box of boxes) await user.click(box);
    expect(screen.getByText('Wähle einen Blickwinkel aus.')).toBeInTheDocument();
    await user.click(boxes[0]);
    expect(screen.getByText(stories[0].perspectives[0].headline)).toBeInTheDocument();
    expect(screen.queryByText(stories[0].perspectives[1].headline)).not.toBeInTheDocument();
  });
  it('keeps archive searches source-specific and paginated', async () => {
    const user = userEvent.setup();
    mount('/archiv?quelle=nzz');
    await screen.findByText('Schlagzeilen im Originalwortlaut');
    expect(screen.getByRole('combobox')).toHaveValue('nzz');
    expect(screen.getAllByRole('listitem')).toHaveLength(30);
    await user.click(screen.getByRole('button', { name: 'Weitere 30 Einträge anzeigen' }));
    expect(screen.getAllByRole('listitem')).toHaveLength(60);
    await user.type(screen.getByRole('searchbox', { name: 'Archiv durchsuchen' }), 'unfindbar123');
    expect(screen.getByText('Keine Schlagzeile gefunden.')).toBeInTheDocument();
  });
  it('recovers from a network failure using retry', async () => {
    const user = userEvent.setup();
    vi.mocked(fetch).mockRejectedValueOnce(new Error('offline'));
    mount();
    expect(
      await screen.findByText('Die Inhalte sind gerade nicht erreichbar.'),
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Erneut versuchen' }));
    expect(
      await screen.findByRole('searchbox', { name: 'Themen durchsuchen' }),
    ).toBeInTheDocument();
  });
  it('shows loading before the repository responds', async () => {
    let resolve!: (value: Response | PromiseLike<Response>) => void;
    vi.mocked(fetch).mockImplementationOnce(
      () =>
        new Promise((r) => {
          resolve = r;
        }) as Promise<Response>,
    );
    mount();
    expect(screen.getByRole('status')).toHaveTextContent('Perspektiven werden geladen');
    resolve({ ok: true, json: async () => raw } as Response);
    await screen.findByRole('searchbox', { name: 'Themen durchsuchen' });
  });
  it('handles unknown routes and unknown topics', async () => {
    const view = mount('/thema/unbekannt');
    expect(await screen.findByText('Dieses Thema ist nicht dabei.')).toBeInTheDocument();
    view.unmount();
    mount('/unbekannt');
    expect(screen.getByText('Hier geht der Blick ins Leere.')).toBeInTheDocument();
  });
  it('keeps edits working when local storage is unavailable', async () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    mount();
    fireEvent.click(await screen.findByRole('button', { name: `${stories[0].title}: merken` }));
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: `${stories[0].title}: aus Merkliste entfernen` }),
      ).toHaveAttribute('aria-pressed', 'true'),
    );
    expect(screen.getByText(/Dein Browser verhindert das Speichern/)).toBeInTheDocument();
  });
});
