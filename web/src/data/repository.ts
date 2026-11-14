import { archiveSchema, feedSchema } from './schema';
import type { Story, ArchiveRecord } from './schema';
export interface NewsRepository {
  getStories(signal?: AbortSignal): Promise<Story[]>;
  getArchive(signal?: AbortSignal): Promise<ArchiveRecord[]>;
}
async function getJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Daten konnten nicht geladen werden (${response.status}).`);
  return response.json();
}
// Replace this implementation with an API adapter; pages depend only on the contract.
export const newsRepository: NewsRepository = {
  async getStories(signal) {
    return feedSchema.parse(await getJson('/data/stories.json', signal)).stories;
  },
  async getArchive(signal) {
    return archiveSchema.parse(await getJson('/data/archive.json', signal)).records;
  },
};
export function filterStories(stories: Story[], query: string, category: string, saved?: string[]) {
  const words = query.toLocaleLowerCase('de').trim().split(/\s+/).filter(Boolean);
  return stories.filter(
    (s) =>
      (category === 'Alle Themen' || s.category === category) &&
      (!saved || saved.includes(s.id)) &&
      words.every((w) =>
        `${s.title} ${s.summary} ${s.category}`.toLocaleLowerCase('de').includes(w),
      ),
  );
}
