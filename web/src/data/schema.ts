import { z } from 'zod';
export const perspectiveSchema = z.object({
  id: z.string(),
  source: z.string(),
  focus: z.enum(['gesellschaft', 'wirtschaft', 'umsetzung']),
  headline: z.string(),
  argument: z.string(),
  evidence: z.string(),
  blindSpot: z.string(),
});
export const storySchema = z.object({
  id: z.string(),
  slug: z.string(),
  category: z.string(),
  title: z.string(),
  summary: z.string(),
  image: z.string(),
  readMinutes: z.number().positive(),
  kind: z.literal('demo'),
  question: z.string(),
  context: z.array(z.string()),
  unknowns: z.array(z.string()),
  perspectives: z.array(perspectiveSchema).min(2),
});
export const feedSchema = z.object({ version: z.literal(1), stories: z.array(storySchema) });
export type Story = z.infer<typeof storySchema>;
export type Perspective = z.infer<typeof perspectiveSchema>;
export const archiveSchema = z.object({
  version: z.literal(1),
  records: z.array(
    z.object({
      id: z.string(),
      source: z.string(),
      headline: z.string(),
      capturedAt: z.string().nullable(),
      sourceFile: z.string(),
      articleUrl: z.null(),
    }),
  ),
});
export type ArchiveRecord = z.infer<typeof archiveSchema>['records'][number];
export const focusLabels = {
  gesellschaft: 'Gesellschaft & Teilhabe',
  wirtschaft: 'Kosten & Wirtschaft',
  umsetzung: 'Praxis & Umsetzung',
};
