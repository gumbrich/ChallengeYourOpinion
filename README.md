# Challenge Your Opinion → WEITBLICK

**Versteh mehr als eine Seite.** A German-language reading experience for comparing perspectives without assigning permanent political labels to news outlets.

The original project collected German-language headlines and explored word frequencies, LDA and clustering. This repository now includes a responsive React application while retaining the original corpus and research tools.

## Run the website

Node.js 22.12+ (CI uses Node 24), npm, and Python 3.10+ for corpus export.

```sh
cd web
npm ci
npm run dev
```

Open the local URL printed by Vite. To verify and produce a deployable build:

```sh
npm test
npm run format:check
npm run build
npm run preview
```

`web/dist/` is a static SPA. Host it at the domain root and rewrite **page routes** to `index.html` (including `/thema/*`), but let missing `/data/` and asset URLs return 404. No backend, API keys, external font requests or account are needed. `public/_redirects` provides the common Netlify-style history fallback. Other hosts need an equivalent configuration. GitHub Pages subpath deployment is not preconfigured.

## What is implemented

- Editorial landing/feed with category filters, shareable search parameters and a responsive navigation.
- Six illustrative topic dossiers, with selectable side-by-side perspectives, evidence gaps and blind spots.
- Local reading list and a three-step private reflection, with editable/deletable answers and graceful storage failure handling.
- Source context pages and a searchable, paginated archive of 3,698 genuine repository records.
- Loading, empty, retry, unknown-route and error-boundary states.
- Keyboard focus styles, skip link, native form labels, reduced-motion support, mobile layouts, German metadata and per-route browser titles/descriptions.
- Original SVG editorial illustrations and self-hosted Inter/Newsreader fonts.

## Data honesty and methodology

**This is a working product prototype, not a live news service.** Demo stories and the three fictional outlets (Gesellschaftslabor, Wirtschaftsfenster, Praxisblick) are labelled throughout. Nothing in these examples is attributed to NZZ, SPIEGEL, taz or FOCUS.

Colors mean *society/participation*, *costs/economy*, and *practical implementation*. They do not mean left/center/right, credibility or factual correctness. Demo bars count authored perspectives; they are not measurements of real media coverage.

The historical CSVs contain headlines, some collection dates, and noise such as navigation/advertising. They do **not** contain article URLs, publication dates, authors, verified political assessments or ownership data. Missing values remain missing. No links or source ratings are fabricated. The corpus is historical and not representative.

Rebuild the browser data from the existing files:

```sh
python3 scripts/export_archive.py
python3 scripts/seed_demo.py
python3 scripts/create_illustrations.py
```

The latter two scripts reproduce the authored demo content and original illustrations. Export normalizes whitespace and removes exact normalized duplicates within each source. Record IDs are deterministic hashes of source and normalized headline. Collection dates are never treated as publication dates. Original CSVs are unchanged.

## Architecture

```text
web/src/data/schema.ts        Runtime-validated Story / Perspective / ArchiveRecord contracts
web/src/data/repository.ts    NewsRepository adapter; fetch + validation, no UI-specific fixtures
web/src/data/sources.ts       Historical source directory with explicitly unknown assessments
web/src/components/          Reusable layout, article cards, controls, feedback, reflection
web/src/pages/               Feed, detail, archive/source directory, method/privacy
web/src/hooks.ts             Abortable resource loading and validated local persistence
web/public/data/             Versioned JSON from the corpus exporter and demo seed
scripts/                     Reproducible data and illustration generation
```

React + TypeScript + Vite were added because there was no existing web application to preserve. React Router supplies deep links and URL-based filters. Zod validates the data boundary and local storage. The UI consumes `NewsRepository`; replace its implementation with an API without rebuilding pages. The current `kind: "demo"` contract intentionally cannot silently represent live reporting: add a separate reviewed live-story schema before introducing real data.

State is scoped to this browser. `weitblick:saved`, `weitblick:onboarded` and `weitblick:reflection:<story-id>` are local-storage keys. No political profiling, tracking or third-party requests are implemented. Shared-browser users can see stored reflections; explain this in onboarding/privacy information (already included).

Shared tokens and reusable class patterns live in `styles.css`. Text headings use Newsreader, UI text uses Inter; both are served locally under their SIL Open Font Licenses. Lucide provides icons. No Ground News branding or assets are used.

## Original Python research

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python clustering.py
.venv/bin/python plot_frequencies.py
.venv/bin/python lda.py
```

Run LDA from the repository root. Analysis scripts write plots; their results are exploratory, **not** verified editorial or political classifications. The historical plots in the repo are retained, and do not claim to reflect the updated analysis.

Scraping is explicit, with bounded request timeouts and atomic CSV writes:

```sh
.venv/bin/python scrape.py --once
.venv/bin/python scrape.py --interval 1800 --sources nzz spiegel taz
```

Selectors are historical and may no longer match publisher markup. Zero matches produce a warning and leave existing data unchanged. Verify publisher permissions and selectors before running collection. The app does not start a scraper. Imports no longer start network access or an endless CPU loop. Scraping uses the actual collection date each run and supports the old single-column FOCUS file.

Fixes include headline-level (rather than word-level) clustering, modern scikit-learn APIs, German stemming in LDA, corrected taz input, pandas 2-compatible row construction, complete document ranges, safe short-topic handling, reproducible model seeds, and local-only word-frequency analysis with return values for every source.

## Validation and next steps

CI verifies TypeScript/build, formatting, interaction/data tests, Python tests, syntax and reproducible JSON/SVG exports. The web tests cover filters, bookmarks, persistence corruption/failures, reflections, comparison selection, archive paging, loading, retry and unknown routes.

Before a real public launch:

1. Add permitted current feeds and preserve canonical article URLs, authors, publication/update times, genre and access conditions.
2. Implement event clustering behind the repository boundary. Validate matches; word similarity alone is not enough.
3. Add evidence-backed article-level annotations with rationale, uncertainty, reviewer, revision date and a corrections process. Separate source ownership, factual accuracy and political framing.
4. Add server rendering or prerendered topic pages for crawler-visible per-topic metadata/social previews. Current SPA metadata updates work in the browser; basic HTML metadata is shared by crawlers that do not execute JavaScript. Configure canonical URL, sitemap and social image for the chosen production domain.
5. Add operator contact/required publication information and document the chosen host's logging. Review publishing rights for the corpus and excerpts. Run broader assistive-technology and browser testing.

The current branch is intended for review; no production deployment is performed automatically.
