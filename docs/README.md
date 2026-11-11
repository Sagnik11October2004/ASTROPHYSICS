# Sagnik Bhowmick — Theoretical & Computational Astrophysics

This GitHub Pages website is a redesign based on the previous personal academic site, **"Sagnik Bhowmick — Theoretical & Computational Astrophysics"**. It restores that site's major sections and visual language (dark astronomical colors with cyan and amber accents) rather than replacing them with a generic research portfolio.

## Website sections

- **Hero:** "Tracing the physics of the early universe." An original, animated Canvas illustration progresses across a five-stage cosmic-time background: primordial hot plasma, recombination/CMB, dark ages, cosmic dawn, and cosmic-web assembly. The epoch strip calls out that the sequence is illustrative and **not a linear time scale**.
- **01 / Profile:** Research philosophy, IISER Kolkata, integrated BS–MS Physics, DST INSPIRE.
- **02 / Selected Work:** Filterable research summaries for Population III.1/dark matter, Monte Carlo Comptonization, halo merger statistics, GADGET HDF5 I/O, and relativistic accretion.
- **03 / Experience:** Research and training timeline.
- **04 / Research Mentors:** Mentors and collaborators, with affiliations and research themes.
- **05 / Travel With Me:** Dark interactive map with country-level markers for India, Sweden, Denmark, Finland, Netherlands and Norway, plus country filters.
- **06 / Toolkit:** Simulation, scientific computing, and analysis.
- **Interactive lab:** Adjustable power-law dark-matter density graph, retained from the first GitHub design.
- **07 / CV:** Public academic summary and print-friendly browser view; detailed personal CV available by request.
- **Contact:** Email, GitHub, LinkedIn.

## Source files

- `docs/index.html`: text, navigation and semantic page structure
- `docs/styles.css`: initial typographic/layout baseline plus live density plot and animated black hole artwork
- `docs/revamp.css`: design system for the previous site's restored sections and cosmology background
- `docs/script.js`: theme toggle, old black-hole artwork, interactive density plot, navigation
- `docs/cosmos.js`: cosmic evolution canvas, research filters, dark Leaflet map, print profile
- `docs/favicon.svg`: vector browser icon

**Scientific note:** The cosmic-time backdrop and black-hole art are illustrations, **not outputs from scientific simulations**. The log–log density cusp is an educational normalized power law, not measured research data. The cosmic timeline is not proportional to cosmic time.

**Privacy and publication:** No private research repositories or unpublished datasets are embedded. Project descriptions are high-level. The full CV PDF is **not** made publicly downloadable; the website instead offers a public summary and an email request link.

## Enable the GitHub Pages site

1. Review and merge the pull request into `main`.
2. Open **Repository → Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Set branch **main** and directory **/docs**, then **Save**.
5. The anticipated Pages URL is `https://sagnik11october2004.github.io/ASTROPHYSICS/`.

The Pages URL is not confirmed live until the repository's Pages deployment succeeds.

## Preview locally

```bash
cd docs
python3 -m http.server 8000
```

Open `http://localhost:8000`.

The rest of the site requires only vanilla HTML/CSS/JS. The restored map uses Leaflet 1.9.4 and dark basemap tiles from CARTO (via external URLs), with OpenStreetMap/CARTO attribution. If Leaflet is blocked, a fallback message and the country list remain visible. Google Fonts use system-font fallbacks. The animation respects reduced-motion preferences and pauses when its canvas is offscreen.

## Custom domain

This site currently expects the default GitHub Pages path. A domain previously used for the personal site can be connected later once it is configured in GitHub Pages and the DNS provider; do not add a CNAME file until that migration is ready.
