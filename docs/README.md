# Sagnik Bhowmick — Theory & Simulations Portfolio

Official academic research website, hosted on free GitHub Pages from the `main` branch and `/docs` folder:

https://sagnik11october2004.github.io/ASTROPHYSICS/

The design is based on the earlier dark astronomy-themed personal webpage, preserving the Profile, Research, Experience, Mentors, Travel, Toolkit, CV and Contact sections. The site deliberately stresses **original theory, numerical implementation, simulations and verification**, not simulations alone.

## Research

Eight individual CV-grounded projects cover primordial minihalo collapse; GADGET-4 primordial chemistry and adaptive particle splitting; Comptonization and polarization; halo merger statistics from Shin-Uchuu/Phi-4096; analytical transonic accretion; AGN accretion and feedback; nonlinear fluid dynamics; and an independent CFD solver. Project cards describe the author's contributions in first person.

## Visualizations and video provenance

### Cosmological evolution: real simulation footage

**Default:** NASA's *Imagine the Universe!* hosts an MPE/V. Springel simulation movie about structure formation, gravitational clustering, and galaxies. The primary source is `https://imagine.gsfc.nasa.gov/Videos/news/darkenergy_anim2_QT.mp4`, with information and credits on the [NASA description page](https://imagine.gsfc.nasa.gov/descriptions/darkenergy_anim2_QT.html). NASA credits **MPE / V. Springel**. Its exact initial redshift is not verified as z=499; the website does not claim that.

**Backup when the primary movie fails:** the official Illustris Collaboration movie, *A Virtual Universe*, embedded from YouTube, with [Illustris Media](https://www.illustris-project.org/media/) and appropriate credit. Illustris begins at **z=127**, rather than z=499.

**User-selectable visualization:** the existing locally rendered six-stage 2D educational animation remains available for a *CMB prologue (z~1100)*, *z=499 initial conditions*, minihalo assembly, first stars, galaxies, and z=0. Its matter displacement is a stylized first-order Fourier/Zel'dovich calculation; the other effects are illustrations, not actual z=499 full-physics simulation data. **Never present the CMB stage as a frame from a film starting at z=499.**

### Black-hole feedback: self-hosted periodic density simulation video

The site embeds the author's own original generated `AGN_Density_Accretion_Feedback_Looping.mp4` as a self-hosted video, committed to:

- `docs/media/agn-cycle.mp4` — **32 seconds, 960×540, H.264/MP4**, 24 fps, two repeated 16-second periods
- `docs/media/agn-cycle-poster.jpg` — static poster frame extracted from the same original video

The video shows a time-varying projected gas-density field, a thin accretion disk whose outer radius tracks the relative disk mass, luminosity and jet/wind feedback that respond with finite delays, and later replenishment by gas fallback. The video was transferred byte-for-byte from the original generated MP4 to GitHub, verified against the Git blob SHA `8ac13ee6895bfde01779e8cf12991940b713d8bf`.

The old NASA AGN animation is no longer embedded in this section. Playback uses the compact 560px-wide, uncropped 16:9 player, native controls, autoplay muted and continuous loop. The video loads from the repository when the player approaches the viewport; no external video host or public Drive permissions are involved.

**Scientific caution:** the reservoir, disk mass, feedback, gas density and fallback evolution are normalized and schematic. This is a physically motivated, periodic visualization — not a self-consistent GRMHD or galaxy-formation simulation.

### Original Pop III.1 comparison video — unchanged

**This section now embeds the exact 15-second original MP4 that the user uploaded**, rather than an animation recreated from the storyboard. The uploaded file `popiii_dm_heating_comparison (1).mp4` was transferred *unchanged* into the author's connected Google Drive account with name `popiii_dm_heating_comparison_exact.mp4`.

Google Drive file: https://drive.google.com/file/d/10GCHZUes3954PJawJc-R598l89ScF_7U/view

Embedded player: https://drive.google.com/file/d/10GCHZUes3954PJawJc-R598l89ScF_7U/preview

**Public playback limitation:** Google Drive reported `shared=false` immediately after upload. Only the file's owner can grant public link access using their Google Drive UI. For visitors to see the exact original video on the public site: open the file → **Share → General access → Anyone with the link → Viewer → Done**. The present connector does not allow changing a Drive file's permission to public/anyone, so no public playback claim is made until this is done.

The Pop III.1 dark-matter annihilation heating scenario remains scientifically conditional, not a guaranteed outcome.

### Monte Carlo radiative transfer — source-based educational browser port

A new standalone module `docs/mcrt-single.js` implements a **fresh, self-contained JavaScript port of the physics algorithms** inspected in the author's private GitHub repository `Sagnik11October2004/nordita-mcrt`, including:

- `src/transport.c`: exponential free-path / optical-depth sampling `tau=-ln(U)`, flight and boundary-escape logic.
- `src/physics.c`: relativistic photon Lorentz boosts between the lab and electron rest frame; exact electron-rest-frame Klein–Nishina recoil and angular rejection sampling.
- Maxwell–Jüttner thermal electron proposals, energy-dependent KN interaction weighting, and an educational nonthermal electron component.
- **Two selectable geometries only:** H homogeneous Maxwell–Jüttner plasma and S0 with static thermal plasmoids plus nonthermal interblob electrons.
- **One photon visible at any time:** show all flight segments and collisions, photon energy changes, packet escape, then start the next photon. Each escaping packet contributes exactly one count to the cumulative log-energy histogram and smoothed visual guide. Play/pause, speed 1×/2×/4×, next photon and reset are available.

The **original research code is not republished**. The visualization is not a compiled run of the complete production C/MPI engine, does not include polarized Stokes transport or full orbital/gravitational plasma dynamics, and must **not** be presented as a quantitatively validated spectrum or scientific result.

### Mentor portraits

The Research Mentors section includes institutional links and images where available. The Naoki Yoshida portrait now uses a source from Research Organization of Information and Systems as the primary image, with university and other public academic fallbacks; if all sources block hotlinking, accessible initials remain visible.

### World map

Bundled [Natural Earth 1:110m country map](https://github.com/nvkelso/natural-earth-vector) at `docs/world-countries.svg`, with country boundaries/names only. No external tile service, API key or map-library dependency.

## Main website files

| File | Role |
|---|---|
| `index.html` | Page content, research cards and media source links |
| `styles.css`, `revamp.css`, `upgrade.css`, `fix.css`, `astrolab.css`, `media-revision.css` | Responsive design |
| `script.js` | Base interactions and black-hole artwork fallback |
| `bh-relativistic.js` | Qualitative GPU accretion-disk shading |
| `cosmos.js` | Research filters and map controls |
| `cosmic-story.js` | Optional illustrated CMB-to-today timeline, loaded only on demand |
| `site-media.js` | On-viewport NASA video loading, source fallback and model toggles |
| `mcrt-single.js` | One-photon-at-a-time source-informed KN scattering experiment |
| `enhancements.js` | Mentor photo fallback chain and project accessibility |
| `lazy-research.js` | Defers WebGL and Compton Monte Carlo engine until sections approach viewport |
| `performance.css` | Responsive single-photon toolbar and media layout fixes |
| `world-countries.svg` | Country-labelled, self-hosted world map |

## Local preview

```bash
cd docs
python3 -m http.server 8000
```

Open `http://localhost:8000`. External videos and institutional photos require Internet connectivity and may be blocked by source-site policies; an error message links to NASA if the AGN film cannot play. Respect reduced-motion preferences and use the playback controls as needed.

## Deployment

GitHub Pages is already configured to deploy from `main /docs`. Merge the feature pull request to publish. Refresh with Ctrl+Shift+R if cached website CSS or JavaScript prevents the update from appearing.

## November 11, 2026 — performance and one-photon revision

Performance was improved by removing the 15 MB 4K, 4000×4000 NASA jet animation from the page's initial media load and replacing it with NASA's 3.3 MB WebM quasar-wind feedback animation. The AGN video now has `preload=none`, `data-src` media sources, a static poster, and viewport-triggered source activation with offscreen pausing. The removed AGN Canvas model no longer loads; the optional cosmology Canvas model initializes only on demand, and the WebGL disk and MCRT module are deferred until their sections approach the viewport. The original Pop III.1 Google Drive player remains embedded and lazily loaded. These changes reduce work and requests at initial paint; they do not eliminate delays caused by external video hosts or the private Google Drive permissions.

The new Compton transport page shows one photon at a time following event-sampled optical depth with energy-dependent Klein–Nishina scattering and electron-frame Compton recoil, in either H or S0. The cumulative plot builds **one escaped-photon contribution at a time**. The remaining public site does not load the retired old multi-packet demo. The spectra are educational finite-sample histograms, not fitted scientific results.

## November 12, 2026 — self-hosted AGN density cycle

The locally produced 32-second AGN density cycle and its poster are stored in `docs/media/`. The website no longer embeds the NASA wind movie for the AGN section; instead, it plays the original generated looping H.264 file with a download link. The AGN playback controller still loads and pauses it according to viewport visibility. No other research sections or pages were changed.
