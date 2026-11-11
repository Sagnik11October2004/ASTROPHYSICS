# Sagnik Bhowmick — Theoretical & Computational Astrophysics

This GitHub Pages website is a redesign based on the previous personal academic site, **"Sagnik Bhowmick — Theoretical & Computational Astrophysics"**. It restores that site's major sections and visual language (dark astronomical colors with cyan and amber accents) rather than replacing them with a generic research portfolio.

## Website sections and updated features

- **Hero:** A live, interactive, **two-dimensional Zel'dovich-approximation** particle-displacement calculation illustrates how small density perturbations grow into a structured cosmic web. A reproducible initial displacement field is generated from Fourier modes; particles evolve according to `x(q,a)=q+D(a)ψ(q)`. An evolving density raster shows the resulting structure. The timeline includes earlier plasma, recombination and first-star phases using **artistic visual overlays** because the Zel'dovich approximation does not simulate that microphysics. Controls allow play/pause and timeline scrubbing; the 33-second playback loop is deliberately not linear in cosmic time or redshift.
- **Profile:** Research overview and a GPU-based accretion-disk illustration showing qualitative Doppler beaming, disk temperature gradients, a black-hole shadow and photon-ring-inspired lensing arcs. Adjust the **inclination** slider. This is **not** a general-relativistic ray-tracer, hydrodynamic run or scientific prediction. The original Canvas 2D animation remains as the fallback when WebGL is unavailable.
- **Selected Research:** **Eight separate CV-grounded project cards** showing the author's own specific contributions in primordial collapse; GADGET-4 chemistry/particle splitting; Monte Carlo Comptonization/polarization; halo statistics; transonic accretion; AGN feedback; nonlinear dynamics; independent two-phase CFD. Filter by topic.
- **Research Mentors:** Eight linked institutional or academic portraits, with source credit and initials fallback if a picture is blocked by its original host. The images are **hotlinked** from the respective institutions/professional sources, not copied into this repository. Review any third-party image reuse restrictions before publishing.
- **Travel With Me:** Real CARTO/OpenStreetMap geographic map tiles showing a **global** world view, draggable map, zoom, country filters and an explicit reset-to-world-view button. The selected country markers are indicative rather than city-accurate visitor claims.
- **Experience, Toolkit, Interactive Lab, CV, Contact:** Prior sections and educational dark-matter cusp graph are preserved.

## Source files

| File | Purpose |
| --- | --- |
| `docs/index.html` | Semantic sections and CV-grounded text |
| `docs/styles.css` | Base typography and responsive styles |
| `docs/revamp.css` | Restored personal-site visual language |
| `docs/upgrade.css` | Detailed project cards, portraits and interactive controls |
| `docs/script.js` | Base navigation/theme, density graph, Canvas 2D black-hole fallback |
| `docs/cosmos.js` | Research filters, global map and Zel'dovich-inspired cosmic model |
| `docs/bh-relativistic.js` | WebGL qualitative relativistic disk shading |
| `docs/enhancements.js` | Broken-image fallbacks and accessible project labels |
| `docs/favicon.svg` | Vector favicon |

**Scientific transparency:** These are performant browser visualizations. The hero is based on a linearized cosmological displacement scheme in two spatial dimensions; it does not solve the coupled gravity/hydrodynamics/radiation equations for the entire observable universe. The accretion-disk image is an approximate shader, not ray tracing in a Schwarzschild or Kerr metric. The dark-matter plot is a normalized analytic power law, not measured research data.

**Privacy:** No private research code, unpublished datasets, personally identifying phone number or full CV PDF have been added. Research methods are described from the author's existing CV, and no ongoing paper is falsely described as published.

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
