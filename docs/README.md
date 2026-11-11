# Sagnik Bhowmick — Theoretical & Computational Astrophysics

This GitHub Pages website is a redesign based on the previous personal academic site, **"Sagnik Bhowmick — Theoretical & Computational Astrophysics"**. It restores that site's major sections and visual language (dark astronomical colors with cyan and amber accents) rather than replacing them with a generic research portfolio.

## Website sections and updated features

- **Hero:** Six chapter animation from an earlier CMB prologue (z≈1100) to initial conditions at z=499, then rare minihalo collapse, first stars, galaxy assembly and today's z=0 cosmic web. Redshift is annotated; the displacement-field evolution is 2D and illustrative. Play/pause and scrubbing available.
- **Profile:** Research overview and a GPU-based accretion-disk illustration showing qualitative Doppler beaming, disk temperature gradients, a black-hole shadow and photon-ring-inspired lensing arcs. Adjust the **inclination** slider. This is **not** a general-relativistic ray-tracer, hydrodynamic run or scientific prediction. The original Canvas 2D animation remains as the fallback when WebGL is unavailable.
- **Selected Research:** **Eight separate CV-grounded project cards** showing the author's own specific contributions in primordial collapse; GADGET-4 chemistry/particle splitting; Monte Carlo Comptonization/polarization; halo statistics; transonic accretion; AGN feedback; nonlinear dynamics; independent two-phase CFD. Filter by topic.
- **Research Mentors:** Eight linked institutional or academic portraits, with source credit and initials fallback if a picture is blocked by its original host. The images are **hotlinked** from the respective institutions/professional sources, not copied into this repository. Review any third-party image reuse restrictions before publishing.
- **Travel With Me:** Self-contained Natural Earth political world map with real country boundaries and country-name labels, with no extra geographic labels. Click country pins and use pan/zoom/region filters. No third-party mapping services.
- **Research visualizations:** Separate black-hole accretion/AGN feedback duty cycle, four-stage Pop III.1 protostellar evolution comparison based on the supplied video, and a stochastic Monte Carlo photon Comptonization transport illustration. Experience, toolkit, CV and contact sections remain.

## Source files

| File | Purpose |
| --- | --- |
| `docs/index.html` | Semantic sections and CV-grounded text |
| `docs/styles.css` | Base typography and responsive styles |
| `docs/revamp.css` | Restored personal-site visual language |
| `docs/upgrade.css` | Detailed project cards, portraits and interactive controls |
| `docs/script.js` | Base navigation/theme, legacy optional density chart support, Canvas 2D BH fallback |
| `docs/cosmos.js` | Research filters, printable CV and offline atlas controls |
| `docs/bh-relativistic.js` | WebGL qualitative relativistic disk shading |
| `docs/enhancements.js` | Broken-image fallbacks and accessible project labels |
| `docs/cosmic-story.js` | Six-stage CMB prologue to z=0 cosmic-history animation |
| `docs/world.svg` | Superseded geographic vector map (retained for compatibility) |
| `docs/world-countries.svg` | Natural Earth political map with country labels |
| `docs/astrolab.js` | AGN, Pop III.1 and Comptonization animation logic |
| `docs/astrolab.css` | Animation layouts and world map refinements |
| `docs/fix.css` | High-contrast cosmic-story stage and self-contained atlas layout |
| `docs/favicon.svg` | Vector favicon |

**Scientific transparency:** These are performant browser visualizations. The hero is based on a linearized cosmological displacement scheme in two spatial dimensions; it does not solve the coupled gravity/hydrodynamics/radiation equations for the entire observable universe. The accretion-disk image is an approximate shader, not ray tracing in a Schwarzschild or Kerr metric. The earlier dark-matter-cusp graph has been replaced by a schematic AGN feedback-cycle animation.

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

The site requires only vanilla HTML/CSS/JS. The world geography is shipped with the repository as a self-contained Natural Earth SVG. No map tile libraries, API keys or remotely hosted map services are needed. Google Fonts have system-font fallbacks. The animation respects reduced-motion preferences and pauses when its canvas is offscreen.

## Custom domain

This site currently expects the default GitHub Pages path. A domain previously used for the personal site can be connected later once it is configured in GitHub Pages and the DNS provider; do not add a CNAME file until that migration is ready.

## Troubleshooting the map and animation

Previously, the world map relied on externally fetched Leaflet scripts and CARTO tiles. Some visitors saw a blank map when those resources did not load. The replacement bundles the coastline map with the website so it renders regardless of third-party tile availability. The original cosmological animation was replaced first with a bright panel; the current version adds redshift-anchored sequences through z=0. After merging these changes, use Ctrl+Shift+R to avoid cached scripts. GitHub Pages is already enabled on main/docs.


## November 2026 — cosmic-evolution and numerical-visualization update

- **Country-only world map:** The map now uses a bundled high-resolution political vector derived from the public-domain [Natural Earth 1:110m countries GeoJSON](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson). Country borders and names appear; prior ocean/grid/atlas labels have been removed. Geographic zoom/pan/selection are kept. `world-countries.svg` is self-hosted, with no externally hosted map services.
- **Hero animation:** Six chronological chapters, beginning with an *earlier* CMB reference at z~1100 followed by initial conditions at z=499, minihalo collapse, first stars, galaxy assembly and z=0. The 2D Fourier-displacement matter evolution is a physically motivated visualization. It does not solve Poisson+hydrodynamics self-consistently or reproduce a quantitative LCDM run. The CMB rendering, first stars and galaxy art are illustrative. Redshift evolves by chapter with labeled anchors; timeline playback rate is not proportional to physical time.
- **BH feedback:** The former dark-matter cusp plot has been replaced by an animated conceptual AGN cycle: gas supply, disk accretion, mechanical/radiative feedback, hot-gas suppression and renewed inflow. It is not a GRMHD model and does not imply every AGN is fully quenched.
- **Pop III.1 formation:** A side-by-side four-stage animated recreation based on the user's 15-second uploaded video `popiii_dm_heating_comparison (1).mp4`. The original video was reviewed locally, including its stages (formation, divergence, feedback, late growth). The source video is **not embedded**; its sequence informs the new independent Canvas illustration. Annihilation-heated supermassive protostar formation remains hypothetical, conditional on sustained heating and accretion. Avoid interpreting the displayed relative radii/meter values as simulation results.
- **Monte Carlo Comptonization:** Separate live packet-transport experiment with stochastic scattering, a qualitative energy-exchange kernel, changing colour as photon energy increases, escaping-packet histogram and geometry switches (H, S0 and Sv). It is not the user's production radiative-transfer solver or a calibrated physical spectrum. No unsupported quantitative polarization claims are made.
- **Code:** `astrolab.js` and `astrolab.css` implement the three interactive research animations. The title animation remains in `cosmic-story.js`; the existing `cosmos.js` controls the self-contained map and research filters.

### Privacy / publication

The full original Pop III.1 presentation video has not been republished to GitHub; no private datasets, collaborator material or unpublished figures were uploaded. The animations are new client-side educational illustrations grounded in the user's published CV narrative and provided storyboard.
