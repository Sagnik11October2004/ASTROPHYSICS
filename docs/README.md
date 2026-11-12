# Theoretical & Computational Astrophysics — Research Portfolio

Public academic research website for Sagnik Bhowmick, built with static HTML/CSS/JavaScript and GitHub Pages. The publishing root is **main /docs**.

**Website:** https://sagnik11october2004.github.io/ASTROPHYSICS/

## Research and interactive demonstrations

The page presents work in primordial minihalo collapse, dark-matter halo statistics, accretion and black-hole feedback, Monte Carlo Comptonization, and numerical fluid dynamics. All educational browser demonstrations are separately identified from production research simulation outputs.

### IllustrisTNG cosmological-evolution film

The homepage embeds the **IllustrisTNG TNG300 gas-density supercluster-formation movie** by D. Nelson via the [official TNG media pages](https://www.tng-project.org/media/). It starts roughly 500 Myr after the Big Bang and does **not** depict the CMB, z=499, or first Population III stars.

- Default: 1080p H.264; visitors can explicitly choose the larger 4K file.
- The video is an actual cosmological hydrodynamics visualization. Its 16:9 image is displayed without exaggerated cropping.
- The old NASA/MPE movie, unrelated YouTube fallback, invented CMB-to-today storyboard, epoch buttons, local `cosmic-story.js`, and dark-matter power-law chart have all been **removed from the current page**.
- If the official video host blocks access, the page provides a link to the original official TNG media library instead of displaying a different animation as if it were identical footage.

### 2D front-tracking fluid interface (new)

The interactive interface-advection exhibit under **#fluid-lab** adapts a benchmark in the author's public [Fluids/Front_Tracking.py](https://github.com/Sagnik11October2004/Fluids/blob/main/Front_Tracking.py).

- An initially circular interface with center (0.5, 0.75) and radius 0.15 moves in the divergence-free prescribed flow with streamfunction
  `ψ = cos(π t / 8) sin²(πx) sin²(πy)`.
- The flow amplitude changes sign at t = 4 and returns its integrated displacement to approximately zero at t = 8.
- The browser demonstration precomputes 400 steps at Δt = 0.02, with 180 tracked markers and the three-stage explicit integration coefficients taken from the prototype, evaluating the prescribed velocity at intermediate stages.
- Controls: play/pause, single-step, timeline scrubbing, reset, speed, velocity vectors, and marker visibility. The display includes interface area relative to its initial area and marker RMS displacement from initial positions.
- **Limitations:** this is a JavaScript adaptation, not a direct execution of Python/SciPy. Marker identities are preserved, but the source's SciPy spline redistribution is omitted. It is a kinematic interface-advection benchmark, **not** the full two-phase Navier–Stokes/free-surface solver with capillary, viscosity and pressure dynamics. The author's public `Fluids` repository presently includes this front tracker, a 2D wave demonstration, lid-driven cavity, Kelvin–Helmholtz and Rayleigh–Bénard examples; it does not publicly include a compiled C++ two-phase free-surface implementation.

### Black-hole accretion disk

`bh-relativistic.js` renders a high-DPI WebGL accretion-disk illustration with an inclination slider, differentially rotating texture and Doppler-beaming on/off comparison. It is **qualitative shading**, not Kerr ray-tracing or validated GRMHD data. The lazy loader activates it near the About section.

### AGN gas-density and feedback-cycle animation

The original self-hosted `docs/media/agn-cycle.mp4` is a 32-second 960×540 H.264 looping animation, with `docs/media/agn-cycle-poster.jpg` as its preview. Its inflow, delayed feedback, jets/winds and gas fallback are normalized and schematic, not a numerical galaxy-formation or GRMHD calculation. It has **not** been upscaled or re-rendered in this revision.

### Population III.1 stellar comparison

The Pop III.1 comparison embeds the original author-provided 15-second MP4 from Google Drive. The left-hand track illustrates normal metal-free stellar evolution, and the right-hand track illustrates a hypothetical sustained dark-matter-annihilation-heated pathway. A supermassive star is **conditional on adequate gas accretion and continued DM heating**.

Google Drive preview: https://drive.google.com/file/d/10GCHZUes3954PJawJc-R598l89ScF_7U/preview

The author must select **Share → General access → Anyone with the link → Viewer** for the video to play publicly. The repository does not contain the MP4.

### Monte Carlo Comptonization demo

`mcrt-single.js` shows one photon at a time in simplified homogeneous H or structured S0 plasma, with event sampling, rest-frame Klein–Nishina scattering, and incremental escape histogram. It is an educational browser adaptation and does not publish or execute the private production research framework.

## Author portrait

The About card links to the correct author's [LinkedIn profile](https://www.linkedin.com/in/sagnik-bhowmick-161668292/).

**Important:** LinkedIn does not expose a retrievable direct headshot to these integration tools. The displayed avatar is currently the owner's GitHub profile image with an initials fallback. To use the **exact LinkedIn headshot**, the author must supply the original JPEG/PNG or a verifiable direct image URL; do not silently substitute photographs of similarly named people. The site can then store a small optimized portrait under `docs/media/`.

## Files

- `docs/index.html` — page markup, sections and source attribution
- `docs/site-media.js` — real TNG film controls and lazy AGN video
- `docs/fluid-front-demo.js`, `docs/fluid-front-demo.css` — interactive front-tracking exhibit and portrait/black-hole controls
- `docs/bh-relativistic.js` — qualitative GPU accretion illustration
- `docs/mcrt-single.js` — browser Monte Carlo scattering demonstration
- `docs/cosmos.js`, `docs/script.js` — research filters, navigation and travel-map interactions
- `docs/lazy-research.js` — on-viewport research modules
- `docs/media/` — local video poster and AGN clip
- `docs/world-countries.svg` — self-hosted Natural Earth world map

The obsolete `docs/research-interactions.js`, `docs/research-interactions.css`, and `docs/cosmic-story.js` were deleted.

## Running locally

```bash
cd docs
python3 -m http.server 8000
```

Visit http://localhost:8000. The cosmological video and university portrait sources require an Internet connection, and some hosts may block hotlinking. Browsers that disable animation automatically are respected where possible. Manual QA should check source video playback (HD/4K and failure handling), mobile controls, the front-tracking animation and the WebGL renderer before relying on the public page as the primary research portfolio.
