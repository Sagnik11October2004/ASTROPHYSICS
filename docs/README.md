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

### Level-set free-surface CFD: dam break, droplet impact and bubble experiment

The **interactive fluid laboratory** at `#fluid-lab` is now based on the author's correct C++ [REAL_FLUID repository](https://github.com/Sagnik11October2004/REAL_FLUID), *not* the unrelated Python `Fluids/Front_Tracking.py` benchmark.

- **Original C++ code:** [`src/liquid2D.cpp`](https://github.com/Sagnik11October2004/REAL_FLUID/blob/main/src/liquid2D.cpp), [`src/levelset2D.cpp`](https://github.com/Sagnik11October2004/REAL_FLUID/blob/main/src/levelset2D.cpp) and [`src/solver.cpp`](https://github.com/Sagnik11October2004/REAL_FLUID/blob/main/src/solver.cpp), published in a repository containing a GPL-3.0 LICENSE.
- **Original three initial conditions** (verified in the C++ source): dam break `x < 0.4 && y < 0.6`, droplet/pond `hypot(x-0.5,y-0.8) < 0.15 || y < 0.2`, and shallow equilibrium layer `y < 0.1`.
- **Website's experimental fourth case:** a gas cavity in a liquid layer, shown as *Bubble / cavity · experimental*. This is **not** an original C++ test case and is **not** an independently validated two-phase bubble-rise simulation.
- The C++ method uses cell-centered level sets, a staggered grid, gravity, semi-Lagrangian advection, explicit viscosity, a curvature-localized surface tension force, redistancing and a pressure-projection solver with a modified incomplete-Cholesky-preconditioned conjugate-gradient method.
- The website cannot run the native OpenGL/GLUT C++ executable directly. Instead, `docs/levelset-fluid.js` contains a self-contained **new JavaScript numerical adaptation** using a 64² grid; bilinear semi-Lagrangian transport; signed-distance front evolution; gravity, viscosity and capillary forcing; a simpler free-surface SOR Poisson solve; approximate redistancing; and small bounded area correction. These are not byte-for-byte or floating-point-equivalent results of the original C++ code.
- Visitors may switch cases, pause/restart, step through the solve, change gravity and a normalized capillary coefficient, adjust speed, toggle velocity vectors and grid display, and inspect relative liquid area and interior RMS divergence.
- The equilibrium demo starts with an extra *small perturbation* to make relaxation visible; the C++ code's original layer is initially flat. The experimental bubble cavity merges into the free surface and the short demonstration loops automatically. The browser visualization is **qualitative and should not be cited as a validated numerical solution**.
- The original repository contains source screenshots and a short recorded solver run under `images/`, linked on the website.

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
- `docs/levelset-fluid.js`, `docs/levelset-fluid.css` — level-set fluid experiments, profile and black-hole controls
- `docs/bh-relativistic.js` — qualitative GPU accretion illustration
- `docs/mcrt-single.js` — browser Monte Carlo scattering demonstration
- `docs/cosmos.js`, `docs/script.js` — research filters, navigation and travel-map interactions
- `docs/lazy-research.js` — on-viewport research modules
- `docs/media/` — local video poster and AGN clip
- `docs/world-countries.svg` — self-hosted Natural Earth world map

Obsolete research-interaction scripts, illustrative cosmology code and the prior front-tracking webpage have been removed.

## Running locally

```bash
cd docs
python3 -m http.server 8000
```

Visit http://localhost:8000. The cosmological video and university portrait sources require an Internet connection, and some hosts may block hotlinking. Browsers that disable animation automatically are respected where possible. Manual QA should check source video playback (HD/4K and failure handling), mobile controls, the front-tracking animation and the WebGL renderer before relying on the public page as the primary research portfolio.

## November 12, 2026 — level-set CFD integration checks

The fluid demo was executed using simulated DOM/Canvas controls for each of the four modes. After 20 and again after 128 integration steps, the reported density-independent area and interior divergence remained finite in every scenario. The area-correction stage is applied each step to limit numerical volume loss from semi-Lagrangian advection, particularly in the experimental cavity mode. Verified JS syntax for the current site files and verified controls and unique IDs in the HTML. Tests do not establish convergence, quantitative consistency with the original C++ code, or visual/browser compatibility on every device.
