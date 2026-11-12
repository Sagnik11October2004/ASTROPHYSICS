# Theoretical & Computational Astrophysics — Research Portfolio

Public academic research website for Sagnik Bhowmick, built with static HTML/CSS/JavaScript and GitHub Pages. The publishing root is **main /docs**.

**Website:** https://sagnik11october2004.github.io/ASTROPHYSICS/

## Opening profile and chronological research journey

The first screen identifies the author as **Sagnik Bhowmick**, a fourth-year Integrated BS–MS Physics student at IISER Kolkata and a **DST INSPIRE Scholar**. A small portrait is displayed next to the programme, while the research summary names **black-hole accretion, feedback and seeding; large-scale structure formation; relativistic astrophysics; and radiative transport** rather than listing host institutions. The About section below contains only research narrative, academic facts, contact links and the interactive black-hole visual; duplicate portrait, LinkedIn and academic-graphic cards have been removed.

The **#experience** section uses an alternating, responsive zigzag timeline, following the author's requested progression: **Soumitro Banerjee → Sandeep Kumar Kataria → Indranil Chattopadhyay → Tomoaki Ishiyama → Nordita (Alexandra Veledina) and Chalmers CASSUM (Jonathan C. Tan, Naoki Yoshida, Shingo Hirano) as a shared final stage**. The final stage contains two companion research cards; on smaller screens the entire timeline becomes an accessible vertical sequence.

### Greeting, photograph and simulation layout

The homepage headline now reads **“Hi, I am Sagnik Bhowmick.”** The academic degree and department line appears directly beneath it. On desktop, the introduction and qualifications occupy the left column, the **complete original Nobel Prize Museum photograph** is positioned to their **right**, and the **full-width IllustrisTNG TNG300 film** is placed in the next row below both. On phones, the content stacks naturally as introduction → photo → film, retaining all existing video playback and quality controls. The movie still loads as the 5 MB optimized fast-start preview, with the original 1080p/4K modes selectable. The photo remains uncropped and clickable at full resolution.

## Uploaded portrait and academic contact details

The hero now uses a **full-frame, 1536 × 1152 WebP version of the user's original photograph outside the Nobel Prize Museum**, saved as `docs/media/sagnik-museum-full.webp` (227,718 bytes). Unlike the previous 4 KB cropped thumbnail (`docs/media/sagnik-portrait.webp`), the new image contains the complete original composition, including the museum facade and sign. It is displayed at up to 460 CSS pixels wide (full 4:3 proportions, no `object-fit: cover` cropping), scales responsively, and opens the full-resolution image when clicked. The photo remains next to the academic introduction, while the IllustrisTNG video stays in its own column. The two redundant About profile cards remain removed. The original uploaded 1536 × 1152 JPEG remains available in the conversation; the website uses a high-quality WebP conversion for efficient loading.

Both public contact addresses are linked in the top introduction, About panel, and contact section: institutional `sb23ms007@iiserkol.ac.in` and personal `sagnikbh7@gmail.com`. No phone number or personal location is published. The department is **Department of Physical Sciences**, IISER Kolkata; programme is **Integrated BS–MS Physics**; status is **DST INSPIRE Scholar**.

The existing 1920×1032 Pop III.1 comparison video is displayed with `object-fit: contain` at the full native aspect ratio, including browser fullscreen; the original MP4 is unmodified.

## Research and interactive demonstrations

The page presents work in primordial minihalo collapse, dark-matter halo statistics, accretion and black-hole feedback, Monte Carlo Comptonization, and numerical fluid dynamics. All educational browser demonstrations are separately identified from production research simulation outputs.

### IllustrisTNG cosmological-evolution film

The homepage embeds the **IllustrisTNG TNG300 gas-density supercluster-formation movie** by D. Nelson via the [official TNG media pages](https://www.tng-project.org/media/). It starts roughly 500 Myr after the Big Bang and does **not** depict the CMB, z=499, or first Population III stars.

- **Default:** self-hosted, fast-start **720p H.264** re-encoding of D. Nelson's actual **71-second TNG300 supercluster** movie (5,022,473 bytes), with a tiny **33,077-byte poster** and MP4 metadata placed at the start. The original 1080p download is **116,810,584 bytes**, around 23× larger.
- The original **1080p and 4K versions** remain available on demand via the video-quality dropdown. The default preview runs at 2× playback speed (user-selectable 1×–3×), with the entire original 71-second evolution and temporal order retained (at a reduced 24 fps).
- GitHub Actions workflow `.github/workflows/build-tng-preview.yml` downloads the author-published original, transcodes it via ffmpeg and commits the video and source attribution in `docs/media/TNG300-PREVIEW-SOURCE.txt`.
- This remains actual cosmological hydrodynamics footage, with no simulated/reconstructed replacement. The full 16:9 image is shown with `object-fit: contain`.
- The old NASA/MPE movie, unrelated YouTube fallback, invented CMB-to-today storyboard, epoch buttons, local `cosmic-story.js`, and dark-matter power-law chart have all been **removed from the current page**.
- The local preview greatly reduces start-up bandwidth; the interface has a visible loading indicator and, if the local preview fails, attempts the original 1080p source. If neither source plays, the official TNG media library link remains available.

- Visitors can choose **1×, 1.5×, 2× or 3×** speed; the default is **2×**. The HTML5 `playbackRate` control is reapplied after resolution switching; the original full-resolution source files themselves remain unmodified.

### Original C++ REAL_FLUID numerical solver in the browser (WebAssembly)

The **#fluid-lab** section runs the actual numerical kernels from [REAL_FLUID](https://github.com/Sagnik11October2004/REAL_FLUID) (GPL-3.0), compiled to WebAssembly using Emscripten and the reproducible GitHub Actions workflow `.github/workflows/build-real-fluid-wasm.yml`. Its source is pinned to revision `8f3b2d3e0e8e43945e2194546934564e0414c7e7`.

Unlike the earlier independent JavaScript demonstration, this build **uses the original C++ physics routines**, including:
- MAC staggered-grid velocities and a cell-centered signed-distance level set;
- semi-Lagrangian velocity/interface advection and Catmull–Rom (or linear) interpolation;
- gravitational forcing, curvature-dependent surface tension and explicit viscosity;
- the original pressure Poisson equation and selectable CG/ICCG/**MICCG** solvers (MICCG is the default);
- geometric redistancing, liquid-only pressure projection and the original feedback-based volume correction.

**Build adaptations**, recorded in `docs/native-fluid/prepare_source.py` and `docs/native-fluid/bridge.cpp`, remove OpenGL/GLUT drawing and verbose logging, replace the image with a responsive HTML Canvas drawn from the original C++ solution fields, expose parameter and solver selections, and limit the website selector to the two requested original cases. This is not a rewritten JavaScript approximation. The original C++ uses a fixed time step Δt=0.01 and 64² is selected as the default web grid; the desktop default is 128².

**Website cases:** only original dam break (`x<0.4 && y<0.6`) and droplet impact (`sqrt((x-0.5)²+(y-0.8)²)<0.15 || y<0.2`). The original repository also defines a flat-layer initial condition, but it is not selectable on this website; the experimental cavity was removed from the WebAssembly build.

**Controls:** run, pause, step, restart, the two cases, gravity, original σ, CG/ICCG/MICCG pressure solver, interpolation method, geometric redistancing, volume correction, velocity overlay and grid overlay. Live displays include C++ pressure solver residual, fluid area and simulated time. A recording of the original desktop application is available in a collapsed player.

**Deployment and licensing:** `docs/native-fluid/fluid-core.js` and `.wasm` are generated by Actions from the pinned source; the matching GPL-3.0 corresponding sources (including the build patches and bridge code) are published in `docs/native-fluid/corresponding-source/`. If a build fails or binaries have not been committed, the website shows a failure message rather than silently running the simplified JavaScript demonstration.

The native solver still needs quantitative convergence tests against controlled benchmark solutions. Running original code and passing a few time-step smoke tests is not a substitute for physical validation.

### Black-hole accretion disk

`bh-relativistic.js` renders a high-DPI WebGL accretion-disk illustration with an inclination slider, differentially rotating texture and Doppler-beaming on/off comparison. It is **qualitative shading**, not Kerr ray-tracing or validated GRMHD data. The lazy loader activates it near the About section.

### AGN gas-density and feedback-cycle animation

The original self-hosted `docs/media/agn-cycle.mp4` is a 32-second 960×540 H.264 looping animation, with `docs/media/agn-cycle-poster.jpg` as its preview. Its inflow, delayed feedback, jets/winds and gas fallback are normalized and schematic, not a numerical galaxy-formation or GRMHD calculation. It has **not** been upscaled or re-rendered in this revision.

### Population III.1 stellar comparison

The Pop III.1 comparison uses the author's **exact original 15-second 1920×1032 H.264 MP4**, now self-hosted as `docs/media/popiii_dm_heating_comparison_exact.mp4` rather than an embedded private Google Drive player. The video was copied **without cropping, re-rendering, or transcoding**: the published Git blob hash `298f929f6e268765c358b1301116582b69299574` matches the original file exactly (3,360,554 bytes).

The HTML5 player uses the true `1920/1032` aspect ratio, native playback controls, an accessible fullscreen button, and a direct download link. It can play publicly without requiring visitors to log into the author's Google account. The left-hand track illustrates ordinary metal-free stellar growth, and the right-hand track illustrates a hypothetical sustained dark-matter-annihilation-heated pathway. Formation of a supermassive star is **conditional on continued gas accretion and sufficient dark-matter heating**.

### Monte Carlo Comptonization demo

`mcrt-single.js` shows one photon at a time in simplified homogeneous H or structured S0 plasma, with event sampling, rest-frame Klein–Nishina scattering, and incremental escape histogram. It is an educational browser adaptation and does not publish or execute the private production research framework.

## Author portrait

The site's small hero portrait comes from the author's uploaded photograph; it is not a generic GitHub/LinkedIn avatar. LinkedIn remains available as a standard link among the other social/contact links, but the separate LinkedIn profile card has been removed.

## Files

- `docs/index.html` — page markup, sections and source attribution
- `docs/site-media.js` — real TNG film controls and lazy AGN video
- `docs/native-fluid/ui.js` / `bridge.cpp` / `prepare_source.py` — full C++ WebAssembly simulator, browser UI, build glue
- `docs/levelset-fluid.css` — responsive CFD UI, profile and black-hole controls
- `docs/profile-journey.css`, `docs/profile-journey.js` — academic introduction, research journey, original-video layout and fullscreen control
- `docs/bh-relativistic.js` — qualitative GPU accretion illustration
- `docs/mcrt-single.js` — browser Monte Carlo scattering demonstration
- `docs/cosmos.js`, `docs/script.js` — research filters, navigation and travel-map interactions
- `docs/lazy-research.js` — on-viewport research modules
- `docs/media/` — self-hosted original Pop III.1 comparison MP4, AGN cycle and poster
- `docs/world-countries.svg` — self-hosted Natural Earth world map

Obsolete research-interaction scripts, illustrated cosmology code, front-tracking and simplified JavaScript fluid solver have been removed.

## Running locally

```bash
cd docs
python3 -m http.server 8000
```

Visit http://localhost:8000. The cosmological video and university portrait sources require an Internet connection, and some hosts may block hotlinking. Browsers that disable animation automatically are respected where possible. Manual QA should check Pop III.1 MP4 playback and fullscreen at the native aspect ratio, source TNG movie handling, mobile timeline layout, native fluid solver controls and the WebGL renderer before relying on the public page as the primary research portfolio.

## Page ordering and selected cases

The interactive fluid laboratory is placed **after** all three astrophysics animation sections (AGN accretion/feedback, Population III.1 star formation, and Monte Carlo Comptonization), immediately before the CV. Its only visible scenarios are Dam Break and Droplet Impact, each using the original REAL_FLUID numerical engine. The native WebAssembly build also limits cycling to these two initial conditions. This does not modify the original upstream REAL_FLUID repository.

## Native build validation

The Emscripten workflow compiles the exact original numerical kernels with headless graphics and instruments their output. Its Node smoke test initializes both website-enabled original C++ cases and checks for finite level-set, velocity, liquid-volume and pressure-residual values. The website remains in a clear unavailable state if the build is missing.

