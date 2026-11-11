# Sagnik Bhowmick — Research Portfolio

A responsive, dependency-free academic website featuring:
- Original Canvas 2D black-hole / accretion-disk artwork (an **artistic illustration**, not a GR ray-tracing simulation).
- An adjustable, scientifically labeled log–log dark-matter power-law cusp demonstration: `rho(r)/rho(1 pc) = (r/1 pc)^(-gamma)`. This is a **toy model**, not a simulation measurement.
- Research themes, verified links to public GitHub repositories, and high-level research experience.
- Responsive layout, light/dark themes, keyboard accessibility, and reduced-motion support.

## Host on GitHub Pages (free)

This directory intentionally lives inside the existing `ASTROPHYSICS` repository so its original notebooks and code remain untouched.

1. Merge the `feat/research-portfolio-site` pull request into `main`.
2. Open **Repository → Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select branch **main**, folder **/docs**, and click **Save**.
5. After GitHub completes deployment, the website should be available at:

   https://sagnik11october2004.github.io/ASTROPHYSICS/

GitHub Pages can take a short time to publish. The URL above is the expected address, not confirmation that it is currently live.

## Run locally

Open `index.html` directly in a browser; it uses no build tools or external JavaScript dependencies. Web fonts use Google Fonts with system-font fallbacks. For a local web server instead, run:

```bash
cd docs
python3 -m http.server 8000
```

Then visit http://localhost:8000.

## Customize

- `index.html`: text, projects, institutional descriptions and links.
- `styles.css`: color palette, layout and responsive styles.
- `script.js`: animated Canvas artwork, theme toggle and density-profile graph.
- `favicon.svg`: original vector browser icon.

Do not add unpublished results, private contact information or attribution claims without reviewing them first.

## Custom domain

A custom domain is optional. Configure it in **Settings → Pages** and set the required DNS records at your domain provider. Only add a `CNAME` file after the domain and DNS are correctly configured. DNS hosting requirements and DNS records depend on your chosen hostname.

All code is ordinary HTML, CSS and JavaScript and can also be moved to a separate `username.github.io` repository later.
