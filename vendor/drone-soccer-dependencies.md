# Drone-soccer browser dependencies

The 32 game pages load these fixed local assets. No package upgrade or engine
change is involved. Three.js and Tailwind are the same bundles previously loaded
by the games and retained in the local validation cache.

| Directory | Upstream | License |
| --- | --- | --- |
| `three-r128/` | [Three.js r128](https://github.com/mrdoob/three.js/tree/r128), including OrbitControls and BufferGeometryUtils | [MIT](three-r128/LICENSE) |
| `tailwindcss-3.4.17/` | [Tailwind CSS 3.4.17](https://github.com/tailwindlabs/tailwindcss/tree/v3.4.17), Play CDN browser runtime | [MIT](tailwindcss-3.4.17/LICENSE) |
| `fontawesome-6.4.0/` | [Font Awesome Free 6.4.0](https://github.com/FortAwesome/Font-Awesome/tree/6.4.0), CSS and WOFF2 webfonts | [Font, icon and code licenses](fontawesome-6.4.0/LICENSE.txt) |
| `drone-soccer-fonts/` | [Inter and Share Tech Mono from Google Fonts](https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Share+Tech+Mono&display=swap) | [Inter OFL](drone-soccer-fonts/inter-OFL.txt), [Share Tech Mono OFL](drone-soccer-fonts/sharetechmono-OFL.txt) |

`drone-soccer-fonts/sources.json` records the individual font URLs; its CSS uses
local files. Font Awesome retains both WOFF2 assets and its TTF fallbacks.
License and attribution comments in the upstream assets are retained.

All runtime files must remain alongside the game when copying the project.
There is no runtime dependency on the `artifacts/` validation directory.
