#!/usr/bin/env bash
# Regenerate docs/assets/figures/k1.png (Fig. K1, centroid selection) from k1.js in this folder.
# Uses the figure-kit renderer of the lorrax_flowcharts project (Node + Playwright Chromium; KaTeX and the
# IBM Plex fonts load from their CDNs, so it needs the network). Override its location with $FLOWCHARTS.
#   bash figures_src/centroid-selection/make_k1.sh [out.png]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
kit="${FLOWCHARTS:-$HOME/projects/lorrax_flowcharts}"
out="$(realpath -m "${1:-$here/../../docs/assets/figures/k1.png}")"
cd "$kit"
node render/render.js "$here/k1.js" --out "$out"
