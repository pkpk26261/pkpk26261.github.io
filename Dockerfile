# Keep the browser image and apps/drone-soccer/lan/package.json in sync.
FROM mcr.microsoft.com/playwright:v1.56.1-noble

ENV NODE_ENV=production \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PORT=10000 \
    DATA_DIR=/data \
    MAX_ARENAS=2

WORKDIR /app/apps/drone-soccer/lan
COPY apps/drone-soccer/lan/package.json ./package.json
RUN npm install --omit=dev --package-lock=false --no-audit --no-fund \
    && npm cache clean --force

WORKDIR /app
COPY index.html ./index.html
COPY apps/drone-soccer/index.html ./apps/drone-soccer/index.html
COPY apps/drone-soccer/assets/ ./apps/drone-soccer/assets/
COPY apps/drone-soccer/games/ ./apps/drone-soccer/games/
COPY apps/drone-soccer/cloud/help.html ./apps/drone-soccer/cloud/help.html
COPY apps/drone-soccer/lan/server.cjs apps/drone-soccer/lan/room.cjs apps/drone-soccer/lan/engine-runner.cjs apps/drone-soccer/lan/service-store.cjs apps/drone-soccer/lan/certificate-setup.cjs ./apps/drone-soccer/lan/
COPY apps/drone-soccer/lan/public/ ./apps/drone-soccer/lan/public/
COPY vendor/three-r128/ ./vendor/three-r128/
COPY vendor/tailwindcss-3.4.17/ ./vendor/tailwindcss-3.4.17/
COPY vendor/fontawesome-6.4.0/ ./vendor/fontawesome-6.4.0/
COPY vendor/drone-soccer-fonts/ ./vendor/drone-soccer-fonts/

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD node -e "require('node:http').get('http://127.0.0.1:'+(process.env.PORT||10000)+'/api/health',r=>{r.resume();process.exit(r.statusCode===200?0:1)}).on('error',()=>process.exit(1))"

CMD ["node", "apps/drone-soccer/lan/server.cjs", "--cloud"]
