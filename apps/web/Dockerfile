FROM node:20-alpine

# pnpm ships with corepack, which is bundled in Node 20. Enable it
# so the lock-file (``pnpm-lock.yaml``) is honoured exactly the way
# the dev workstation resolved it. The previous version of this
# Dockerfile used ``npm install`` with a wildcard for
# ``package-lock.json``; the lock file in this repo is actually
# ``pnpm-lock.yaml`` so npm fell back to a fresh install and
# occasionally produced a tree that didn't include ``next`` at the
# expected path.
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable && corepack prepare pnpm@9.15.0 --activate

WORKDIR /app

# Install deps first for better layer caching. Copy both the
# manifest and the lock-file so pnpm uses frozen resolution.
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Copy source
COPY . .

# Clean any stale Next.js build cache from the host (just in case)
RUN rm -rf .next

# Build
RUN pnpm run build

EXPOSE 3000

CMD ["pnpm", "run", "start"]
