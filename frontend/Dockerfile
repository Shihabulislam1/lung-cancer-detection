# syntax=docker/dockerfile:1

# Install dependencies (including native build tools for any native modules)
FROM node:22-bookworm-slim AS deps
WORKDIR /app

# Install build tools for native Node modules (e.g., bcrypt, better-sqlite3)
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 make g++ \
  && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci

# Build the Next.js app (skip drizzle migrations by invoking next directly)
FROM node:22-bookworm-slim AS builder
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Use Next's local binary directly to avoid running package.json build script
RUN npx next build

# Production runtime image
FROM node:22-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV NEXT_TELEMETRY_DISABLED=1

# Copy only the standalone output and static assets
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
USER node

CMD ["node", "server.js"]
