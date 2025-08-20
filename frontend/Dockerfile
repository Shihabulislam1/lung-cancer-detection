# Stage 1: Base Node.js image
FROM node:20-alpine AS base

# Install required native dependencies
RUN apk add --no-cache libc6-compat

# Stage 2: Install dependencies
FROM base AS deps
WORKDIR /app/frontend

# Copy only package.json and lockfile to install dependencies
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies with peer dependency workaround
RUN npm ci --legacy-peer-deps

# Stage 3: Build the application
FROM base AS builder
WORKDIR /app/frontend

# Copy installed modules
COPY --from=deps /app/frontend/node_modules ./node_modules

# Copy all source files
COPY frontend/ .

# Optional: Disable Next.js telemetry
ENV NEXT_TELEMETRY_DISABLED=1

# Build the app
RUN npm run build

# Stage 4: Production runner
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3003
ENV HOSTNAME=0.0.0.0

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy required files for standalone deployment
COPY --from=builder /app/frontend/public ./public
#COPY --from=builder --chown=nextjs:nodejs /app/frontend/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/frontend/.next/static ./.next/static

USER nextjs

EXPOSE 3003

# Start the server
CMD ["node", "server.js"]
