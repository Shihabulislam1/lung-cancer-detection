# Stage 1: Base Node.js image
FROM node:20-alpine AS base

# Required for some native dependencies
RUN apk add --no-cache libc6-compat

# Stage 2: Install dependencies
FROM base AS deps
WORKDIR /app/frontend

# Copy dependency files
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Stage 3: Build the application
FROM base AS builder
WORKDIR /app/frontend

# Copy installed node_modules from deps stage
COPY --from=deps /app/frontend/node_modules ./node_modules

# Copy the full frontend source
COPY frontend/ .

# Build the Next.js app
RUN npm run build

# Stage 4: Production image
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Create non-root user
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

# Copy the output from the build stage
COPY --from=builder /app/frontend/public ./public
COPY --from=builder /app/frontend/.next/standalone ./
COPY --from=builder /app/frontend/.next/static ./.next/static

USER nextjs

EXPOSE 3000

# Run the standalone output
CMD ["node", "server.js"]
