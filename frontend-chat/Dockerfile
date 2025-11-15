FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build Next.js app
RUN npm run build

# Production image
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm install --omit=dev

# Copy built application
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/next.config.js ./

# Expose port
EXPOSE 3000

# Start Next.js
CMD ["npm", "start"]


