#!/bin/bash
# Test Docker builds locally (simulates GitHub Actions)

set -e

echo "🐳 Testing Docker builds locally..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get version
VERSION=$(cat VERSION | tr -d '[:space:]')
echo -e "${BLUE}Building images for version: ${VERSION}${NC}"
echo ""

# Test API build
echo -e "${BLUE}1️⃣  Building API image...${NC}"
if docker build \
    -t ghcr.io/raolivei/nima-api:test \
    -t ghcr.io/raolivei/nima-api:${VERSION} \
    -f Dockerfile \
    . > /tmp/nima-api-build.log 2>&1; then
    echo -e "${GREEN}✅ API image built successfully${NC}"
    docker images | grep nima-api | head -2
else
    echo -e "${RED}❌ API build failed${NC}"
    echo "Last 20 lines of build log:"
    tail -20 /tmp/nima-api-build.log
    exit 1
fi

echo ""

# Test Frontend build
echo -e "${BLUE}2️⃣  Building Frontend image...${NC}"
if docker build \
    -t ghcr.io/raolivei/nima-frontend:test \
    -t ghcr.io/raolivei/nima-frontend:${VERSION} \
    -f frontend-chat/Dockerfile \
    ./frontend-chat > /tmp/nima-frontend-build.log 2>&1; then
    echo -e "${GREEN}✅ Frontend image built successfully${NC}"
    docker images | grep nima-frontend | head -2
else
    echo -e "${RED}❌ Frontend build failed${NC}"
    echo "Last 20 lines of build log:"
    tail -20 /tmp/nima-frontend-build.log
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All Docker builds successful!${NC}"
echo ""
echo "Built images:"
echo "  • ghcr.io/raolivei/nima-api:test"
echo "  • ghcr.io/raolivei/nima-api:${VERSION}"
echo "  • ghcr.io/raolivei/nima-frontend:test"
echo "  • ghcr.io/raolivei/nima-frontend:${VERSION}"
echo ""
echo "To test the images:"
echo "  docker run -p 8002:8000 ghcr.io/raolivei/nima-api:test"
echo "  docker run -p 3002:3000 ghcr.io/raolivei/nima-frontend:test"

