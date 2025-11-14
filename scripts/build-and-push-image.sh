#!/bin/bash
# Build and push nima-api Docker image to GHCR
# Usage: ./scripts/build-and-push-image.sh [tag]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default tag
TAG="${1:-v0.3.2}"
IMAGE_NAME="ghcr.io/raolivei/nima-api"

echo -e "${GREEN}🐳 Building and pushing nima-api Docker image${NC}"
echo -e "${YELLOW}Tag: ${TAG}${NC}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if we're logged into GHCR
if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}Not logged into Docker. Attempting to login to GHCR...${NC}"
    
    # Try to use gh CLI token, but note it may not have write:packages scope
    if command -v gh &> /dev/null; then
        GITHUB_TOKEN=$(gh auth token 2>/dev/null || echo "")
        if [ -n "$GITHUB_TOKEN" ]; then
            echo -e "${YELLOW}Attempting login with gh CLI token...${NC}"
            echo -e "${YELLOW}Note: If this fails, create a token with 'write:packages' scope at:${NC}"
            echo -e "${YELLOW}  https://github.com/settings/tokens/new${NC}"
            echo "$GITHUB_TOKEN" | docker login ghcr.io -u raolivei --password-stdin || {
                echo -e "${RED}Login failed. Please create a GitHub Personal Access Token with 'write:packages' scope${NC}"
                echo -e "${YELLOW}Then login manually:${NC}"
                echo -e "  echo YOUR_TOKEN | docker login ghcr.io -u raolivei --password-stdin"
                exit 1
            }
        else
            echo -e "${YELLOW}Please login to GHCR manually:${NC}"
            echo -e "  docker login ghcr.io -u raolivei"
            exit 1
        fi
    else
        echo -e "${YELLOW}Please login to GHCR manually:${NC}"
        echo -e "  docker login ghcr.io -u raolivei"
        exit 1
    fi
fi

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo -e "${GREEN}Building image...${NC}"
docker build -t "${IMAGE_NAME}:${TAG}" -t "${IMAGE_NAME}:latest" .

echo ""
echo -e "${GREEN}Pushing image with tag ${TAG}...${NC}"
docker push "${IMAGE_NAME}:${TAG}"

echo ""
echo -e "${GREEN}Pushing image with tag latest...${NC}"
docker push "${IMAGE_NAME}:latest"

echo ""
echo -e "${GREEN}✅ Successfully built and pushed:${NC}"
echo -e "  ${IMAGE_NAME}:${TAG}"
echo -e "  ${IMAGE_NAME}:latest"
echo ""
echo -e "${YELLOW}To update the Kubernetes deployment, run:${NC}"
echo -e "  kubectl set image deployment/nima-api api=${IMAGE_NAME}:${TAG} -n nima"

