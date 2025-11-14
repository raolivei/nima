#!/bin/bash
# Update GHCR token in Vault with a token that has read:packages scope
# This script helps you store a GitHub Personal Access Token in Vault

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔐 Updating GHCR token in Vault for nima${NC}"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

# Get Vault pod
echo -e "${GREEN}Finding Vault pod...${NC}"
VAULT_POD=$(kubectl get pods -n vault -l app.kubernetes.io/name=vault -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$VAULT_POD" ]; then
    echo -e "${RED}Error: Vault pod not found. Is Vault deployed?${NC}"
    exit 1
fi

echo -e "${GREEN}Found Vault pod: ${VAULT_POD}${NC}"

# Get Vault token
echo -e "${GREEN}Getting Vault token...${NC}"
VAULT_TOKEN=$(kubectl logs -n vault $VAULT_POD 2>/dev/null | grep "Root Token" | tail -1 | awk '{print $NF}')

if [ -z "$VAULT_TOKEN" ]; then
    echo -e "${YELLOW}Warning: Could not find root token in logs. Using 'root' as default for dev mode.${NC}"
    VAULT_TOKEN="root"
fi

# Prompt for GitHub token
echo ""
echo -e "${YELLOW}To pull images from GHCR, you need a GitHub Personal Access Token with 'read:packages' scope.${NC}"
echo -e "${YELLOW}Create one at: https://github.com/settings/tokens/new${NC}"
echo ""
echo -e "${BLUE}Required scopes:${NC}"
echo -e "  - read:packages (to pull container images)"
echo ""

# Check if token is provided as argument
if [ $# -eq 1 ]; then
    GITHUB_TOKEN="$1"
    echo -e "${GREEN}Using provided token${NC}"
else
    read -p "Enter GitHub Personal Access Token: " GITHUB_TOKEN
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${RED}Error: GitHub token is required${NC}"
    exit 1
fi

# Store token in Vault
echo ""
echo -e "${GREEN}Storing token in Vault at secret/canopy/ghcr-token...${NC}"
if kubectl exec -n vault $VAULT_POD -- sh -c "export VAULT_ADDR=http://127.0.0.1:8200 && export VAULT_TOKEN='${VAULT_TOKEN}' && vault kv put secret/canopy/ghcr-token token='${GITHUB_TOKEN}'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Token stored in Vault${NC}"
else
    echo -e "${RED}❌ Failed to store token in Vault${NC}"
    exit 1
fi

# Force ExternalSecret to sync
echo ""
echo -e "${GREEN}Forcing ExternalSecret to sync...${NC}"
kubectl annotate externalsecret ghcr-secret -n nima force-sync=$(date +%s) --overwrite 2>&1 | grep -v "Warning" || true

# Wait a moment for sync
sleep 3

# Verify secret was updated
echo ""
echo -e "${GREEN}Verifying secret was updated...${NC}"
if kubectl get secret ghcr-secret -n nima > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Secret 'ghcr-secret' updated successfully in nima namespace${NC}"
else
    echo -e "${YELLOW}⚠️  Secret not found. External Secrets Operator may need more time to sync.${NC}"
    echo -e "${YELLOW}Check status with: kubectl get externalsecret ghcr-secret -n nima${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${YELLOW}The deployment should now be able to pull images from GHCR.${NC}"
echo -e "${YELLOW}Check pods with: kubectl get pods -n nima${NC}"

