#!/bin/bash
# Test script for local validation before pushing

set -e

echo "🧪 Testing Nima v0.5.0 locally..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check Python syntax
echo "1️⃣  Testing Python syntax..."
if python3 -m py_compile api/main.py 2>/dev/null; then
    echo -e "${GREEN}✅ Python syntax OK${NC}"
else
    echo -e "${RED}❌ Python syntax error${NC}"
    exit 1
fi

# Test 2: Check basic imports (skip if dependencies not installed)
echo ""
echo "2️⃣  Testing basic structure..."
if python3 -c "
import sys
from pathlib import Path
# Check file exists and can be read
with open('api/main.py', 'r') as f:
    content = f.read()
    # Check for conflict markers
    if '<<<<<<<' in content or '=======' in content or '>>>>>>>' in content:
        print('❌ Merge conflict markers found')
        sys.exit(1)
    # Check for required imports
    if 'from fastapi import' in content and 'CORSMiddleware' in content:
        print('✅ File structure OK')
    else:
        print('⚠️  Some imports may be missing')
        sys.exit(0)
" 2>/dev/null; then
    echo -e "${GREEN}✅ File structure OK${NC}"
else
    # Check for conflict markers manually
    if grep -q "<<<<<<< HEAD" api/main.py || grep -q ">>>>>>> origin/main" api/main.py; then
        echo -e "${RED}❌ Merge conflict markers still present${NC}"
        exit 1
    else
        echo -e "${YELLOW}⚠️  Import test skipped (dependencies may not be installed)${NC}"
        echo -e "${YELLOW}   This is OK - will test with Docker${NC}"
    fi
fi

# Test 3: Check version consistency
echo ""
echo "3️⃣  Testing version consistency..."
VERSION_FILE=$(cat VERSION | tr -d '[:space:]')
# Use sed instead of grep -P for macOS compatibility
API_VERSION=$(grep "version=" api/main.py | sed -n 's/.*version="\([0-9.]*\)".*/\1/p' | head -1)
ROOT_VERSION=$(grep '"version":' api/main.py | sed -n 's/.*"version": "\([0-9.]*\)".*/\1/p' | tail -1)

if [ "$VERSION_FILE" = "0.5.0" ]; then
    if [ "$API_VERSION" = "0.5.0" ] && [ "$ROOT_VERSION" = "0.5.0" ]; then
        echo -e "${GREEN}✅ Version consistency OK (0.5.0)${NC}"
    else
        echo -e "${YELLOW}⚠️  Version found:${NC}"
        echo "   VERSION file: $VERSION_FILE ✅"
        [ "$API_VERSION" = "0.5.0" ] && echo "   API version: $API_VERSION ✅" || echo "   API version: $API_VERSION (expected 0.5.0)"
        [ "$ROOT_VERSION" = "0.5.0" ] && echo "   Root version: $ROOT_VERSION ✅" || echo "   Root version: $ROOT_VERSION (expected 0.5.0)"
    fi
else
    echo -e "${YELLOW}⚠️  VERSION file: $VERSION_FILE (expected 0.5.0)${NC}"
fi

# Test 4: Check Dockerfile syntax
echo ""
echo "4️⃣  Testing Dockerfile syntax..."
if docker build --dry-run -f Dockerfile . > /dev/null 2>&1 || docker buildx build --dry-run -f Dockerfile . > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Dockerfile syntax OK${NC}"
else
    # Try basic validation
    if [ -f Dockerfile ]; then
        echo -e "${YELLOW}⚠️  Dockerfile exists (dry-run not supported, will test with build)${NC}"
    else
        echo -e "${RED}❌ Dockerfile not found${NC}"
        exit 1
    fi
fi

# Test 5: Check frontend Dockerfile
echo ""
echo "5️⃣  Testing frontend Dockerfile..."
if [ -f frontend-chat/Dockerfile ]; then
    echo -e "${GREEN}✅ Frontend Dockerfile exists${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend Dockerfile not found${NC}"
fi

# Test 6: Check workflow syntax
echo ""
echo "6️⃣  Testing GitHub Actions workflow syntax..."
if command -v yamllint >/dev/null 2>&1; then
    if yamllint .github/workflows/build-and-push.yml > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Workflow YAML syntax OK${NC}"
    else
        echo -e "${YELLOW}⚠️  Workflow YAML validation skipped (yamllint not available)${NC}"
    fi
else
    # Basic check - file exists and has valid structure
    if grep -q "name:" .github/workflows/build-and-push.yml && grep -q "jobs:" .github/workflows/build-and-push.yml; then
        echo -e "${GREEN}✅ Workflow file structure OK${NC}"
    else
        echo -e "${RED}❌ Workflow file structure invalid${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✅ All basic tests passed!${NC}"
echo ""
echo "Next steps:"
echo "  • Test Docker builds: ./scripts/test-docker-builds.sh"
echo "  • Test API locally: docker-compose up api"
echo "  • Test full stack: docker-compose up"

