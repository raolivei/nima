#!/bin/bash
# Test Memory Engine API endpoints

set -e

API_URL="${API_URL:-http://localhost:8002}"

echo "🧪 Testing Memory Engine API endpoints..."
echo "API URL: $API_URL"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if API is running
echo "1️⃣  Checking if API is running..."
if curl -s "$API_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is running${NC}"
else
    echo -e "${RED}❌ API is not running${NC}"
    echo "   Start with: python -m uvicorn api.main:app --reload --port 8002"
    echo "   Or: docker-compose up api"
    exit 1
fi

# Test 2: Create a fact
echo ""
echo "2️⃣  Creating a fact..."
FACT_RESPONSE=$(curl -s -X POST "$API_URL/v1/memory/facts" \
    -H "Content-Type: application/json" \
    -d '{
        "fact": "User prefers dark mode",
        "category": "preferences",
        "source": "explicit",
        "confidence": 0.9
    }')

if echo "$FACT_RESPONSE" | grep -q '"id"'; then
    FACT_ID=$(echo "$FACT_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    echo -e "${GREEN}✅ Fact created: $FACT_ID${NC}"
    echo "   Response: $FACT_RESPONSE"
else
    echo -e "${RED}❌ Failed to create fact${NC}"
    echo "   Response: $FACT_RESPONSE"
    exit 1
fi

# Test 3: List facts
echo ""
echo "3️⃣  Listing facts..."
FACTS_LIST=$(curl -s "$API_URL/v1/memory/facts")
if echo "$FACTS_LIST" | grep -q '"fact"'; then
    COUNT=$(echo "$FACTS_LIST" | grep -o '"fact"' | wc -l | tr -d ' ')
    echo -e "${GREEN}✅ Listed $COUNT facts${NC}"
else
    echo -e "${YELLOW}⚠️  No facts found or error${NC}"
    echo "   Response: $FACTS_LIST"
fi

# Test 4: Get specific fact
echo ""
echo "4️⃣  Getting fact by ID..."
if [ -n "$FACT_ID" ]; then
    GET_RESPONSE=$(curl -s "$API_URL/v1/memory/facts/$FACT_ID")
    if echo "$GET_RESPONSE" | grep -q '"id"'; then
        echo -e "${GREEN}✅ Retrieved fact${NC}"
    else
        echo -e "${RED}❌ Failed to retrieve fact${NC}"
        echo "   Response: $GET_RESPONSE"
    fi
fi

# Test 5: Search facts
echo ""
echo "5️⃣  Testing semantic search..."
SEARCH_RESPONSE=$(curl -s -X POST "$API_URL/v1/memory/search" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "user preferences",
        "limit": 5
    }')

if echo "$SEARCH_RESPONSE" | grep -q '\[\]' || echo "$SEARCH_RESPONSE" | grep -q '"fact"'; then
    echo -e "${GREEN}✅ Search completed${NC}"
    echo "   Response: $SEARCH_RESPONSE"
else
    echo -e "${YELLOW}⚠️  Search may have failed${NC}"
    echo "   Response: $SEARCH_RESPONSE"
fi

# Test 6: Get profile
echo ""
echo "6️⃣  Getting user profile..."
PROFILE_RESPONSE=$(curl -s "$API_URL/v1/memory/profile")
if echo "$PROFILE_RESPONSE" | grep -q '"user_id"'; then
    echo -e "${GREEN}✅ Profile retrieved${NC}"
    echo "   Response: $PROFILE_RESPONSE"
else
    echo -e "${YELLOW}⚠️  Profile retrieval may have failed${NC}"
    echo "   Response: $PROFILE_RESPONSE"
fi

# Test 7: Delete fact
echo ""
echo "7️⃣  Deleting fact..."
if [ -n "$FACT_ID" ]; then
    DELETE_RESPONSE=$(curl -s -X DELETE "$API_URL/v1/memory/facts/$FACT_ID" -w "\n%{http_code}")
    HTTP_CODE=$(echo "$DELETE_RESPONSE" | tail -1)
    if [ "$HTTP_CODE" = "204" ]; then
        echo -e "${GREEN}✅ Fact deleted${NC}"
    else
        echo -e "${YELLOW}⚠️  Delete returned HTTP $HTTP_CODE${NC}"
    fi
fi

echo ""
echo -e "${GREEN}✅ API tests completed!${NC}"
echo ""
echo "View API docs at: $API_URL/docs"

