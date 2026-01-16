# Testing Memory Engine

## Quick Test (Without Dependencies)

Test file structure and syntax:

```bash
python3 scripts/test-memory-engine.py
```

**Note**: This requires dependencies. If not installed, it will fail at import.

## Test with Docker Compose (Recommended)

1. **Start the API**:

```bash
source ../workspace-config/ports/.env.ports
docker-compose up api
```

2. **In another terminal, run API tests**:

```bash
./scripts/test-api-memory.sh
```

Or test manually:

```bash
# Health check
curl http://localhost:8002/health

# Create a fact
curl -X POST http://localhost:8002/v1/memory/facts \
  -H "Content-Type: application/json" \
  -d '{
    "fact": "User prefers dark mode",
    "category": "preferences",
    "source": "explicit",
    "confidence": 0.9
  }'

# List facts
curl http://localhost:8002/v1/memory/facts

# Search facts
curl -X POST http://localhost:8002/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user preferences",
    "limit": 5
  }'

# Get profile
curl http://localhost:8002/v1/memory/profile
```

## Test with Local Python (If Dependencies Installed)

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run test script**:

```bash
python3 scripts/test-memory-engine.py
```

3. **Start API**:

```bash
python -m uvicorn api.main:app --reload --port 8002
```

4. **Test endpoints** (use curl commands above)

## Expected Results

### Storage

- ✅ Facts stored in `data/memory/facts.json`
- ✅ Profile stored in `data/memory/profile.json`
- ✅ Embeddings generated and stored with facts

### API Endpoints

- ✅ `GET /v1/memory/facts` - Returns list of facts
- ✅ `POST /v1/memory/facts` - Creates new fact
- ✅ `GET /v1/memory/facts/{id}` - Returns specific fact
- ✅ `DELETE /v1/memory/facts/{id}` - Deletes fact
- ✅ `POST /v1/memory/search` - Semantic search
- ✅ `GET /v1/memory/profile` - Returns user profile
- ✅ `PUT /v1/memory/profile` - Updates profile

### Semantic Search

- ✅ Returns relevant facts based on query
- ✅ Uses embeddings for similarity matching
- ✅ Filters by category and confidence

## Troubleshooting

### "No module named 'pydantic'"

Install dependencies:

```bash
pip install -r requirements.txt
```

### "sentence-transformers not available"

Install embeddings:

```bash
pip install sentence-transformers
```

### "API not running"

Start the API:

```bash
docker-compose up api
# or
python -m uvicorn api.main:app --reload --port 8002
```

### "Import error: memory.models"

Make sure you're running from project root and `src/` is in Python path.

## Next Steps

After testing:

1. ✅ Verify facts are stored correctly
2. ✅ Test semantic search returns relevant results
3. ✅ Integrate with chat endpoint (Phase 1 continuation)
4. ✅ Add fact extraction from conversations
