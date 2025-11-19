# Local Testing Guide

This guide shows how to test Nima locally before pushing to GitHub.

## Quick Test (Recommended)

Run the basic validation script:

```bash
./scripts/test-local.sh
```

This checks:
- ✅ Python syntax
- ✅ File structure (no merge conflicts)
- ✅ Version consistency
- ✅ Dockerfile existence
- ✅ Workflow file structure

## Test Docker Builds

Test that Docker images build correctly (simulates GitHub Actions):

```bash
./scripts/test-docker-builds.sh
```

This will:
- Build API image: `ghcr.io/raolivei/nima-api:test` and `:0.5.0`
- Build Frontend image: `ghcr.io/raolivei/nima-frontend:test` and `:0.5.0`
- Show build logs if any errors occur

**Note**: This requires Docker to be running and may take several minutes.

## Test API Locally

### Option 1: Docker Compose (Recommended)

```bash
# Load port assignments
source ../workspace-config/ports/.env.ports

# Start API only
docker-compose up api

# Or start all services
docker-compose up
```

Then test:
```bash
# Health check
curl http://localhost:8002/health

# API docs
open http://localhost:8002/docs

# Test chat endpoint
curl -X POST http://localhost:8002/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### Option 2: Local Python (if dependencies installed)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8002
```

**Note**: This requires the model checkpoint to be available at the path specified in environment variables.

## Test Frontend Locally

```bash
cd frontend-chat
npm install
npm run dev
```

Then open http://localhost:3000 (or the port Next.js assigns).

## Test GitHub Actions Workflow Locally

You can use `act` to run GitHub Actions workflows locally:

```bash
# Install act (macOS)
brew install act

# Run the workflow
act push -W .github/workflows/build-and-push.yml
```

**Note**: `act` requires Docker and may need configuration for secrets.

## Manual Docker Build Test

Test individual builds:

```bash
# Build API
docker build -t nima-api:test -f Dockerfile .

# Build Frontend
docker build -t nima-frontend:test -f frontend-chat/Dockerfile ./frontend-chat

# Test API container
docker run -p 8002:8000 nima-api:test

# Test Frontend container
docker run -p 3002:3000 nima-frontend:test
```

## What Gets Tested

### Before Push
- ✅ No merge conflicts
- ✅ Version consistency
- ✅ Python syntax
- ✅ File structure

### Docker Builds
- ✅ API Dockerfile builds successfully
- ✅ Frontend Dockerfile builds successfully
- ✅ Images are tagged correctly
- ✅ Multi-platform builds (if using buildx)

### Runtime
- ✅ API starts without errors
- ✅ Health endpoint responds
- ✅ Chat endpoints work
- ✅ CORS headers are set correctly

## Troubleshooting

### "Docker not running"
```bash
# Start Docker Desktop or Docker daemon
open -a Docker  # macOS
```

### "Port already in use"
```bash
# Check what's using the port
lsof -i :8002

# Kill the process or use different port
export NIMA_API_PORT=8003
```

### "Model checkpoint not found"
The API needs a trained model checkpoint. Either:
1. Use Docker Compose (handles volumes)
2. Set `NIMA_CHECKPOINT_PATH` environment variable
3. Train a model first: `python scripts/train.py --config configs/base_model.yaml`

### "Build fails"
Check build logs:
```bash
docker build -t nima-api:test -f Dockerfile . 2>&1 | tee build.log
```

## Next Steps

After local testing passes:
1. ✅ Commit changes
2. ✅ Push to branch
3. ✅ Create PR
4. ✅ GitHub Actions will run automatically
5. ✅ Review workflow results
6. ✅ Merge PR
7. ✅ Tag release: `git tag v0.5.0 && git push origin v0.5.0`

