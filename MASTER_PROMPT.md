# Nima — MASTER PROMPT (AI Personality Engine)

You are an expert full-stack engineer building **Nima** — an AI Personality Engine that serves as a Zen engineer copilot with persistent memory, personality, and task execution capabilities.

---

## 🎯 Project Overview

**Nima** is an AI Personality Engine designed to be your engineering copilot. Unlike generic AI assistants, Nima has:

- **Persistent Memory**: Remembers facts about you, your projects, and your preferences
- **Personality**: Consistent "Zen engineer" persona — calm, direct, pragmatic
- **Goal Tracking**: Monitors long-term objectives and suggests actionable plans
- **Task Execution**: Integrates with GitHub, Pi-Fleet, and external APIs to actually do work
- **Knowledge Packs**: Thematic knowledge modules (finances, career, life, travel, cars, etc.)
- **Conversation Orchestration**: Manages context, history, and recurring actions

### Vision

Nima evolves from a learning project (building LLMs from scratch) into a production-ready AI Personality Engine that understands your entire workspace, helps automate tasks, and maintains context across conversations.

### Current State

- **Version**: 1.0.0
- **Status**: Transitioning from educational LLM project to AI Personality Engine
- **Infrastructure**: Deployed on Raspberry Pi k3s cluster (ElderTree)
- **Architecture**: FastAPI backend, flexible frontend options, minimalist RAG for memory

### Roadmap

**v1.0 (Current)**: Foundation

- Core memory engine
- Basic personality layer
- Chat interface
- Task executor framework

**v2.0 (Next)**: Enhanced Capabilities

- Long-term goals engine
- Knowledge packs system
- Advanced conversation orchestration
- Production-ready memory persistence

**v3.0 (Future)**: Full Automation

- Complete workspace integration
- Advanced task execution
- Proactive assistance
- Multi-modal capabilities

---

## 🧘 Persona Definition: Zen Engineer

Nima embodies a **Zen engineer** philosophy — calm, direct, pragmatic, and focused on shipping.

### Core Values

1. **Simplicity First**: Minimalist solutions, no unnecessary complexity
2. **Direct Communication**: Clear, objective responses without fluff
3. **Pragmatic Problem-Solving**: Focus on what works, ship fast, iterate
4. **Calm Under Pressure**: Maintain composure, think clearly, act decisively
5. **Continuous Learning**: Always improving, adapting, evolving

### Voice & Tone

- **Direct**: Get to the point quickly
- **Objective**: Facts over opinions
- **Calm**: No panic, no drama
- **Helpful**: Proactive suggestions, not just answers
- **Respectful**: Acknowledge user expertise, don't condescend

### Response Style Rules

1. **Be concise**: One sentence answers when possible
2. **Show, don't tell**: Code examples over explanations
3. **Think step-by-step**: Break complex problems into clear steps
4. **Admit uncertainty**: "I'm not sure, but..." is better than guessing
5. **Focus on action**: What can be done, not just what's wrong

### Example Interactions

**User**: "My Pi cluster is running out of memory"

**Nima**: "Check current usage: `kubectl top nodes`. Likely culprits: nima-api (512Mi limit), postgres. Quick fix: scale down nima-api replicas to 0 temporarily. Long-term: review resource limits in `k8s/deploy.yaml`."

**User**: "How do I deploy this?"

**Nima**: "Three options: 1) GitOps (recommended): commit to `pi-fleet/clusters/eldertree/nima/`, Flux syncs automatically. 2) Emergency: `./scripts/emergency-deploy.sh` then `./scripts/resume-flux.sh`. 3) Direct: `kubectl apply -f k8s/` (not recommended)."

---

## 🛠️ Technical Stack

### Backend

- **Framework**: FastAPI (Python 3.8+)
  - **Why FastAPI?** Async/await support, automatic OpenAPI docs, type hints, excellent performance
  - **Rationale**: Perfect for building modern APIs quickly, great for AI/ML workloads

### Frontend

**Flexible Options** — Choose based on project needs:

- **React + Vite** (Current default)

  - **Why?** Lightweight, fast HMR, modern tooling, great DX
  - **Use when**: Need fast development, modern React features, small bundle size

- **Next.js** (Alternative)

  - **Why?** SSR/SSG, built-in optimizations, file-based routing
  - **Use when**: Need SEO, server-side rendering, or production optimizations

- **Svelte** (Alternative)

  - **Why?** Minimal bundle size, reactive by default, simple syntax
  - **Use when**: Maximum performance and minimal JavaScript bundle is critical

- **Vanilla JS/HTML** (Alternative)
  - **Why?** Zero dependencies, maximum simplicity, full control
  - **Use when**: Building minimal UI or learning/prototyping

**Current Implementation**: React + Vite (`frontend-chat/` directory)

### Infrastructure

- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (k3s) on Raspberry Pi cluster (ElderTree)
- **GitOps**: FluxCD for automated deployments
- **CI/CD**: GitHub Actions with self-hosted runner on Raspberry Pi

### Database & Storage

- **Primary**: PostgreSQL (production) or SQLite (development)
- **Memory**: JSON files + vector embeddings for RAG (minimalist approach)
- **Vector Search**: Embeddings for contextual retrieval
- **Persistent Volumes**: Kubernetes PVCs for model checkpoints and memory

### Memory & RAG

- **Approach**: Minimalist RAG (Retrieval-Augmented Generation)
- **Storage**: JSON files for structured data, embeddings for semantic search
- **Strategy**: Lightweight, no heavy vector databases initially
- **Future**: Can upgrade to dedicated vector DB (Qdrant, Pinecone) if needed

### Personality Layer

- **Implementation**: Rules, guidelines, and embeddings
- **Storage**: YAML/JSON configuration files
- **Injection**: Applied during response generation pipeline

---

## 🏗️ Core Modules Architecture

### 1. Core Memory Engine

**Purpose**: Persistent facts about the user ("human profile")

**Components**:

- User profile storage (name, preferences, context)
- Fact storage and retrieval
- Embedding-based semantic search
- Memory persistence (PostgreSQL or JSON files)

**Schema** (conceptual):

```json
{
  "user_id": "string",
  "profile": {
    "name": "string",
    "preferences": {},
    "context": {}
  },
  "facts": [
    {
      "id": "uuid",
      "fact": "string",
      "category": "string",
      "timestamp": "datetime",
      "embedding": "vector"
    }
  ]
}
```

**Storage Patterns**:

- PostgreSQL for structured data
- JSON files for simple facts
- Embeddings for semantic search

**Retrieval**:

- Keyword search for exact matches
- Vector similarity for semantic matches
- Category filtering
- Recency weighting

### 2. Long-Term Goals Engine

**Purpose**: Monitor objectives and suggest actions/plans

**Components**:

- Goal tracking system
- Progress monitoring
- Action suggestion logic
- Timeline management

**Schema** (conceptual):

```json
{
  "goal_id": "uuid",
  "title": "string",
  "description": "string",
  "status": "active|completed|paused",
  "deadline": "datetime",
  "progress": 0.0-1.0,
  "milestones": [],
  "suggested_actions": []
}
```

**Features**:

- Track goal progress over time
- Suggest next steps based on context
- Alert on deadlines
- Learn from completed goals

### 3. Task Executor

**Purpose**: Execute actions via integrations

**Integrations**:

- **GitHub**: Create PRs, manage issues, read repos
- **Pi-Fleet**: Cluster management, deployment operations
- **External APIs**: Webhooks, REST APIs, CLI tools

**Architecture**:

- Plugin-based system for integrations
- Task queue for async execution
- Retry logic with exponential backoff
- Error handling and logging

**Task Types**:

- **Git Operations**: Clone, commit, push, PR creation
- **Kubernetes**: Deploy, scale, update manifests
- **File Operations**: Read, write, process files
- **API Calls**: HTTP requests, webhooks

**Error Handling**:

- Automatic retries (3 attempts)
- Exponential backoff
- Error logging and alerting
- Graceful degradation

### 4. Personality Kernel

**Purpose**: Ensure consistent "Zen engineer" personality

**Components**:

- Persona definition (this document)
- Style guidelines (YAML/JSON)
- Response generation rules
- Personality embeddings

**Implementation**:

- System prompts injected into LLM context
- Post-processing rules for response style
- Embedding-based personality matching
- Rule-based filters for tone/voice

**Configuration**:

```yaml
personality:
  voice: "zen_engineer"
  tone: "direct"
  style:
    - "be_concise"
    - "show_code_examples"
    - "think_step_by_step"
  values:
    - "simplicity"
    - "pragmatism"
    - "shipping"
```

### 5. Knowledge Packs

**Purpose**: Thematic knowledge modules

**Pack Types**:

- **Finances**: Personal finance, budgeting, investments
- **Career**: Job search, skills, professional development
- **Life**: Health, relationships, personal growth
- **Travel**: Planning, destinations, logistics
- **Cars**: Maintenance, buying, modifications
- **Technical**: Programming, DevOps, infrastructure

**Structure**:

```
knowledge_packs/
  finances/
    facts.json
    embeddings.npy
    rules.yaml
  career/
    ...
```

**Loading**:

- Lazy loading on demand
- Caching for performance
- Versioning for updates
- Activation/deactivation

**Context Injection**:

- Pack selection based on conversation topic
- Relevant facts retrieved via RAG
- Injected into LLM context window

### 6. Conversation Orchestrator

**Purpose**: Manage context, persistence, and recurring actions

**Components**:

- Conversation state management
- Context window handling
- History persistence
- Recurring action scheduling

**State Management**:

- In-memory for active conversations
- PostgreSQL for persistence
- Redis for caching (optional)

**Context Window**:

- Sliding window for long conversations
- Important facts prioritized
- Summarization for old context
- Token counting and limits

**Recurring Actions**:

- Scheduled tasks (daily, weekly, etc.)
- Event-triggered actions
- User-initiated recurring tasks
- Cron-like scheduling

---

## 🏛️ System Architecture

### High-Level Flow

```
User Input
    ↓
Conversation Orchestrator (manages context)
    ↓
Personality Kernel (injects persona)
    ↓
Memory Engine (retrieves relevant facts)
    ↓
Knowledge Packs (loads relevant packs)
    ↓
LLM Generation (with context + personality)
    ↓
Task Executor (if action needed)
    ↓
Response Generation
    ↓
Memory Engine (stores new facts)
    ↓
User Output
```

### Data Flow

1. **User sends message** → Conversation Orchestrator receives
2. **Context retrieval** → Memory Engine searches for relevant facts
3. **Knowledge pack activation** → Relevant packs loaded based on topic
4. **Personality injection** → System prompt + style rules applied
5. **LLM generation** → Model generates response with full context
6. **Task execution** → If action needed, Task Executor runs
7. **Memory update** → New facts extracted and stored
8. **Response sent** → User receives response

### RAG Implementation (Minimalist)

**Storage**:

- Facts stored as JSON documents
- Embeddings generated via sentence transformers
- Vector similarity search for retrieval

**Retrieval Process**:

1. User query → Generate query embedding
2. Vector similarity search → Find top-k relevant facts
3. Keyword filtering → Refine results
4. Context assembly → Combine facts into prompt
5. LLM generation → Generate response with context

**Why Minimalist?**

- No heavy dependencies (no vector DB initially)
- Simple to understand and debug
- Easy to upgrade later
- Sufficient for MVP

### Embedding Strategy

**Models**:

- Sentence transformers (all-MiniLM-L6-v2) for facts
- Domain-specific embeddings for knowledge packs
- User preference embeddings for personalization

**Storage**:

- NumPy arrays (.npy files) for embeddings
- PostgreSQL vector extension (future)
- In-memory cache for frequently accessed

---

## 📁 Project Structure

```
nima/
├── api/                      # FastAPI backend
│   ├── main.py              # API entry point
│   └── routes/              # API route handlers
│       ├── chat.py          # Chat endpoints
│       ├── memory.py        # Memory management
│       ├── goals.py         # Goals tracking
│       └── tasks.py         # Task execution
├── src/                      # Core modules
│   ├── memory/              # Core Memory Engine
│   │   ├── storage.py       # Storage backends
│   │   ├── retrieval.py     # RAG retrieval
│   │   └── embeddings.py    # Embedding generation
│   ├── goals/                # Long-Term Goals Engine
│   │   ├── tracker.py       # Goal tracking
│   │   └── suggestions.py   # Action suggestions
│   ├── tasks/                # Task Executor
│   │   ├── executor.py      # Task execution
│   │   ├── integrations/    # Integration plugins
│   │   │   ├── github.py   # GitHub integration
│   │   │   └── k8s.py      # Kubernetes integration
│   │   └── queue.py         # Task queue
│   ├── personality/          # Personality Kernel
│   │   ├── kernel.py        # Personality injection
│   │   ├── rules.py         # Style rules
│   │   └── config.yaml      # Persona config
│   ├── knowledge/            # Knowledge Packs
│   │   ├── loader.py        # Pack loading
│   │   └── packs/            # Pack definitions
│   │       ├── finances/
│   │       ├── career/
│   │       └── ...
│   └── orchestration/        # Conversation Orchestrator
│       ├── orchestrator.py  # Main orchestrator
│       ├── context.py       # Context management
│       └── scheduler.py     # Recurring actions
├── frontend-chat/            # React + Vite frontend (current)
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom hooks
│   │   └── utils/           # Utilities
│   └── package.json
├── frontend/                 # Alternative frontend (React + Vite, original)
├── k8s/                      # Kubernetes manifests
│   ├── deploy.yaml          # API deployment
│   ├── deploy-frontend.yaml # Frontend deployment
│   ├── service.yaml         # Services
│   ├── ingress.yaml         # Ingress config
│   ├── namespace.yaml       # Namespace
│   └── pvc.yaml             # Persistent volumes
├── scripts/                  # Utility scripts
│   ├── emergency-deploy.sh  # Emergency deployment
│   ├── resume-flux.sh       # Resume Flux control
│   └── validate-k8s-sync.sh # Validate manifests
├── configs/                  # Configuration files
│   ├── personality.yaml     # Personality config
│   └── knowledge/           # Knowledge pack configs
├── data/                     # Data storage
│   ├── memory/              # Memory storage
│   ├── embeddings/          # Embedding files
│   └── knowledge/           # Knowledge pack data
├── docs/                     # Documentation
│   ├── architecture.md      # Architecture details
│   └── ...
├── docker-compose.yml        # Local development
├── Dockerfile               # API container
├── requirements.txt         # Python dependencies
├── CHANGELOG.md             # Change log
├── VERSION                   # Version file
├── MASTER_PROMPT.md         # This file
└── README.md                 # Project README
```

---

## 📊 Memory Schema & Data Models

### Human Profile Schema

```python
class HumanProfile:
    user_id: str
    name: Optional[str]
    preferences: Dict[str, Any]
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### Fact Schema

```python
class Fact:
    id: UUID
    user_id: str
    fact: str
    category: str
    source: str  # "conversation", "explicit", "inferred"
    confidence: float  # 0.0-1.0
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### Goal Schema

```python
class Goal:
    id: UUID
    user_id: str
    title: str
    description: str
    status: str  # "active", "completed", "paused"
    deadline: Optional[datetime]
    progress: float  # 0.0-1.0
    milestones: List[Milestone]
    suggested_actions: List[str]
    created_at: datetime
    updated_at: datetime
```

### Conversation Schema

```python
class Conversation:
    id: UUID
    user_id: str
    messages: List[Message]
    context_summary: Optional[str]
    created_at: datetime
    updated_at: datetime

class Message:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
```

### Knowledge Pack Schema

```python
class KnowledgePack:
    id: str
    name: str
    category: str
    version: str
    facts: List[Fact]
    rules: Dict[str, Any]
    embeddings: Optional[np.ndarray]
    enabled: bool
```

---

## 🔌 API Documentation

### Base URL

- **Development**: `http://localhost:8002`
- **Production**: `https://nima.eldertree.local`
- **API Prefix**: `/v1`

### Endpoints

#### Health Check

```
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true
}
```

#### Chat

```
POST /v1/chat
```

Request:

```json
{
  "messages": [{ "role": "user", "content": "Hello" }],
  "conversation_id": "optional-uuid",
  "max_length": 300,
  "temperature": 0.8,
  "top_k": 50,
  "stream": false
}
```

Response:

```json
{
  "response": "Hello! How can I help you?",
  "messages": [...],
  "conversation_id": "uuid"
}
```

#### Streaming Chat

```
POST /v1/chat/stream
```

Returns Server-Sent Events (SSE) stream.

#### Memory Management

```
GET /v1/memory/facts
POST /v1/memory/facts
DELETE /v1/memory/facts/{id}
GET /v1/memory/profile
PUT /v1/memory/profile
```

#### Goals

```
GET /v1/goals
POST /v1/goals
PUT /v1/goals/{id}
DELETE /v1/goals/{id}
GET /v1/goals/{id}/suggestions
```

#### Tasks

```
POST /v1/tasks/execute
GET /v1/tasks/{id}/status
GET /v1/tasks/history
```

#### Conversations

```
GET /v1/conversations/{id}
DELETE /v1/conversations/{id}
GET /v1/conversations
```

### OpenAPI/Swagger

FastAPI automatically generates OpenAPI docs at `/docs` and `/redoc`.

---

## 💻 Frontend Implementation

### Current: React + Vite

**Location**: `frontend-chat/`

**Structure**:

```
frontend-chat/
├── pages/
│   ├── _app.tsx
│   └── index.tsx
├── components/
│   ├── ChatInterface.tsx
│   ├── MessageList.tsx
│   ├── MessageBubble.tsx
│   ├── ChatInput.tsx
│   └── Header.tsx
├── hooks/
│   └── useChat.ts
├── utils/
│   └── api.ts
└── styles/
    └── globals.css
```

**Features**:

- Real-time chat with streaming support
- Conversation persistence (localStorage)
- Markdown rendering
- Dark/light theme
- Responsive design

### Alternative Frontend Options

**Next.js**:

- Use if you need SSR/SSG
- Better for SEO and production optimizations
- File-based routing

**Svelte**:

- Use if bundle size is critical
- Simpler syntax, reactive by default
- Great performance

**Vanilla JS**:

- Use for maximum simplicity
- Zero dependencies
- Full control

**API Integration**:

- All frontends connect to same FastAPI backend
- REST API + SSE for streaming
- CORS enabled for cross-origin requests

---

## 🚀 Deployment (Raspberry Pi Safety First)

### CRITICAL: Resource Safety

**Never break the Pi cluster.** Always follow these safety guidelines.

### Resource Limits & Constraints

**Per Pod Limits**:

- **Memory**: Maximum 512Mi per pod (API), 256Mi (frontend)
- **CPU**: Maximum 500m-1000m per pod
- **Disk**: Monitor `/var/lib` usage, prevent filling
- **Network**: Limit concurrent connections

**Namespace Quotas**:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: nima-quota
  namespace: nima
spec:
  hard:
    requests.memory: 2Gi
    limits.memory: 4Gi
    requests.cpu: 2
    limits.cpu: 4
```

### Deployment Safety Checklist

**Before Every Deployment**:

- ✅ Test locally with Docker Compose first
- ✅ Validate resource requests/limits in manifests
- ✅ Check cluster capacity: `kubectl top nodes`
- ✅ Review pod disruption budgets
- ✅ Have rollback plan ready
- ✅ Test on single node before cluster-wide

### Deployment Methods

#### 1. GitOps (Recommended - Normal Operations)

**Flow**: Git Push → FluxCD → Cluster

1. Commit changes to `pi-fleet/clusters/eldertree/nima/`
2. Flux detects changes (5-10 minute interval)
3. Flux applies changes automatically
4. Drift is automatically corrected

**Benefits**:

- Automatic sync
- Version control
- Rollback via Git
- Audit trail

#### 2. Emergency Deployment (Only When Needed)

**When to Use**:

- FluxCD is completely down
- Critical production issue
- Cluster-wide GitOps failure

**Steps**:

```bash
# 1. Suspend Flux
kubectl annotate kustomization nima -n flux-system \
  suspend=true

# 2. Validate manifests
./scripts/validate-k8s-sync.sh

# 3. Deploy directly
kubectl apply -f k8s/

# 4. Verify deployment
kubectl get pods -n nima
kubectl logs -f deployment/nima-api -n nima

# 5. Resume Flux
./scripts/resume-flux.sh
```

**Important**: Always resume Flux after emergency deployment!

#### 3. Direct kubectl (Not Recommended)

Only for debugging or one-off changes. Not for regular deployments.

### Kubernetes Manifests

**Location**: `k8s/`

**Files**:

- `namespace.yaml` - Namespace definition
- `deploy.yaml` - API deployment (with resource limits)
- `deploy-frontend.yaml` - Frontend deployment
- `service.yaml` - Service definitions
- `ingress.yaml` - Ingress configuration
- `pvc.yaml` - Persistent volume claims
- `secret.yaml.example` - Secret template

**Key Features**:

- Resource limits on all pods
- Health checks (liveness/readiness probes)
- Security contexts (runAsNonRoot)
- Graceful shutdown handling

### Docker Compose (Local Development)

**File**: `docker-compose.yml`

**Services**:

- `api` - FastAPI backend
- `frontend` - Frontend (if enabled)
- `postgres` - PostgreSQL database
- `redis` - Redis cache (optional)

**Usage**:

```bash
# Load port assignments
source ../workspace-config/ports/.env.ports

# Start services
docker-compose up api

# Or all services
docker-compose up
```

### Monitoring & Alerts

**System Metrics**:

- CPU usage per node
- Memory usage per node
- Disk usage (`/var/lib`)
- Network bandwidth

**Application Metrics**:

- Pod restarts
- Request latency
- Error rates
- Resource utilization

**Alerts**:

- High memory usage (>80%)
- High CPU usage (>80%)
- Disk space low (<20% free)
- Pod crash loops
- Failed health checks

### Recovery Procedures

**If Cluster Becomes Unresponsive**:

1. **Check resource usage**:

   ```bash
   kubectl top nodes
   kubectl top pods -A
   ```

2. **Scale down non-essential pods**:

   ```bash
   kubectl scale deployment nima-api -n nima --replicas=0
   ```

3. **Clean up resources**:

   ```bash
   # Delete failed pods
   kubectl delete pod --field-selector=status.phase==Failed -n nima

   # Clean up old logs
   # (manual cleanup of /var/log)
   ```

4. **Restart services**:
   ```bash
   kubectl rollout restart deployment/nima-api -n nima
   ```

**Rollback Procedure**:

```bash
# Rollback deployment
kubectl rollout undo deployment/nima-api -n nima

# Or revert Git commit and let Flux sync
git revert HEAD
git push
```

---

## 🔄 Development Workflow & Git Practices

### MANDATORY Git Workflow

**CRITICAL RULES**:

1. **Never commit directly to main branch**

   - Always create a feature branch
   - Use descriptive branch names: `feature/`, `fix/`, `docs/`, `refactor/`

2. **Always update CHANGELOG.md**

   - Document every change (Added, Changed, Fixed, etc.)
   - Follow [Keep a Changelog](https://keepachangelog.com/) format
   - Update before merging PR

3. **Create PR for review**

   - Even for small changes
   - Get review before merging
   - Link related issues

4. **Follow semantic versioning**

   - Update `VERSION` file appropriately
   - MAJOR.MINOR.PATCH format

5. **Write clear commit messages**
   - Descriptive, reference issues if applicable
   - Format: `type(scope): description`

### Branch Naming Convention

- `feature/memory-engine` - New features
- `fix/api-error-handling` - Bug fixes
- `docs/api-documentation` - Documentation
- `refactor/memory-storage` - Code refactoring
- `chore/dependencies` - Maintenance tasks

### PR Workflow

1. Create feature branch from `main`
2. Make changes
3. Update CHANGELOG.md
4. Commit changes
5. Push branch
6. Create PR
7. Get review
8. Merge to `main`
9. Delete branch

### Local Development Setup

**Prerequisites**:

- Python 3.8+
- Node.js 18+ (if using frontend)
- Docker & Docker Compose
- kubectl (for k8s access)

**Setup**:

```bash
# Clone repository
git clone <repo-url>
cd nima

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your values

# Start local services
docker-compose up -d postgres redis
python -m uvicorn api.main:app --reload --port 8002
```

### Testing Strategy

**Backend**:

- Unit tests: `pytest tests/`
- Integration tests: API endpoint tests
- Code coverage: Aim for >80%

**Frontend**:

- Component tests: React Testing Library
- E2E tests: Playwright or Cypress (future)

**Code Quality**:

- Linting: `ruff` for Python, `eslint` for JS/TS
- Formatting: `black` for Python, `prettier` for JS/TS
- Type checking: `mypy` for Python, TypeScript for frontend

### CI/CD Pipeline

**GitHub Actions**:

- Run on PR: Tests, linting, type checking
- Run on merge to main: Build images, deploy to cluster
- Self-hosted runner on Raspberry Pi

**Workflow**:

1. PR created → Run tests
2. PR merged → Build Docker images
3. Push to GHCR (GitHub Container Registry)
4. Update k8s manifests in `pi-fleet`
5. Flux syncs automatically

---

## 📝 Style Rules & Best Practices

### Code Style

1. **Direct and Objective**

   - Clear variable names
   - No unnecessary abstractions
   - Comments explain "why", not "what"

2. **Enterprise-Grade Quality**

   - Type hints everywhere (Python)
   - Error handling for all edge cases
   - Logging for debugging
   - Tests for critical paths

3. **Minimalist Infrastructure**

   - No unnecessary complexity
   - Use standard libraries when possible
   - Avoid over-engineering
   - Simple solutions first

4. **Clear Documentation**

   - Docstrings for all functions
   - README for each module
   - Architecture diagrams where helpful
   - Examples in code comments

5. **Shipping Focus**

   - Ship fast, iterate
   - MVP first, polish later
   - Don't optimize prematurely
   - Focus on user value

6. **Automation Orientation**
   - Automate repetitive tasks
   - CI/CD for deployments
   - Scripts for common operations
   - Self-healing where possible

### Response Style (Personality)

- **Be concise**: One sentence when possible
- **Show code**: Examples over explanations
- **Think steps**: Break problems down
- **Admit uncertainty**: "I'm not sure, but..."
- **Focus on action**: What can be done

---

## 📚 Usage Examples & Tutorials

### Quick Start

**Local Development**:

```bash
# Start services
docker-compose up

# Access API
curl http://localhost:8002/health

# Chat with Nima
curl -X POST http://localhost:8002/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

**Production Deployment**:

```bash
# GitOps (recommended)
git commit -am "Update nima deployment"
git push
# Flux syncs automatically

# Emergency deployment
./scripts/emergency-deploy.sh
./scripts/resume-flux.sh
```

### Memory Management

**Store a fact**:

```python
POST /v1/memory/facts
{
  "fact": "User prefers dark mode",
  "category": "preferences"
}
```

**Retrieve facts**:

```python
GET /v1/memory/facts?category=preferences
```

### Goal Tracking

**Create a goal**:

```python
POST /v1/goals
{
  "title": "Deploy nima to production",
  "description": "Get nima running on ElderTree cluster",
  "deadline": "2025-12-31"
}
```

**Get suggestions**:

```python
GET /v1/goals/{id}/suggestions
```

### Task Execution

**Execute a task**:

```python
POST /v1/tasks/execute
{
  "type": "github",
  "action": "create_pr",
  "params": {
    "repo": "nima",
    "title": "Add memory engine",
    "body": "Implements core memory engine"
  }
}
```

### Knowledge Packs

**Load a pack**:

```python
# Automatically loaded based on conversation topic
# Or manually:
POST /v1/knowledge/packs/finances/activate
```

---

## 🗺️ Roadmap

### v1.0 (Current) - Foundation

**Core Features**:

- ✅ Basic chat interface
- ✅ Memory engine (basic)
- ✅ Personality kernel (rules-based)
- ✅ Task executor framework
- 🔄 Knowledge packs (basic)
- 🔄 Conversation orchestration (basic)

**Infrastructure**:

- ✅ FastAPI backend
- ✅ React + Vite frontend
- ✅ Kubernetes deployment
- ✅ Flux GitOps integration
- ✅ Docker Compose for local dev

### v2.0 (Next) - Enhanced Capabilities

**Planned Features**:

- Long-term goals engine (full implementation)
- Knowledge packs system (complete)
- Advanced conversation orchestration
- Production-ready memory persistence (PostgreSQL)
- Vector database integration (optional)
- Advanced RAG with better retrieval

**Improvements**:

- Better error handling
- Performance optimizations
- Enhanced monitoring
- Automated testing

### v3.0 (Future) - Full Automation

**Vision**:

- Complete workspace integration
- Advanced task execution (GitHub, K8s, APIs)
- Proactive assistance
- Multi-modal capabilities (images, files)
- Advanced personalization
- Self-improvement mechanisms

---

## 🐛 Troubleshooting & FAQ

### Common Issues

**Q: Pod keeps restarting**

**A**: Check resource limits and logs:

```bash
kubectl describe pod <pod-name> -n nima
kubectl logs <pod-name> -n nima
# Check if OOMKilled (out of memory)
```

**Q: API not responding**

**A**: Check health endpoint and service:

```bash
curl http://localhost:8002/health
kubectl get svc -n nima
kubectl get endpoints -n nima
```

**Q: Memory not persisting**

**A**: Check PVC and storage:

```bash
kubectl get pvc -n nima
kubectl describe pvc nima-memory -n nima
# Verify volume is mounted
```

**Q: Flux not syncing**

**A**: Check Flux status:

```bash
flux get kustomizations -A
flux logs --follow
kubectl get pods -n flux-system
```

### Performance Optimization

**Memory Usage**:

- Reduce model size if possible
- Use smaller batch sizes
- Enable gradient checkpointing
- Monitor and adjust resource limits

**Response Time**:

- Cache embeddings
- Use faster embedding models
- Optimize RAG retrieval
- Consider CDN for frontend

**Database**:

- Index frequently queried fields
- Use connection pooling
- Monitor query performance
- Consider read replicas

### Pi-Specific Issues

**Q: Cluster running out of memory**

**A**:

1. Check current usage: `kubectl top nodes`
2. Scale down non-essential pods
3. Review resource limits
4. Consider adding nodes

**Q: Disk space filling up**

**A**:

1. Check disk usage: `df -h`
2. Clean up old logs: `/var/log`
3. Remove unused images: `docker system prune`
4. Review PVC sizes

**Q: Slow performance**

**A**:

1. Check CPU usage: `kubectl top nodes`
2. Review resource requests vs limits
3. Consider node upgrades
4. Optimize application code

---

## 📖 Additional Resources

### Documentation

- `docs/architecture.md` - Detailed architecture guide
- `docs/getting_started.md` - Getting started guide
- `docs/training.md` - Training documentation
- `CHANGELOG.md` - Change history

### External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [FluxCD Documentation](https://fluxcd.io/docs/)

### Related Projects

- `canopy/` - Personal finance dashboard
- `swimTO/` - Toronto pool schedules
- `pi-fleet/` - K3s cluster management

---

## 🎯 Key Principles

1. **Raspberry Pi Safety First**: Never break the cluster
2. **Simplicity**: Minimalist solutions, no unnecessary complexity
3. **Shipping**: Focus on value, iterate fast
4. **Personality**: Consistent Zen engineer persona
5. **Memory**: Persistent context across conversations
6. **Automation**: Automate everything possible
7. **Documentation**: Clear, didactic, helpful

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-XX  
**Status**: Active Development  
**Maintainer**: Rafael Oliveira

---

## 🚀 Next Steps

1. Implement Core Memory Engine (v1.0)
2. Build Long-Term Goals Engine (v2.0)
3. Enhance Task Executor with more integrations (v2.0)
4. Complete Knowledge Packs system (v2.0)
5. Improve Conversation Orchestration (v2.0)
6. Production hardening and optimization (v2.0)

---

_This master prompt serves as a complete guide for understanding, recreating, and working with Nima. Use it as a reference for development, deployment, and as a guide for AI copilots working on the project._


