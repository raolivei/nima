# Nima AI Personality Engine - Implementation Plan

**Version**: 0.5.0  
**Status**: Planning Phase  
**Last Updated**: 2025-01-XX

## Overview

This document outlines the phased implementation plan for transforming Nima from an educational LLM project into a production-ready AI Personality Engine. Each phase will be implemented on a separate feature branch following Git workflow best practices.

## Implementation Strategy

- **Each phase = One feature branch**
- **Never commit directly to main**
- **Update CHANGELOG.md for each phase**
- **Create PR for review before merging**
- **Tag releases appropriately**

---

## Phase 0: Foundation & Cleanup

**Branch**: `fix/merge-conflicts-and-version`  
**Goal**: Resolve merge conflicts, update version to 0.5.0, establish clean baseline

### Tasks

1. **Resolve merge conflicts in `api/main.py`**
   - Keep HEAD changes (chat functionality, CORS, streaming)
   - Remove conflict markers
   - Ensure all endpoints work correctly
   - Test chat endpoints locally

2. **Update version to 0.5.0**
   - Update `VERSION` file: `0.5.0`
   - Update `api/main.py` version string: `"0.5.0"`
   - Update `CHANGELOG.md` with version 0.5.0 entry

3. **Create git tag**
   - Tag current state as `v0.5.0`
   - Push tag to remote

4. **Verify baseline**
   - Ensure API starts without errors
   - Test `/health` endpoint
   - Test `/v1/chat` endpoint
   - Test `/v1/chat/stream` endpoint

### Deliverables

- Clean `api/main.py` without conflicts
- Version 0.5.0 tagged
- Updated CHANGELOG.md
- All existing endpoints functional

---

## Phase 1: Core Memory Engine

**Branch**: `feature/core-memory-engine`  
**Goal**: Implement persistent memory system for storing and retrieving user facts

### Architecture

```
src/memory/
├── __init__.py
├── storage.py          # Storage backends (JSON, PostgreSQL)
├── retrieval.py        # RAG retrieval logic
├── embeddings.py       # Embedding generation
└── models.py           # Pydantic models (Fact, HumanProfile)
```

### Implementation Steps

1. **Create module structure**
   - Create `src/memory/` directory
   - Add `__init__.py` files
   - Create base models (Fact, HumanProfile)

2. **Implement storage backends**
   - JSON file storage (development)
   - PostgreSQL storage (production)
   - Storage interface/abstraction
   - Migration from JSON to PostgreSQL

3. **Implement embedding generation**
   - Add `sentence-transformers` dependency
   - Use `all-MiniLM-L6-v2` model
   - Generate embeddings for facts
   - Store embeddings (NumPy arrays)

4. **Implement RAG retrieval**
   - Vector similarity search
   - Keyword search fallback
   - Category filtering
   - Recency weighting
   - Top-k retrieval

5. **Create API routes**
   - Create `api/routes/memory.py`
   - `GET /v1/memory/facts` - List facts with filters
   - `POST /v1/memory/facts` - Store new fact
   - `DELETE /v1/memory/facts/{id}` - Delete fact
   - `GET /v1/memory/profile` - Get user profile
   - `PUT /v1/memory/profile` - Update user profile
   - `GET /v1/memory/search` - Semantic search facts

6. **Integrate with chat**
   - Retrieve relevant facts before generation
   - Inject facts into system prompt
   - Extract facts from conversations
   - Store inferred facts

### Dependencies to Add

```txt
sentence-transformers>=2.2.0
psycopg[binary]>=3.1.0
sqlalchemy>=2.0.0
alembic>=1.12.0  # For migrations
```

### Data Storage

- **Development**: `data/memory/facts.json`, `data/memory/profile.json`
- **Production**: PostgreSQL tables (`facts`, `human_profiles`)
- **Embeddings**: `data/embeddings/` (NumPy arrays)

### API Schema

```python
class FactCreate(BaseModel):
    fact: str
    category: str
    source: str = "conversation"  # "conversation", "explicit", "inferred"
    confidence: float = 0.8
    metadata: Dict[str, Any] = {}

class FactResponse(BaseModel):
    id: str
    fact: str
    category: str
    source: str
    confidence: float
    created_at: datetime
    updated_at: datetime
```

### Testing

- Unit tests for storage backends
- Unit tests for retrieval logic
- Integration tests for API endpoints
- Test fact extraction from conversations

### Deliverables

- Complete memory engine module
- API routes for memory management
- Integration with chat endpoint
- Documentation for memory usage

---

## Phase 2: Personality Kernel

**Branch**: `feature/personality-kernel`  
**Goal**: Implement consistent "Zen engineer" personality injection

### Architecture

```
src/personality/
├── __init__.py
├── kernel.py           # Main personality injection
├── rules.py            # Style rules and guidelines
├── config.yaml         # Persona configuration
└── formatter.py        # Response formatting
```

### Implementation Steps

1. **Create module structure**
   - Create `src/personality/` directory
   - Create `configs/personality.yaml` configuration

2. **Implement persona configuration**
   - Define Zen engineer values
   - Define voice and tone rules
   - Define response style guidelines
   - Create YAML config file

3. **Implement personality kernel**
   - Load persona config
   - Generate system prompt from config
   - Inject personality into chat context
   - Apply style rules post-generation

4. **Implement response formatting**
   - Apply conciseness rules
   - Format code examples
   - Ensure direct communication
   - Maintain calm tone

5. **Integrate with chat endpoint**
   - Inject personality system prompt
   - Apply post-processing rules
   - Ensure consistent persona across responses

### Configuration File

```yaml
# configs/personality.yaml
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
  system_prompt: |
    You are Nima, a Zen engineer AI assistant.
    Core values: simplicity, directness, pragmatism.
    Communication style: concise, objective, calm.
```

### Integration Points

- Chat endpoint system prompt generation
- Response post-processing
- Memory fact extraction (personality-aware)

### Testing

- Test personality injection
- Test response formatting
- Verify consistency across conversations
- Test with different conversation contexts

### Deliverables

- Personality kernel module
- Configuration system
- Integration with chat
- Consistent persona in responses

---

## Phase 3: Conversation Orchestrator

**Branch**: `feature/conversation-orchestrator`  
**Goal**: Manage conversation context, history, and state

### Architecture

```
src/orchestration/
├── __init__.py
├── orchestrator.py     # Main orchestrator
├── context.py          # Context management
├── scheduler.py        # Recurring actions
└── models.py           # Conversation models
```

### Implementation Steps

1. **Create module structure**
   - Create `src/orchestration/` directory
   - Define conversation models

2. **Implement context management**
   - Sliding window for long conversations
   - Token counting and limits
   - Context summarization for old messages
   - Important fact prioritization

3. **Implement conversation persistence**
   - Store conversations in PostgreSQL
   - Load conversation history
   - Update conversation state
   - Conversation metadata tracking

4. **Implement orchestrator**
   - Coordinate memory retrieval
   - Coordinate personality injection
   - Coordinate knowledge pack loading
   - Manage full conversation flow

5. **Update chat endpoint**
   - Use orchestrator instead of direct generation
   - Integrate with memory engine
   - Integrate with personality kernel
   - Store conversation history

6. **Implement recurring actions** (basic)
   - Scheduled task framework
   - Event-triggered actions
   - User-initiated recurring tasks

### Database Schema

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    context_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    role TEXT NOT NULL,  -- "user", "assistant", "system"
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now(),
    metadata JSONB
);
```

### Integration Points

- Memory engine (retrieve facts)
- Personality kernel (inject persona)
- Knowledge packs (load relevant packs)
- Chat endpoint (orchestrate flow)

### Testing

- Test context window management
- Test conversation persistence
- Test orchestrator flow
- Test token limits and summarization

### Deliverables

- Conversation orchestrator module
- Context management system
- Conversation persistence
- Updated chat endpoint integration

---

## Phase 4: Knowledge Packs

**Branch**: `feature/knowledge-packs`  
**Goal**: Implement thematic knowledge modules

### Architecture

```
src/knowledge/
├── __init__.py
├── loader.py           # Pack loading logic
├── packs/              # Pack definitions
│   ├── finances/
│   │   ├── facts.json
│   │   ├── rules.yaml
│   │   └── embeddings.npy
│   ├── career/
│   ├── life/
│   └── technical/
└── activator.py        # Pack activation logic
```

### Implementation Steps

1. **Create module structure**
   - Create `src/knowledge/` directory
   - Create `data/knowledge/` storage directory

2. **Implement pack structure**
   - Define pack schema
   - Create example packs (finances, career, technical)
   - Pack metadata (name, category, version)

3. **Implement pack loader**
   - Load pack from filesystem
   - Cache loaded packs
   - Version management
   - Activation/deactivation

4. **Implement pack activator**
   - Topic detection (simple keyword matching initially)
   - Automatic pack activation
   - Manual pack activation
   - Pack context injection

5. **Create API routes**
   - `GET /v1/knowledge/packs` - List available packs
   - `POST /v1/knowledge/packs/{id}/activate` - Activate pack
   - `DELETE /v1/knowledge/packs/{id}/activate` - Deactivate pack
   - `GET /v1/knowledge/packs/{id}` - Get pack details

6. **Integrate with orchestrator**
   - Auto-activate packs based on conversation topic
   - Inject pack context into system prompt
   - Use pack facts in RAG retrieval

### Pack Structure

```json
{
  "id": "finances",
  "name": "Personal Finance",
  "category": "life",
  "version": "1.0.0",
  "enabled": true,
  "facts": [
    {
      "fact": "Emergency fund should be 3-6 months expenses",
      "category": "budgeting"
    }
  ],
  "rules": {
    "topics": ["finance", "money", "budget", "investment"]
  }
}
```

### Integration Points

- Conversation orchestrator (auto-activation)
- Memory engine (pack facts in RAG)
- Chat endpoint (context injection)

### Testing

- Test pack loading
- Test pack activation
- Test topic detection
- Test context injection

### Deliverables

- Knowledge packs module
- Example packs (finances, career, technical)
- API routes for pack management
- Integration with orchestrator

---

## Phase 5: Long-Term Goals Engine

**Branch**: `feature/goals-engine`  
**Goal**: Track objectives and suggest actionable plans

### Architecture

```
src/goals/
├── __init__.py
├── tracker.py          # Goal tracking logic
├── suggestions.py      # Action suggestion logic
└── models.py           # Goal models
```

### Implementation Steps

1. **Create module structure**
   - Create `src/goals/` directory
   - Define Goal model

2. **Implement goal tracking**
   - Create goals
   - Update goal progress
   - Track milestones
   - Goal status management (active, completed, paused)

3. **Implement progress monitoring**
   - Calculate progress percentage
   - Track milestone completion
   - Deadline tracking
   - Progress history

4. **Implement action suggestions**
   - Analyze goal context
   - Suggest next steps
   - Generate actionable items
   - Learn from completed goals

5. **Create API routes**
   - `GET /v1/goals` - List goals
   - `POST /v1/goals` - Create goal
   - `PUT /v1/goals/{id}` - Update goal
   - `DELETE /v1/goals/{id}` - Delete goal
   - `GET /v1/goals/{id}/suggestions` - Get action suggestions
   - `POST /v1/goals/{id}/progress` - Update progress

6. **Integrate with memory**
   - Store goal-related facts
   - Retrieve relevant context for suggestions
   - Track goal completion patterns

### Database Schema

```sql
CREATE TABLE goals (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,  -- "active", "completed", "paused"
    deadline TIMESTAMPTZ,
    progress FLOAT DEFAULT 0.0,  -- 0.0-1.0
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE milestones (
    id UUID PRIMARY KEY,
    goal_id UUID REFERENCES goals(id),
    title TEXT NOT NULL,
    completed BOOLEAN DEFAULT false,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### Integration Points

- Memory engine (store goal facts)
- Conversation orchestrator (suggest goals in context)
- Chat endpoint (goal management)

### Testing

- Test goal CRUD operations
- Test progress tracking
- Test action suggestions
- Test milestone management

### Deliverables

- Goals engine module
- API routes for goal management
- Action suggestion system
- Integration with memory and chat

---

## Phase 6: Task Executor

**Branch**: `feature/task-executor`  
**Goal**: Execute actions via integrations (GitHub, K8s, APIs)

### Architecture

```
src/tasks/
├── __init__.py
├── executor.py         # Task execution engine
├── queue.py            # Task queue
├── integrations/       # Integration plugins
│   ├── __init__.py
│   ├── base.py         # Base integration class
│   ├── github.py       # GitHub integration
│   ├── k8s.py          # Kubernetes integration
│   └── file.py         # File operations
└── models.py           # Task models
```

### Implementation Steps

1. **Create module structure**
   - Create `src/tasks/` directory
   - Create `integrations/` subdirectory
   - Define base integration interface

2. **Implement task executor**
   - Task queue system
   - Async task execution
   - Retry logic with exponential backoff
   - Error handling and logging
   - Task status tracking

3. **Implement GitHub integration**
   - GitHub API client
   - Create PRs
   - Manage issues
   - Read repository info
   - Commit and push changes

4. **Implement Kubernetes integration**
   - kubectl wrapper
   - Deploy operations
   - Scale operations
   - Update manifests
   - Read cluster state

5. **Implement file operations**
   - Read files
   - Write files
   - Process files
   - Directory operations

6. **Create API routes**
   - `POST /v1/tasks/execute` - Execute task
   - `GET /v1/tasks/{id}/status` - Get task status
   - `GET /v1/tasks/history` - Get task history
   - `GET /v1/tasks/integrations` - List available integrations

7. **Integrate with chat**
   - Detect task requests in chat
   - Execute tasks from conversation
   - Report task results
   - Store task history

### Task Schema

```python
class Task(BaseModel):
    id: UUID
    type: str  # "github", "k8s", "file", "api"
    action: str  # "create_pr", "deploy", "read_file"
    params: Dict[str, Any]
    status: str  # "pending", "running", "completed", "failed"
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
```

### Integration Points

- Chat endpoint (detect and execute tasks)
- Memory engine (store task results)
- Goals engine (suggest tasks for goals)

### Security Considerations

- Authentication for GitHub/K8s
- Secret management
- Permission checks
- Audit logging

### Testing

- Test task execution
- Test retry logic
- Test error handling
- Test integrations (mocked)

### Deliverables

- Task executor module
- GitHub integration
- Kubernetes integration
- File operations integration
- API routes for task management
- Integration with chat

---

## Dependencies Summary

### Phase 1 (Memory Engine)
```txt
sentence-transformers>=2.2.0
psycopg[binary]>=3.1.0
sqlalchemy>=2.0.0
alembic>=1.12.0
```

### Phase 3 (Orchestrator)
```txt
# Uses dependencies from Phase 1
```

### Phase 4 (Knowledge Packs)
```txt
# Uses dependencies from Phase 1
```

### Phase 5 (Goals Engine)
```txt
# Uses dependencies from Phase 1
```

### Phase 6 (Task Executor)
```txt
pygithub>=1.59.0  # GitHub integration
kubernetes>=28.1.0  # Kubernetes integration
```

---

## Database Migrations

Each phase that adds database tables will include Alembic migrations:

- Phase 1: `facts`, `human_profiles` tables
- Phase 3: `conversations`, `messages` tables
- Phase 5: `goals`, `milestones` tables
- Phase 6: `tasks` table (optional, can use in-memory queue)

---

## Testing Strategy

### Unit Tests
- Each module has unit tests
- Mock external dependencies
- Test edge cases

### Integration Tests
- Test API endpoints
- Test module interactions
- Test database operations

### End-to-End Tests
- Test full conversation flow
- Test memory persistence
- Test task execution

---

## Documentation Updates

Each phase will update:

- `CHANGELOG.md` - Document changes
- `README.md` - Update usage examples
- `docs/architecture.md` - Update architecture docs
- API documentation (auto-generated via FastAPI)

---

## Git Workflow

For each phase:

1. Create feature branch: `git checkout -b feature/phase-name`
2. Make changes
3. Update CHANGELOG.md
4. Commit changes: `git commit -m "feat(phase-name): description"`
5. Push branch: `git push origin feature/phase-name`
6. Create PR
7. Get review
8. Merge to main
9. Delete branch

---

## Version Progression

- **v0.5.0** - Current baseline (after Phase 0)
- **v0.6.0** - After Phase 1 (Memory Engine)
- **v0.7.0** - After Phase 2 (Personality Kernel)
- **v0.8.0** - After Phase 3 (Conversation Orchestrator)
- **v0.9.0** - After Phase 4 (Knowledge Packs)
- **v0.10.0** - After Phase 5 (Goals Engine)
- **v1.0.0** - After Phase 6 (Task Executor) - Full AI Personality Engine

---

## Risk Mitigation

### Phase 1 Risks
- **Risk**: Embedding generation slow
- **Mitigation**: Cache embeddings, use smaller model initially

### Phase 3 Risks
- **Risk**: Context window too large
- **Mitigation**: Implement summarization early, set token limits

### Phase 6 Risks
- **Risk**: Security vulnerabilities in task execution
- **Mitigation**: Strict permission checks, audit logging, sandboxing

---

## Success Criteria

### Phase 1
- ✅ Facts can be stored and retrieved
- ✅ Semantic search works
- ✅ Memory persists across conversations

### Phase 2
- ✅ Responses have consistent personality
- ✅ Style rules are applied
- ✅ Persona is recognizable

### Phase 3
- ✅ Conversations persist
- ✅ Context is managed correctly
- ✅ Long conversations don't break

### Phase 4
- ✅ Packs can be loaded and activated
- ✅ Pack context is injected
- ✅ Topic detection works

### Phase 5
- ✅ Goals can be tracked
- ✅ Progress is monitored
- ✅ Suggestions are generated

### Phase 6
- ✅ Tasks can be executed
- ✅ Integrations work
- ✅ Error handling is robust

---

## Timeline Estimate

- **Phase 0**: 1-2 hours
- **Phase 1**: 2-3 days
- **Phase 2**: 1-2 days
- **Phase 3**: 2-3 days
- **Phase 4**: 1-2 days
- **Phase 5**: 2-3 days
- **Phase 6**: 3-5 days

**Total**: ~2-3 weeks for complete implementation

---

## Next Steps

1. Review and approve this plan
2. Start Phase 0 (merge conflicts and version)
3. Proceed sequentially through phases
4. Update plan as needed based on learnings

---

**Status**: Ready for implementation  
**Last Updated**: 2025-01-XX

