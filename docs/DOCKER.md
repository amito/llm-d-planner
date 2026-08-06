# Docker Deployment Guide for Planner

This guide explains how to run Planner using Docker and Docker Compose.

## Overview

Planner is containerized into the following services:

- **ollama**: Ollama LLM service for intent extraction (qwen2.5:7b model)
- **backend**: FastAPI REST API server
- **ui**: Streamlit web interface
- **simulator**: vLLM simulator for GPU-free development (optional)

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose v2.0+
- At least 8GB of RAM available for Docker
- 10GB of free disk space (for Ollama models)

## Quick Start

**Note:** This guide shows raw `docker-compose` commands. For convenience, you can use `make` commands instead (see [Makefile Commands](#makefile-commands-convenience-shortcuts) section below). Run `make help` to see all available shortcuts.

### Production Mode

Run the full stack (backend, UI, Ollama):

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

Access the application:
- **UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Makefile Commands (Convenience Shortcuts)

This repository includes a **Makefile** that provides convenient shortcuts for all Docker operations. You can use either the raw `docker-compose` commands shown throughout this guide OR the equivalent `make` commands.

**Why use Makefile commands?**
- Shorter, easier to remember commands
- Colored output for better visibility
- Built-in health checks and status messages
- Consistent cross-platform behavior (macOS/Linux)

### Quick Command Reference

| Task | Docker Compose Command | Makefile Shortcut |
|------|------------------------|-------------------|
| **Build images** | `docker-compose build` | `make image-build` |
| **Start services** | `docker-compose up -d` | `make docker-up` |
| **Start dev mode** | `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up` | `make docker-up-dev` |
| **Stop services** | `docker-compose down` | `make docker-down` |
| **Stop + remove volumes** | `docker-compose down -v` | `make docker-down-v` |
| **View logs** | `docker-compose logs -f` | `make docker-logs` |
| **Check status** | `docker-compose ps` | `make docker-ps` |

### Examples

**Using docker-compose:**
```bash
docker-compose build
docker-compose up -d
docker-compose logs -f backend
docker-compose down
```

**Using Makefile (equivalent):**
```bash
make image-build
make docker-up
make docker-logs
make docker-down
```

**Complete list of available commands:**
```bash
make help  # Show all available Makefile targets
```

### Development Mode

Run with hot reload enabled for code changes:

```bash
# Using docker-compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Using Makefile (recommended)
make docker-up-dev
```

### With Simulator

Run the vLLM simulator for GPU-free testing:

```bash
# Start with simulator profile
docker-compose --profile simulator up -d

# Simulator will be available at http://localhost:8080
```

## Common Operations

All operations below show both `docker-compose` commands and their `make` equivalents. Use whichever you prefer!

### Building Images

```bash
# Build all images
docker-compose build
make image-build

# Build specific images
make image-build-backend
make image-build-ui
make image-build-simulator

# Build specific service (docker-compose only)
docker-compose build backend

# Force rebuild without cache (docker-compose only)
docker-compose build --no-cache
```

### Managing Services

```bash
# Start services
docker-compose up -d
make docker-up

# Stop services
docker-compose down
make docker-down

# Stop and remove volumes (WARNING: deletes database data)
docker-compose down -v
make docker-down-v

# Restart services
docker-compose restart

# Restart specific service
docker-compose restart backend

# View logs (all services)
docker-compose logs -f
make docker-logs

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f ui

# Check service status
docker-compose ps
make docker-ps
```

### Database Operations

```bash
# Load benchmark data (runs inside backend container)
docker-compose exec backend python scripts/load_benchmarks.py data/benchmarks/performance/benchmarks_BLIS.json
```

### Ollama Operations

```bash
# Pull a different model
docker-compose exec ollama ollama pull llama3.2

# List installed models
docker-compose exec ollama ollama list

# Run model interactively
docker-compose exec ollama ollama run qwen2.5:7b
```

## Configuration

### Environment Variables

Create a `.env` file in the project root to customize configuration:

```env
# Ollama Configuration
LLM_MODEL=qwen2.5:7b

# Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# UI Configuration
STREAMLIT_SERVER_PORT=8501
```

### Port Mapping

Default port mappings (can be customized in docker-compose.yml):

| Service   | Container Port | Host Port |
|-----------|----------------|-----------|
| UI        | 8501           | 8501      |
| Backend   | 8000           | 8000      |
| Ollama    | 11434          | 11434     |
| Simulator | 8080           | 8080      |

To change host ports, edit `docker-compose.yml`:

```yaml
services:
  ui:
    ports:
      - "8080:8501"  # Access UI on http://localhost:8080
```

## Development Workflow

### Hot Reload

Both backend and UI support hot reload in development mode:

1. **Backend**: Uvicorn's `--reload` flag watches for Python file changes
2. **UI**: Streamlit automatically reloads when files change

Start in development mode:

```bash
# Using docker-compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Using Makefile (recommended)
make docker-up-dev
```

Edit files in `src/planner/` or `ui/` - changes will be reflected immediately.

### Debugging

#### View Container Logs

```bash
# Follow logs for all services
docker-compose logs -f
make docker-logs

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f ui
```

#### Access Container Shell

```bash
# Backend container
docker-compose exec backend /bin/bash

# UI container
docker-compose exec ui /bin/bash
```

#### Check Service Health

```bash
# View service status
docker-compose ps
make docker-ps

# Test health endpoints
curl http://localhost:8000/health  # Backend
curl http://localhost:8501/_stcore/health  # UI
curl http://localhost:11434/  # Ollama
```

## Architecture

### Network Topology

All services communicate via a bridge network (`planner-network`):

```
┌─────────────────────────────────────────────┐
│           planner-network (bridge)          │
│                                             │
│  ┌─────────┐    ┌─────────┐                 │
│  │   UI    │───▶│ Backend │                 │
│  │ :8501   │    │  :8000  │                 │
│  └─────────┘    └────┬────┘                 │
│                      │                      │
│                      ▼                      │
│                 ┌─────────┐                 │
│                 │ Ollama  │                 │
│                 │ :11434  │                 │
│                 └─────────┘                 │
└─────────────────────────────────────────────┘
         │         │         │
         ▼         ▼         ▼
    localhost  localhost localhost
      :8501     :8000     :11434
```

### Volume Mounts

**Development Volumes** (source code - hot reload):
- `./src/planner` → `/app/src/planner`
- `./ui` → `/app/ui`
- `./data` → `/app/data`

**Persistent Volumes** (data):
- `ollama_data`: Ollama models and cache

## Troubleshooting

### Services Won't Start

**Check Docker resources:**
```bash
docker system df
docker system prune  # Free up space
```

**Check service logs:**
```bash
docker-compose logs backend
docker-compose logs ollama
```

**Verify dependencies:**
```bash
# Ensure Ollama model is downloaded
docker-compose exec ollama ollama list
```

### Ollama Model Not Loading

**Check if model is downloaded:**
```bash
docker-compose exec ollama ollama list
```

**Manually pull model:**
```bash
docker-compose exec ollama ollama pull qwen2.5:7b
```

**Check Ollama logs:**
```bash
docker-compose logs ollama
```

### Port Conflicts

**Error**: "port is already allocated"

**Solution**: Change host port in `docker-compose.yml`:
```yaml
services:
  ui:
    ports:
      - "8502:8501"  # Use different host port
```

### Slow Performance

**Increase Docker resources:**
- Docker Desktop → Preferences → Resources
- Allocate more CPU cores and RAM (recommended: 4 CPUs, 8GB RAM)

**Check resource usage:**
```bash
docker stats
```

## Production Deployment

### Security Checklist

Before deploying to production:

- [ ] Use secrets management (Docker Swarm secrets or Kubernetes secrets)
- [ ] Enable TLS/HTTPS with reverse proxy (nginx, traefik)
- [ ] Configure firewall rules
- [ ] Set `DEBUG=false` in environment variables
- [ ] Review CORS settings in backend
- [ ] Enable authentication/authorization

### Using Reverse Proxy

Example nginx configuration:

```nginx
server {
    listen 80;
    server_name planner.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

### Resource Requirements

**Minimum (Development)**:
- 2 CPU cores
- 8GB RAM
- 10GB disk space

**Recommended (Production)**:
- 4+ CPU cores
- 16GB RAM
- 50GB disk space (for Ollama models)

## Next Steps

- See [README.md](README.md) for overall project documentation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system architecture
- See [docs/TESTING.md](TESTING.md) for testing procedures

## Support

For issues and questions:
- Check [GitHub Issues](https://github.com/yourusername/planner/issues)
- Review Docker logs: `docker-compose logs -f`
- Verify service health: `docker-compose ps`
