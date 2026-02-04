# KB Platform - Python Services

## Overview
This repository contains the core logic for the Knowledge Base Platform, split into two logical components sharing the same codebase:

1. **LlamaIndex Core (gRPC Server)**: 
   - Handles document parsing, vector embedding, and RAG queries.
   - Exposes a gRPC interface for the Go Gateway.
   
2. **Temporal Worker**:
   - Handles background document ingestion workflows.
   - Orchestrates S3 downloads, parsing, and Qdrant indexing.

## Directory Structure
- `src/core`: gRPC Server application
- `src/worker`: Temporal Worker application
- `src/shared`: Shared database models, parsers, and utilities

## Tech Stack
- Python 3.11+
- LlamaIndex
- Qdrant Client
- Temporal SDK
- gRPC (grpcio)
