# Local Agentic RAG Learning Coach

A small local AI learning agent that explains topics and creates quizzes from
local notes. It can decide when to search the notes and remembers conversation
history within the same session.

## Project flow

```text
Startup:
Documents → chunks → embeddings → FAISS index

Request:
User + memory → agent decision
                    ├── answer directly
                    └── search_notes → FAISS results → final answer
```

The fixed RAG service always retrieves context before generating an answer. The
agentic version exposes retrieval as a tool, allowing the model to decide when
retrieval is necessary.

## Main technical concepts

- **Ollama:** runs the chat and embedding models locally.
- **Embeddings:** convert text into vectors that represent meaning.
- **FAISS:** ranks document chunks using vector similarity.
- **RAG:** gives retrieved knowledge to the model before it answers.
- **Tool calling:** lets the model invoke `search_notes` when needed.
- **Agent loop:** model decides, calls a tool, observes the result, and answers.
- **LangGraph memory:** stores conversation messages using a `thread_id`.
- **Evaluation:** checks tool selection and memory behaviour instead of exact wording.

## Test the layers

Confirm the Docker models if using docker

```bash
docker exec <container_name> ollama list
```

Run each layer:

```bash
python -m app.main
python -m app.knowledge_demo
python -m app.rag_demo
python -m app.agent_demo
python -m app.evaluation_demo
```

The final evaluation should report `4/4 passed`.
