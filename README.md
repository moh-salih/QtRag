# QtRag

A RAG (Retrieval-Augmented Generation) library for Qt6 applications.

## What it does

QtRag handles text chunking, embedding generation coordination, and HNSW vector search. The library provides async, non-blocking operations that work with Qt's signal/slot system. Multiple sessions are supported.

## Design choices

The library does not include a database backend. You implement `IRagStorage` yourself. This gives you freedom to use whatever storage fits your project - SQLite, PostgreSQL with pgvector, files, or something else.

Similarly, `IRagEmbedder` is an interface. You provide the actual embedding generation (local model, API call, etc.).

## Requirements

- Qt6 Core
- hnswlib

## Basic example

```cpp
class MyEmbedder : public QtRag::IRagEmbedder {
    void generateEmbedding(const QString& text, int chunkIndex) override {
        // your embedding logic here
        emit embeddingReady(embedding, text, chunkIndex);
    }
};

class MyStorage : public QtRag::IRagStorage {
    // implement saveChunk, loadChunks, getChunkCount, clearSession
};

// Usage
QtRag::RagConfig config;
config.chunking.maxChunkLength = 500;
config.search.topK = 3;

MyEmbedder embedder;
MyStorage storage;

QtRag::RagEngine engine(&embedder, &storage, config);
engine.indexText(documentText, sessionId);
engine.requestContext("your query here", sessionId);

connect(&engine, &QtRag::RagEngine::contextReady,
        [](const QString& context, int sessionId) {
            // handle result
        });
```

## Configuration options

| Section | Parameters |
|---------|------------|
| Chunking | min/max chunk length, overlap tokens, optional quality filtering |
| Index | dimension (must match embedder), M, efConstruction, initial capacity |
| Search | topK, minimum similarity threshold |

## Building

```cmake
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(hnswlib REQUIRED)
target_link_libraries(your_app QtRag::QtRag)
```


