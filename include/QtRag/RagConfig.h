#pragma once
#include <QObject>
#include <QString>
#include <QtGlobal>

namespace QtRag {
    Q_NAMESPACE

    enum class Error {
        EmbeddingFailed,
        StorageFailed,
        IndexSearchFailed
    };
    Q_ENUM_NS(Error);

    inline QString errorToString(Error error) {
        switch (error) {
            case Error::EmbeddingFailed:   return "Embedding generation failed.";
            case Error::StorageFailed:     return "Failed to save chunk to storage.";
            case Error::IndexSearchFailed: return "HNSW index search encountered an error.";
            default:                       return "Unknown error.";
        }
    }

    struct ChunkingConfig {
        int     minChunkLength   = 40;
        int     maxChunkLength   = 400;
        int     overlapTokens    = 20;
        bool    filterLowQuality = true;
        float   lengthBoostFactor = 1.0f;
        float   fillerWordLimit   = 0.6f;
    };

    struct IndexConfig {
        int     dim             = 384;
        int     M               = 16;
        int     efConstruction  = 200;
        size_t  initialCapacity = 500;
    };

    struct SearchConfig {
        int     topK            = 5;
        float   minSimilarity   = 0.0f;
    };

    struct RagConfig {
        ChunkingConfig  chunking;
        IndexConfig     index;
        SearchConfig    search;
    };

} // namespace QtRag

Q_DECLARE_METATYPE(QtRag::Error)
