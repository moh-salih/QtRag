#pragma once
#include <QtGlobal>

namespace QtRag {

    struct ChunkingConfig {
        int     minChunkLength   = 40;
        int     maxChunkLength   = 400;
        int     overlapTokens    = 20;
        bool    filterLowQuality = true;
        float   lengthBoostFactor = 1.0f;  
        float   fillerWordLimit   = 0.6f;
    };

    struct IndexConfig {
        int     dim             = 384;   // must match your embedding model
        int     M               = 16;    // HNSW M parameter
        int     efConstruction  = 200;
        size_t  initialCapacity = 500;
    };

    struct SearchConfig {
        int     topK            = 5;
        float   minSimilarity   = 0.0f;  // optional threshold, 0 = disabled
    };

    // Aggregate — what RagEngine actually takes
    struct RagConfig {
        ChunkingConfig  chunking;
        IndexConfig     index;
        SearchConfig    search;
    };

} // namespace QtRag
