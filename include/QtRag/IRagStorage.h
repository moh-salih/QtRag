// IRagStorage.h
#pragma once
#include "RagTypes.h"

namespace QtRag {
class IRagStorage {
public:
    virtual ~IRagStorage() = default;
    virtual bool              saveChunk(int sessionId, const QString& text,
                                        int chunkIndex, const std::vector<float>& embedding) = 0;
    virtual QVector<RagChunk> loadChunks(int sessionId) = 0;
    virtual int               getChunkCount(int sessionId) = 0;
    virtual void              clearSession(int sessionId) = 0;
};
} // namespace QtRag
