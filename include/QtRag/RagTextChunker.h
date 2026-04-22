#pragma once
#include <QStringList>
#include <vector>
#include "RagConfig.h"
#include "RagTypes.h"

namespace QtRag {

class RagTextChunker {
public:
    static QStringList        splitIntoChunks(const QString& text, const ChunkingConfig& config = ChunkingConfig());
    static QString            rankRelevantContext(const std::vector<float>& queryVec,
                                                   const QVector<RagChunk>& chunks,
                                                   const SearchConfig& config = SearchConfig());
private:
    static float calculateCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    static bool  checkQuality(const QString& text, const ChunkingConfig& config);
};

} // namespace QtRag
