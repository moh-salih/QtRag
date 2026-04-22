#pragma once
#include <QString>
#include <QVector>
#include <vector>

namespace QtRag {

struct RagChunk {
    int                  id         = -1;
    int                  sessionId  = -1;
    int                  chunkIndex = -1;
    QString              text;
    std::vector<float>   embedding;
};

} // namespace QtRag
