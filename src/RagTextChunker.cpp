#include "QtRag/RagTextChunker.h"
#include <cmath>
#include <numeric>
#include <QRegularExpression>

namespace QtRag {

QStringList RagTextChunker::splitIntoChunks(const QString& text, const ChunkingConfig& config) {
    QStringList result;
    if (text.trimmed().isEmpty()) return result;

    // Split on sentence boundaries first
    QStringList sentences = text.split(QRegularExpression(R"([.!?]+\s+)"), Qt::SkipEmptyParts);

    QString current;
    for (const QString& sentence : sentences) {
        if (current.length() + sentence.length() > config.maxChunkLength && !current.isEmpty()) {
            if (checkQuality(current, config))
                result.append(current.trimmed());
            // Keep overlap
            current = config.overlapTokens > 0
                ? current.right(config.overlapTokens) + " " + sentence
                : sentence;
        } else {
            current += (current.isEmpty() ? "" : " ") + sentence;
        }
    }
    if (!current.trimmed().isEmpty() && checkQuality(current, config))
        result.append(current.trimmed());

    return result;
}

QString RagTextChunker::rankRelevantContext(const std::vector<float>& queryVec,
                                             const QVector<RagChunk>& chunks,
                                             const SearchConfig& config) {
    struct Scored { float score; const RagChunk* chunk; };
    std::vector<Scored> scored;
    scored.reserve(chunks.size());

    for (const auto& chunk : chunks) {
        if (chunk.embedding.empty()) continue;
        scored.push_back({ calculateCosineSimilarity(queryVec, chunk.embedding), &chunk });
    }

    std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b){
        return a.score > b.score;
    });

    QString context;
    int count = 0;
    for (const auto& s : scored) {
        if (count >= config.topK) break;
        if (config.minSimilarity > 0.0f && s.score < config.minSimilarity) break;
        context += "- " + s.chunk->text + "\n";
        ++count;
    }
    return context.trimmed();
}

float RagTextChunker::calculateCosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    float denom = std::sqrt(normA) * std::sqrt(normB);
    return denom > 0.0f ? dot / denom : 0.0f;
}

bool RagTextChunker::checkQuality(const QString& text, const ChunkingConfig& config) {
    if (!config.filterLowQuality) return true;
    QString trimmed = text.trimmed();
    if (trimmed.length() < config.minChunkLength) return false;
    // Reject chunks that are mostly non-alphabetic
    int alphaCount = 0;
    for (const QChar& c : trimmed)
        if (c.isLetter()) ++alphaCount;
    return (static_cast<float>(alphaCount) / trimmed.length()) > 0.4f;
}

} // namespace QtRag
