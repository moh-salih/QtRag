// IRagEmbedder.h
#pragma once
#include <QObject>
#include <vector>

namespace QtRag {
class IRagEmbedder : public QObject {
    Q_OBJECT
public:
    explicit IRagEmbedder(QObject* parent = nullptr) : QObject(parent) {}
    virtual void generateEmbedding(const QString& text, int chunkIndex) = 0;
signals:
    void embeddingReady(const std::vector<float>& embedding, const QString& text, int chunkIndex);
    void errorOccurred(const QString& error);
};
} // namespace QtRag
