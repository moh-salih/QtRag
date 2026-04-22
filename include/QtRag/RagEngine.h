#pragma once
#include <QObject>
#include <QMutex>
#include <QReadWriteLock>
#include <QHash>
#include <QVector>
#include <queue>
#include <vector>

#include "RagConfig.h"
#include "RagTypes.h"

#include <hnswlib/hnswlib.h>

namespace QtRag {

class IRagEmbedder;
class IRagStorage;

class RagEngine : public QObject {
    Q_OBJECT
public:
    explicit RagEngine(IRagEmbedder* embedder,
                       IRagStorage*  storage,
                       const RagConfig& config = RagConfig(),
                       QObject* parent = nullptr);
    ~RagEngine() override;

    void indexText(const QString& text, int sessionId);
    void requestContext(const QString& query, int sessionId);
    void rebuildIndex(int sessionId);
    void resetSession(int sessionId);

signals:
    void contextReady(const QString& context, int sessionId);
    void errorOccurred(const QString& error);

private slots:
    void onEmbeddingReady(const std::vector<float>& embedding, const QString& text, int chunkIndex);
    void onEmbeddingError(const QString& error);

private:
    struct EmbeddingTask {
        QString text;
        int     sessionId;
        int     chunkIndex; // -1 = search query, >= 0 = index
    };

    void    processNextInQueue();
    void    handleTaskFailure();
    QString performHnswSearch(const std::vector<float>& queryVec, int sessionId);
    int     getNextChunkIndex(int sessionId);

    // Dependencies
    IRagEmbedder*  mEmbedder;
    IRagStorage*   mStorage;
    RagConfig      mConfig;

    // HNSW (protected by mIndexLock)
    hnswlib::HierarchicalNSW<float>* mHnswIndex = nullptr;
    hnswlib::L2Space*                mSpace      = nullptr;
    QVector<RagChunk>                mLoadedChunks;
    QReadWriteLock                   mIndexLock;

    // Queue state (protected by mQueueMutex)
    std::queue<EmbeddingTask> mTaskQueue;
    QHash<int, int>           mCounters;
    bool                      mIsProcessing = false;
    QMutex                    mQueueMutex;
};

} // namespace QtRag
