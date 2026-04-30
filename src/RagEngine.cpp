#include "QtRag/RagEngine.h"
#include "QtRag/IRagEmbedder.h"
#include "QtRag/IRagStorage.h"
#include "QtRag/RagTextChunker.h"
#include "QtRag/RagConfig.h"

#include <QMutexLocker>
#include <QReadLocker>
#include <QWriteLocker>
#include <QDebug>
#include <algorithm>

namespace QtRag {

RagEngine::RagEngine(IRagEmbedder* embedder, IRagStorage* storage,
                     const RagConfig& config, QObject* parent)
    : QObject(parent)
    , mEmbedder(embedder)
    , mStorage(storage)
    , mConfig(config)
{
    connect(mEmbedder, &IRagEmbedder::embeddingReady, this, &RagEngine::onEmbeddingReady);
    // IRagEmbedder::errorOccurred still emits a QString from the adapter layer
    // in LiveTextify — we map it here to our typed enum.
    connect(mEmbedder, &IRagEmbedder::errorOccurred,  this, &RagEngine::onEmbeddingError);
}

RagEngine::~RagEngine() {
    QWriteLocker locker(&mIndexLock);
    delete mHnswIndex; mHnswIndex = nullptr;
    delete mSpace;     mSpace     = nullptr;
}

// ── Public ────────────────────────────────────────────────────────────────────

void RagEngine::indexText(const QString& text, int sessionId) {
    QStringList chunks = RagTextChunker::splitIntoChunks(text, mConfig.chunking);
    if (chunks.isEmpty()) return;

    {
        QMutexLocker locker(&mQueueMutex);
        for (const QString& chunk : chunks)
            mTaskQueue.push({ chunk, sessionId, getNextChunkIndex(sessionId) });
    }
    processNextInQueue();
}

void RagEngine::requestContext(const QString& query, int sessionId) {
    {
        QMutexLocker locker(&mQueueMutex);
        mTaskQueue.push({ query, sessionId, -1 });
    }
    processNextInQueue();
}

void RagEngine::rebuildIndex(int sessionId) {
    QVector<RagChunk> chunks = mStorage->loadChunks(sessionId);

    const auto& idx = mConfig.index;
    auto* space = new hnswlib::L2Space(static_cast<size_t>(idx.dim));
    size_t maxElements = std::max(idx.initialCapacity, static_cast<size_t>(chunks.size()) * 2);
    auto* index = new hnswlib::HierarchicalNSW<float>(space, maxElements, idx.M, idx.efConstruction);

    for (int i = 0; i < chunks.size(); ++i) {
        if (chunks[i].embedding.size() == static_cast<size_t>(idx.dim))
            index->addPoint(chunks[i].embedding.data(), i);
    }

    QWriteLocker locker(&mIndexLock);
    delete mHnswIndex; mHnswIndex = nullptr;
    delete mSpace;     mSpace     = nullptr;
    mLoadedChunks = chunks;
    mSpace        = space;
    mHnswIndex    = index;
}

void RagEngine::resetSession(int sessionId) {
    QMutexLocker locker(&mQueueMutex);
    mCounters.remove(sessionId);
    std::queue<EmbeddingTask> empty;
    std::swap(mTaskQueue, empty);
    mIsProcessing = false;
}

// ── Private slots ─────────────────────────────────────────────────────────────

void RagEngine::onEmbeddingReady(const std::vector<float>& embedding, const QString& text, int chunkIndex) {
    EmbeddingTask task;
    {
        QMutexLocker locker(&mQueueMutex);
        if (mTaskQueue.empty()) { mIsProcessing = false; return; }
        task = mTaskQueue.front();
        mTaskQueue.pop();
        mIsProcessing = false;
    }

    if (chunkIndex >= 0) {
        bool saved = mStorage->saveChunk(task.sessionId, text, chunkIndex, embedding);
        if (!saved) {
            qCritical() << "QtRag:" << errorToString(Error::StorageFailed);
            emit errorOccurred(Error::StorageFailed);
        }

        const auto& idx = mConfig.index;
        if (saved && embedding.size() == static_cast<size_t>(idx.dim)) {
            QWriteLocker locker(&mIndexLock);
            if (mHnswIndex) {
                RagChunk chunk{ -1, task.sessionId, chunkIndex, text, embedding };
                mLoadedChunks.append(chunk);
                mHnswIndex->addPoint(embedding.data(), mLoadedChunks.size() - 1);
            }
        }
    } else {
        QString context = performHnswSearch(embedding, task.sessionId);
        emit contextReady(context, task.sessionId);
    }

    processNextInQueue();
}

void RagEngine::onEmbeddingError(const QString& /*error*/) {
    // The adapter (AppRagEmbedder in LiveTextify) still forwards a QString
    // from IRagEmbedder. We discard the string here and emit the typed enum.
    qCritical() << "QtRag:" << errorToString(Error::EmbeddingFailed);
    emit errorOccurred(Error::EmbeddingFailed);
    handleTaskFailure(Error::EmbeddingFailed);
}

// ── Private ───────────────────────────────────────────────────────────────────

void RagEngine::processNextInQueue() {
    QMutexLocker locker(&mQueueMutex);
    if (mIsProcessing || mTaskQueue.empty()) return;
    mIsProcessing = true;
    EmbeddingTask task = mTaskQueue.front();
    locker.unlock();
    mEmbedder->generateEmbedding(task.text, task.chunkIndex);
}

void RagEngine::handleTaskFailure(QtRag::Error error) {
    EmbeddingTask task;
    bool hasTask = false;
    {
        QMutexLocker locker(&mQueueMutex);
        if (!mTaskQueue.empty()) {
            task = mTaskQueue.front();
            mTaskQueue.pop();
            hasTask = true;
        }
        mIsProcessing = false;
    }
    Q_UNUSED(error)
    if (hasTask && task.chunkIndex < 0)
        emit contextReady("", task.sessionId);

    processNextInQueue();
}

QString RagEngine::performHnswSearch(const std::vector<float>& queryVec, int sessionId) {
    QReadLocker locker(&mIndexLock);
    const auto& search = mConfig.search;
    const auto& idx    = mConfig.index;

    if (!mHnswIndex || queryVec.size() != static_cast<size_t>(idx.dim) || mLoadedChunks.isEmpty())
        return "";

    try {
        auto result = mHnswIndex->searchKnn(queryVec.data(), search.topK);

        struct Hit { float dist; size_t id; };
        std::vector<Hit> hits;
        hits.reserve(result.size());
        while (!result.empty()) {
            hits.push_back({ result.top().first, result.top().second });
            result.pop();
        }
        std::reverse(hits.begin(), hits.end());

        QString context;
        for (const auto& hit : hits) {
            if (search.minSimilarity > 0.0f && hit.dist > search.minSimilarity) continue;
            if (hit.id < static_cast<size_t>(mLoadedChunks.size()))
                context += "- " + mLoadedChunks[hit.id].text + "\n";
        }
        return context.trimmed();
    } catch (...) {
        qCritical() << "QtRag:" << errorToString(Error::IndexSearchFailed);
        emit errorOccurred(Error::IndexSearchFailed);
        return "";
    }
}

int RagEngine::getNextChunkIndex(int sessionId) {
    if (!mCounters.contains(sessionId))
        mCounters[sessionId] = mStorage->getChunkCount(sessionId);
    return mCounters[sessionId]++;
}

} // namespace QtRag
