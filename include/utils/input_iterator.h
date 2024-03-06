#ifndef INPUT_PROCESSOR_H
#define INPUT_PROCESSOR_H

#include <vector>
#include <functional>

class DataSpan {
private :
    int index;
    int start;
    int end;

public:
    DataSpan(int index, int start, int end);

    int getIndex() const;

    int getStart() const;

    int getEnd() const;

    int getSize() const;
};

class Chunk : public DataSpan {
public:
    Chunk(int index, int start, int end) : DataSpan(index, start, end) {};
};

using ChunkCallback = std::function<void(Chunk)>;

class Batch : public DataSpan {
public:
    Batch(int index, int start, int end) : DataSpan(index, start, end) {};
};
using BatchCallback = std::function<void(Batch)>;


/**
 * @brief Splits input into chunks and batches and provides an interface to interating over them
 * 
 * @tparam T 
 */
template <typename T>
class InputIterator {
private:
    /// @brief Input data
    std::vector<T> * input;

    /// @brief Number of tokens in a context
    int n_ctx;

    /// @brief Number of tokens in a batch
    int n_batch;

    /// @brief Number of chunks to process
    int n_chunk;

    /// @brief Number of batches in a chunk
    int num_batches;
public:
    InputIterator(std::vector<T> * input, int n_ctx, int n_batch, int n_chunk = -1);

    /// @brief Get number of complete chunks in the input
    int getChunksNumber();

    /// @brief Get number of batches in a chunk
    int getBatchesNumber();

    /// @brief Iterates over all chunks of the input
    /// @param callback is called for every `Chunk`
    void iterate(ChunkCallback callback);

    /// @brief Iterates over all batches in the `chunk`
    /// @param callback is called for every `Batch`
    /// @param chunk Chunk that is splitted into batches
    void iterate(BatchCallback callback, Chunk chunk);

    std::vector<T> * getInput();
};

#endif // INPUT_PROCESSOR_H