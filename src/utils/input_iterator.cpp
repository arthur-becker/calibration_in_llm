#include "common.h"
#include "utils/input_iterator.h"
#include <vector>
#include <span>

/*
 * DataSpan class
 */

DataSpan::DataSpan(int index, int start, int end){
    this->index = index;
    this->start = start;
    this->end = end;
}

int DataSpan::getIndex() const{
    return this->index;
}

int DataSpan::getStart() const{
    return this->start;
}

int DataSpan::getEnd() const{
    return this->end;
}

int DataSpan::getSize() const{
    return this->end - this->start;
}


/*
* InputIterator class
*/


template <typename T>
InputIterator<T>::InputIterator(std::vector<T> * input, int n_ctx, int n_batch, int n_chunk){ 
    this->input = input;
    this->n_ctx = n_ctx;
    this->n_batch = n_batch;
        
    const int n_chunk_max = input->size() / n_ctx;
    this->n_chunk = n_chunk < 0 ? n_chunk_max : std::min(n_chunk, n_chunk_max);
    this->num_batches = (n_ctx + n_batch - 1) / n_batch;
} 


template <typename T>
int InputIterator<T>::getChunksNumber(){
    return this->n_chunk;
}

template <typename T>
int InputIterator<T>::getBatchesNumber(){
    return this->num_batches;
}

template <typename T>
void InputIterator<T>::iterate(const ChunkCallback& callback){
        for (int i = 0; i < this->n_chunk; i++) {
            const int start = i * this->n_ctx;
            const int end = start + this->n_ctx;
            Chunk chunk(i, start, end);
            callback(chunk);
        }
}

template <typename T>
void InputIterator<T>::iterate(const BatchCallback& callback, Chunk chunk){
    for (int i = 0; i < this->num_batches; i++) {
        const int start = chunk.getStart() + i * this->n_batch;
        const int batch_size  = std::min(chunk.getEnd() - start, this->n_batch);
        const int end = start + batch_size;
        Batch batch(i, start, end);
        callback(batch);
    }
}

template <typename T>
std::vector<T> * InputIterator<T>::getInput(){
    return this->input;
}

template class InputIterator<char>;

template class InputIterator<llama_token>;
