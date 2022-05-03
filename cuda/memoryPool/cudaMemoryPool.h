#pragma once
#include "memoryPool.h"
#include <vector>
#include <cuda.h>


namespace memoryPool {
  namespace cuda {

    void dumpStat();

    FastPoolAllocator * getPool(bool onDevice);

    // allocate either on current device or on host
    std::pair<void *,int> alloc(uint64_t size, FastPoolAllocator & pool);

    // schedule free
    void free(cudaStream_t stream, std::vector<int> buckets, FastPoolAllocator & pool);
    
    struct DeleteOne final : public DeleterBase {

      explicit DeleteOne(cudaStream_t const & stream, FastPoolAllocator * pool) :  
           m_stream(stream), m_pool(pool) {}
    
      ~DeleteOne() override = default;
      void operator()(int bucket) override {
          free(m_stream, std::vector<int>(1,bucket), *m_pool);
      }

      cudaStream_t m_stream;
      FastPoolAllocator * m_pool;

    };

    struct BundleDelete final : public DeleterBase {

      explicit BundleDelete(cudaStream_t const & stream, FastPoolAllocator * pool) : 
            m_stream(stream), m_pool(pool) {}

      ~BundleDelete() override {
         free(m_stream, std::move(m_buckets),  *m_pool);
      }

      void operator()(int bucket) override {
         m_buckets.push_back(bucket);
      }

      cudaStream_t m_stream;
      std::vector<int> m_buckets;
      FastPoolAllocator    * m_pool;

    };

    namespace device {
     template<typename T>
      unique_ptr<T> make_unique(uint64_t size, Deleter del) {
        auto ret = alloc(sizeof(T)*size,*getPool(true));
        if (ret.second<0) throw std::bad_alloc();
        del.m_bucket = ret.second;
        return unique_ptr<T>((T*)(ret.first),del);
      }

      template<typename T>
      unique_ptr<T> make_unique(uint64_t size, cudaStream_t const & stream) {
         return make_unique<T>(sizeof(T)*size,Deleter(std::make_shared<DeleteOne>(stream,getPool(true))));
      }
    }

    namespace host {
      template<typename T>
      unique_ptr<T> make_unique(uint64_t size);
      template< class T>
      unique_ptr<T> make_unique(uint64_t size, Deleter del);

      template< class T, class... Args >
      memoryPool::unique_ptr<T> make_unique( Args&&... args );
      template< class T, class... Args >
      memoryPool::unique_ptr<T> make_unique(Deleter del, Args&&... args );
    }
  } // cuda
} // memoryPool
