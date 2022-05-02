#pragma once
#include "memoryPool.h"
#include <vector>
#include <cuda.h>


namespace memoryPool {
  namespace cuda {

    void dumpStat();

    // allocate either on current device or on host
    std::pair<void *,int> alloc(uint64_t size, bool onDevice);

    // schedule free
    void free(cudaStream_t stream, std::vector<int> buckets, bool onDevice);
    
    struct DeleteOne final : public DeleterBase {

      explicit DeleteOne(cudaStream_t const & stream, bool onDevice) :  
           m_stream(stream), m_onDevice(onDevice) {}
    
      ~DeleteOne() override = default;
      void operator()(int bucket) override {
          free(m_stream, std::vector<int>(1,bucket), m_onDevice);
      }

      cudaStream_t m_stream;
      bool m_onDevice;

    };

    struct BundleDelete final : public DeleterBase {

      explicit BundleDelete(cudaStream_t const & stream, bool onDevice) : 
            m_stream(stream), m_onDevice(onDevice) {}

      ~BundleDelete() override {
         free(m_stream, std::move(m_buckets),  m_onDevice);
      }

      void operator()(int bucket) override {
         m_buckets.push_back(bucket);
      }

      cudaStream_t m_stream;
      std::vector<int> m_buckets;
      bool m_onDevice;

    };

    namespace device {
     template<typename T>
      unique_ptr<T> make_unique(uint64_t size, Deleter del) {
        auto ret = alloc(size,true);
        if (ret.second<0) throw std::bad_alloc();
        del.m_bucket = ret.second;
        return unique_ptr<T>((T*)(ret.first),del);
      }

      template<typename T>
      unique_ptr<T> make_unique(uint64_t size, cudaStream_t const & stream) {
         return make_unique<T>(size,Deleter(std::make_shared<DeleteOne>(stream,true)));
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
