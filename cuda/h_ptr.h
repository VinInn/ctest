#ifndef h_ptr_H
#define h_ptr_H

//#ifndef __CUDA_ARCH__
#include <thread>
#include <unordered_map>
// magic magic: edm set, clean this
// filled by SoAFromCUDA
// in real life a thread safe one-to-ne association
thread_local std::unordered_map<unsigned long long,unsigned long long> * relocationTable;
//#endif

// a heterogenous reference pointer
// no need of marshalling
// 0 cost on GPU
// a predictable branch on CPU
template<typename T>
class h_ptr {
public:
#ifdef __CUDA_ARCH__
  __device__ __forceinline__ h_ptr(T const * p=nullptr) : gpu_ptr(p){}
  __device__ __forceinline__ auto & operator=(T const * p) { gpu_ptr=p; return *this;}
#else
  __host__ __forceinline__ h_ptr(T const * p=nullptr) : host_ptr(p){}
  __host__ __forceinline__ auto & operator=(T const * p) { host_ptr=p; return *this;}
#endif

  constexpr T const * get() const {
#ifdef __CUDA_ARCH__
    return gpu_ptr;
#else
    if (host_ptr) return host_ptr;
    else {
     if (!gpu_ptr) return nullptr;  // if null is null
     // look in relocation table
     host_ptr = findptr(gpu_ptr);
     // throw if null?
     return host_ptr;
    } 
#endif
  }

private:
#ifndef __CUDA_ARCH__
  static T const * findptr(T const * p) {
    auto ipg = (unsigned long long)(p);
    auto el = relocationTable->find(ipg);
    if (el==relocationTable->end()) return nullptr;
    return (T const *)(el->second); 
  }
#endif

  T const * gpu_ptr = nullptr;
  mutable T const * host_ptr=nullptr;
};
#endif
