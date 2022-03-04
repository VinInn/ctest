#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <cmath>
#include <chrono>

#include "device_selector.hpp"

#include "cr_log.h"


// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};


// the function to test) 
inline float foo(float x)  { return 
   as_logf(x); 
   // acosh(x);
   // x*x;
  }

void wrapper(sycl::queue &q, const int * k, const float *a, float * ret, size_t size) {

  sycl::range<1> num_items{size};

  // the kernel
  auto kernel = [=](auto i) { ret[i] = foo(a[i]); };
  

  // warm
  for (int j=0; j<10; ++j) {
    auto e = q.parallel_for(num_items,kernel);
    e.wait();
  }

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;


  for (int j=0; j<100; ++j) {
    delta -= (std::chrono::high_resolution_clock::now()-start);
    auto e = q.parallel_for(num_items,kernel);
    e.wait();
    delta += (std::chrono::high_resolution_clock::now()-start);
  }
  std::cout <<"On device Computation took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
            << " ms" << std::endl;

  delta = start-start;;
  delta -= (std::chrono::high_resolution_clock::now()-start);
  for (int j=0; j<100; ++j) {
    auto e = q.submit([&](auto &h) {h.parallel_for(num_items,kernel);});
  }
  q.wait();
  delta += (std::chrono::high_resolution_clock::now()-start);
  
  std::cout <<"On device Computation took "
               << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
            << " ms" << std::endl;

}


void init(int *  k, float *a, size_t size) {
  for (size_t i = 0; i < size; i++) { k[i]=i; a[i] = 0.001f*i; }
}



int main(int argc, char* argv[]) {

   const size_t array_size = 1024*1024;

  // select device
  myDevice::type dev = myDevice::gpu;
  if (argc > 1) dev = (myDevice::type)(std::stoi(argv[1]));
  MyDeviceSelector sel(dev);

  try {
    auto sdev = sel.select_device();
    if (argc == 2) {
      if (sdev.is_cpu()) {
        auto devs = sdev.create_sub_devices<cl::sycl::info::partition_property::partition_equally>(1);
        std::cout << "got " << devs.size() << " sub devices" << std::endl;
        sdev = devs[0];
      }
    }

    sycl::queue q(sdev, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<cl::sycl::info::device::name>() << "\n";
    std::cout << "max_work_group_size : "<< q.get_device().get_info<cl::sycl::info::device::max_work_group_size>() << "\n";
    std::cout << "Vector size: " << array_size << std::endl;



    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    auto *a = sycl::malloc_shared<float>(array_size, q);
    auto *k = sycl::malloc_shared<int>(array_size, q);
    auto *res_sequential = sycl::malloc_shared<float>(array_size, q);
    auto *res_parallel = sycl::malloc_shared<float>(array_size, q);
 
    init(k, a, array_size);

    // on host with host compiler
    for (size_t i = 0; i < array_size; i++) res_sequential[i] = foo(a[i]);

     auto start = std::chrono::high_resolution_clock::now();
     auto delta = start - start;
     for (int j=0; j<100; ++j) {
       delta -= (std::chrono::high_resolution_clock::now()-start);
       for (size_t i = 0; i < array_size; i++) res_sequential[i] = foo(a[i]);
       delta += (std::chrono::high_resolution_clock::now()-start);
     }
     std::cout <<"On host Computation took "
               << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
               << " ms" << std::endl;




    // on device
    wrapper(q, k, a, res_parallel, array_size);

    printf("res %a %a %a\n",a[1],res_parallel[1], res_sequential[1]);


    // Verify that the two arrays are equal.
    float maxd = 0.0;
    long long  nd=0;
    for (size_t i = 0; i < array_size; i++) {
      auto d = std::abs(res_parallel[i] - res_sequential[i]);
      maxd = std::max(maxd,d);
      if (d!=0) nd++;
    }
    std::cout <<  "diff " << maxd <<  ' ' << nd << std::endl;

    sycl::free(a, q);
    sycl::free(k, q);
    sycl::free(res_sequential, q);
    sycl::free(res_parallel, q);
  } catch (std::exception const &e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  return 0;
}
