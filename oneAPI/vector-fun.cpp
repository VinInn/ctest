//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of DPC++. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// DPC++ material used in the code sample:
// •	A one dimensional array of data shared between CPU and offload device.
// •	A device queue and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// #include "oneapi/tbb.h"

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <cmath>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#include "device_selector.hpp"

#include "cr_log.h"


using namespace sycl;

// Array size for this example.
size_t array_size = 1024*1024;

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

//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum".
//************************************
void VectorAdd(queue &q, const int * k, const float *a, float *sum, size_t size) {

  std::cout << "in the kernel wrapper" << std::endl;

  // Create the range object for the arrays.
  range<1> num_items{size};

  // Use parallel_for to run vector addition in parallel on device. This
  // executes the kernel.
  //    1st parameter is the number of work items.
  //    2nd parameter is the kernel, a lambda that specifies what to do per
  //    work item. the parameter of the lambda is the work item id.
  // DPC++ supports unnamed lambda kernel by default.
  auto e = q.parallel_for(num_items, [=](auto i) {
        if (0==i) {
#ifdef __SSE__
#warning compile for cpu???
           printf("in kernel on CPU\n");
          // std::cout << "in kernel on CPU " << i << std::endl;
#else
#warning compile for cpu???
           printf("in kernel on GPU\n");
          //std::cout << "in kernel on CPU " << i << std::endl;
#endif
        } 
        sum[i] = as_logf(a[k[i]]); 

        });

  // q.parallel_for() is an asynchronous call. DPC++ runtime enqueues and runs
  // the kernel asynchronously. Wait for the asynchronous call to complete.
  e.wait();
}

//************************************
// Initialize the array from 0 to array_size - 1
//************************************
void InitializeArray(int *  k, float *a, size_t size) {
  for (size_t i = 0; i < size; i++) { k[i]=i; a[i] = 0.001f*i; if (i==1) a[k[1]] = 0x1.01825ca7da7e5p+0;}
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {

  // oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);

  // select device
  myDevice::type dev = myDevice::gpu;
  if (argc > 1) dev = (myDevice::type)(std::stoi(argv[1]));
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

    MyDeviceSelector sel(dev);

#ifdef __INTEL_COMPILER
  std::cout << "intel compiler defined" << std::endl;
#endif

  try {
    auto sdev = sel.select_device();
    if (sdev.is_cpu()) {
      auto devs = sdev.create_sub_devices<info::partition_property::partition_equally>(1);
      std::cout << "got " << devs.size() << " sub devices" << std::endl;
      sdev = devs[0];
    }
    queue q(sdev, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size : "<< q.get_device().get_info<cl::sycl::info::device::max_work_group_size>() << "\n";
    std::cout << "Vector size: " << array_size << std::endl;



    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    auto *a = malloc_shared<float>(array_size, q);
    auto *k = malloc_shared<int>(array_size, q);
    auto *sum_sequential = malloc_shared<float>(array_size, q);
    auto *sum_parallel = malloc_shared<float>(array_size, q);

    if ((a == nullptr) || (k == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel == nullptr)) {
      if (a != nullptr) free(a, q);
      if (k != nullptr) free(k, q);
      if (sum_sequential != nullptr) free(sum_sequential, q);
      if (sum_parallel != nullptr) free(sum_parallel, q);

      std::cout << "Shared memory allocation failure.\n";
      return -1;
    }

    // Initialize input arrays with values from 0 to array_size - 1
    InitializeArray(k, a, array_size);

    for (int j=0; j<100; ++j) {
    // Compute the sum of two arrays in sequential for validation.
    for (size_t i = 0; i < array_size; i++) sum_sequential[i] = sin(a[k[i]]);

    // Vector addition in DPC++.
    VectorAdd(q, k, a, sum_parallel, array_size);

    if (0==j%100) printf("res %a %a %a\n",a[1],sum_parallel[1], sum_sequential[1]);

    }
    
    
    // Verify that the two arrays are equal.
    float maxd = 0.0;
    long long  nd=0;
    for (size_t i = 0; i < array_size; i++) {
      auto d = std::abs(sum_parallel[i] - sum_sequential[i]);
      maxd = std::max(maxd,d);
      if (d!=0) nd++;
    }
   

    std::cout <<  "diff " << maxd <<  ' ' << nd << std::endl;
 
    

    free(a, q);
    free(k, q);
    free(sum_sequential, q);
    free(sum_parallel, q);
  } catch (exception const &e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
