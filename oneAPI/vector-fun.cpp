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
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
#include <cmath>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#include "device_selector.hpp"


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
void VectorAdd(queue &q, const float *a, float *sum, size_t size) {
  // Create the range object for the arrays.
  range<1> num_items{size};

  // Use parallel_for to run vector addition in parallel on device. This
  // executes the kernel.
  //    1st parameter is the number of work items.
  //    2nd parameter is the kernel, a lambda that specifies what to do per
  //    work item. the parameter of the lambda is the work item id.
  // DPC++ supports unnamed lambda kernel by default.
  auto e = q.parallel_for(num_items, [=](auto i) { sum[i] = cos(a[i]); });

  // q.parallel_for() is an asynchronous call. DPC++ runtime enqueues and runs
  // the kernel asynchronously. Wait for the asynchronous call to complete.
  e.wait();
}

//************************************
// Initialize the array from 0 to array_size - 1
//************************************
void InitializeArray(float *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = 0.001f*i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change array_size if it was passed as argument
  if (argc > 1) array_size = std::stoi(argv[1]);
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

    MyDeviceSelector sel;

#ifdef __INTEL_COMPILER
  std::cout << "intel compiler defined" << std::endl;
#endif

  try {
    queue q(sel, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << array_size << "\n";

    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    float *a = malloc_shared<float>(array_size, q);
    float *b = malloc_shared<float>(array_size, q);
    float *sum_sequential = malloc_shared<float>(array_size, q);
    float *sum_parallel = malloc_shared<float>(array_size, q);

    if ((a == nullptr) || (b == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel == nullptr)) {
      if (a != nullptr) free(a, q);
      if (b != nullptr) free(b, q);
      if (sum_sequential != nullptr) free(sum_sequential, q);
      if (sum_parallel != nullptr) free(sum_parallel, q);

      std::cout << "Shared memory allocation failure.\n";
      return -1;
    }

    // Initialize input arrays with values from 0 to array_size - 1
    InitializeArray(a, array_size);
    InitializeArray(b, array_size);

    for (int k=0; k<100; ++k) {
    // Compute the sum of two arrays in sequential for validation.
    for (size_t i = 0; i < array_size; i++) sum_sequential[i] = sin(a[i]);

    // Vector addition in DPC++.
    VectorAdd(q, a, sum_parallel, array_size);

    printf("res %a %a %a\n",a[1],sum_parallel[1], sum_sequential[1]);

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
    free(b, q);
    free(sum_sequential, q);
    free(sum_parallel, q);
  } catch (exception const &e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
