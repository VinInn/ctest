#include <qatzip.h>

#include <iostream>
#include <cassert>
#include <cstdint>

#include<vector>
#include<thread>
#include<mutex>
#include<atomic>

#include <chrono>
using Clock = std::chrono::high_resolution_clock;
using Delta = Clock::duration;
using Time = Clock::time_point;

namespace {
  std::mutex coutLock;
  std::atomic<bool> sbar(true);
  std::atomic<int> tid(0);
  std::atomic<bool> once(true);
}

// #define USE_MALLOC

void doTest(int sw) {
   int me = tid++;
   while(sbar);

   auto start = std::chrono::high_resolution_clock::now();
   auto delta = start - start;

   QzSessionParamsLZ4_T params;

   auto status = qzGetDefaultsLZ4(&params);
   assert(status>=0);

   if (once){
    std::lock_guard<std::mutex> guard(coutLock);
    std::cout << "default lz4 params" << '\n' <<
    params.common_params.input_sz_thrshold << ' ' << QZ_COMP_THRESHOLD_DEFAULT << '\n' <<
    params.common_params.comp_lvl << '\n' <<
    params.common_params.comp_algorithm << '\n' <<
    params.common_params.hw_buff_sz << '\n' <<
    params.common_params.polling_mode << '\n' <<
    params.common_params.req_cnt_thrshold << '\n' <<
    params.common_params.max_forks << '\n' <<
    params.common_params.sw_backup << '\n' <<
    "END" << std::endl;
    once=false;
    }

   params.common_params.comp_algorithm = QZ_LZ4;  // this is '4' not int(4)
   params.common_params.sw_backup = sw;
   params.common_params.comp_lvl = 5;

    QzSession_T sess = {0};
    status = qzInit(&sess,  params.common_params.sw_backup);
    if (status<0)  std::cout << me << " qzInit failed " << status << std::endl;
    assert(status>=0);

    status = qzSetupSessionLZ4(&sess, &params);
    if (status<0)  std::cout << me << " SetupSessionLZ4 failed " << status << std::endl;
    assert(status>=0);

#ifdef USE_MALLOC
    uint32_t  orig_sz = 4 * 1024 * 1024;
    uint8_t * orig_src = (uint8_t*)malloc(orig_sz);
    uint8_t d=0;
    for (uint32_t k=0; k<orig_sz; ++k) orig_src[k]=d++; 
    auto comp_sz = orig_sz;
    uint8_t *  comp_src = (uint8_t*)malloc(comp_sz);
    auto decomp_sz = orig_sz;
    uint8_t *  decomp_src = (uint8_t*)malloc(decomp_sz);
#else
    uint32_t  orig_sz = 4 * 1024 * 1024;
    uint8_t * orig_src = (uint8_t*)qzMalloc(orig_sz, 0, COMMON_MEM); // PINNED_MEM); // COMMON_MEM);
    uint8_t d=0;
    for (uint32_t k=0; k<orig_sz; ++k) orig_src[k]=d++;
    auto comp_sz = orig_sz;
    uint8_t *  comp_src = (uint8_t*)qzMalloc(comp_sz, 0, COMMON_MEM); // PINNED_MEM); // COMMON_MEM);
    auto decomp_sz = orig_sz;
    uint8_t *  decomp_src = (uint8_t*)qzMalloc(decomp_sz, 0, COMMON_MEM); 
#endif

  {
    std::lock_guard<std::mutex> guard(coutLock);
    std::cout << me << " src " << (void*)orig_src << ' ' << (void*)comp_src << std::endl;
   }

    for (uint32_t k=0; k<orig_sz; ++k) orig_src[k]=d++;


   for (int k=0; k<1000; ++k) { 
   auto pit = k==0 || k==99;

   comp_sz = orig_sz;
   delta -= (std::chrono::high_resolution_clock::now() - start);
   auto rc = qzCompress(&sess, orig_src, &orig_sz, comp_src,
                        &comp_sz, 1);
   delta += (std::chrono::high_resolution_clock::now() - start);
   if (rc !=QZ_OK) std::cout <<  me << " qzCompress failed " << rc << std::endl;
   assert(rc == QZ_OK);

   if (pit) {
    std::lock_guard<std::mutex> guard(coutLock);
    std::cout << me << " orig size " << orig_sz << " comp size " << comp_sz << std::endl;
   }

   decomp_sz = orig_sz;
   delta -= (std::chrono::high_resolution_clock::now() - start);
   rc = qzDecompress(&sess, comp_src, &comp_sz, decomp_src,
                     &decomp_sz);
   delta += (std::chrono::high_resolution_clock::now() - start);
   if (rc !=QZ_OK) std::cout <<  me << " qzDecompress failed " << rc << std::endl;

   assert(rc == QZ_OK);

   if (pit) {
    std::lock_guard<std::mutex> guard(coutLock);
     std::cout << me << " decomp size " << decomp_sz << std::endl;
   }


   } // loop

 {
    std::lock_guard<std::mutex> guard(coutLock);
    std::cout << me << " duration " <<  std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << std::endl;
 }

    qzFree(orig_src);
    qzFree(comp_src);
    qzFree(decomp_src);
    qzTeardownSession(&sess);
    qzClose(&sess);

}


int main() {

  sbar = false;
  std::cout << "running test once with sw-bk" << std::endl;
  doTest(1);

  std::cout << "\n\n\n" << std::endl;
  std::cout << "running test once with NO sw-bk" << std::endl;
  doTest(0);

  std::cout << "\n\n\n" << std::endl;

  int nt = 8;

{
  std::cout << "running test in " << nt <<" threads with sw-bk" << std::endl;
  sbar = true;
  tid = 0;
  std::vector<std::thread> ts;
  for (int i=0; i<nt; ++i) ts.emplace_back(doTest,1);

  sbar = false;
  for (auto & t : ts) t.join();
}

std::cout << "\n\n\n" << std::endl;
{
  nt = 4;
  std::cout << "running test in " << nt <<" threads with NO sw-bk" << std::endl;
  sbar = true;
  tid = 0;
  std::vector<std::thread> ts;
  for (int i=0; i<nt; ++i) ts.emplace_back(doTest,0);

  sbar = false;
  for (auto & t : ts) t.join();
}


std::cout << "\n\n\n" << std::endl;
{
  nt = 8;
  std::cout << "running test in " << nt <<" threads with NO sw-bk" << std::endl;
  sbar = true;
  tid = 0;
  std::vector<std::thread> ts;
  for (int i=0; i<nt; ++i) ts.emplace_back(doTest,0);

  sbar = false;
  for (auto & t : ts) t.join();
}

  return 0;
}

