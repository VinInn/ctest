#include <qatzip.h>
#include "qatseqprod.h"
#include <zstd.h>

#include <iostream>
#include <cassert>
#include <cstdint>

#include<random>
#include<vector>
#include<thread>
#include<mutex>
#include<atomic>

#include <chrono>
using Clock = std::chrono::steady_clock;
using Delta = Clock::duration;
using Time = Clock::time_point;

namespace {
  std::mutex coutLock;
  std::atomic<bool> sbar(true);
  std::atomic<int> tid(0);
  std::atomic<bool> once(true);
}


void doTest(int sw) {
   int me = tid++;
   while(sbar);

   auto start = std::chrono::steady_clock::now();
   auto delta = start - start;

    using Ctx_ptr = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>;
    Ctx_ptr fCtx{ZSTD_createCCtx(), &ZSTD_freeCCtx};
    using Dtx_ptr = std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>;
    Dtx_ptr fDtx{ZSTD_createDCtx(), &ZSTD_freeDCtx};
#ifdef USE_PLUGIN
    /* Create sequence producer state for QAT sequence producer */
    void *sequenceProducerState = QZSTD_createSeqProdState();
    /* register qatSequenceProducer */
    ZSTD_registerSequenceProducer(
        fCtx.get(),
        sequenceProducerState,
        qatSequenceProducer
    );
    /* Enable sequence producer fallback */
    if (sw) ZSTD_CCtx_setParameter(fCtx.get(), ZSTD_c_enableSeqProducerFallback, 1);
#else
    ZSTD_CCtx_setParameter(fCtx.get(),ZSTD_c_compressionLevel,1);
#endif
    uint32_t  orig_sz = 4 * 1024 * 1024;
    uint8_t * orig_src = (uint8_t*)malloc(orig_sz);
    auto comp_sz = 4*ZSTD_compressBound(orig_sz);
    uint8_t *  comp_src = (uint8_t*)malloc(comp_sz);
    auto decomp_sz = orig_sz;
    uint8_t *  decomp_src = (uint8_t*)malloc(decomp_sz);
    
    std::mt19937_64 gen;
    uint64_t * src64 = (uint64_t *)orig_src;
    for (uint32_t k=0; k<orig_sz/sizeof(uint64_t); ++k) src64[k] =k;
    /*;
    uint8_t d=0;
    for (uint32_t k=0; k<orig_sz; ++k) 
       // orig_src[k]=d++;
     {  orig_src[k]= d; d+=7; if (0==(k%16))  d+=131;}
   */

#ifdef VERBOSE
  {
    std::lock_guard<std::mutex> guard(coutLock);
    std::cout << me << " src " << (void*)orig_src << ' ' << (void*)comp_src << std::endl;
   }
#endif


   for (int k=0; k<1000; ++k) { 
#ifdef VERBOSE
   auto pit = k==99;
#endif

   comp_sz = orig_sz;

   delta -= (std::chrono::steady_clock::now() - start);
   auto rc = ZSTD_compress2(fCtx.get(),
                               comp_src,comp_sz,
                               orig_src, orig_sz);
   delta += (std::chrono::steady_clock::now() - start);
     if (ZSTD_isError(rc)) {
            std::cerr << "Error in zip ZSTD. Type = " << ZSTD_getErrorName(rc) <<
            " . Code = " << rc << std::endl;
     } else comp_sz =  rc;

#ifdef VERBOSE
   if (pit) {
#else
   if (once) {
#endif
    once = false;
    std::lock_guard<std::mutex> guard(coutLock);
    std::cout << me << " orig size " << orig_sz << " comp size " << comp_sz << std::endl;
   }

   decomp_sz = orig_sz;

#ifndef COMPRESS_ONLY
   delta -= (std::chrono::steady_clock::now() - start);
#endif
   rc = ZSTD_decompressDCtx(fDtx.get(),
                                 decomp_src, decomp_sz,
                                 comp_src, comp_sz);
#ifndef COMPRESS_ONLY
   delta += (std::chrono::steady_clock::now() - start);
#endif
   if (ZSTD_isError(rc)) {
            std::cerr << "Error in zip ZSTD. Type = " << ZSTD_getErrorName(rc) <<
            " . Code = " << rc << std::endl;
   } else decomp_sz =  rc;

#ifdef VERBOSE
   if (pit) {
    std::lock_guard<std::mutex> guard(coutLock);
     std::cout << me << " decomp size " << decomp_sz << std::endl;
   }
#endif

   } // loop

 {
    std::lock_guard<std::mutex> guard(coutLock);
    std::cout << me << " duration " <<  std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << std::endl;
 }

    free(orig_src);
    free(comp_src);
    free(decomp_src);
#ifdef USE_PLUGIN
    /* Free sequence producer state */
    QZSTD_freeSeqProdState(sequenceProducerState);
#endif
}


int main() {

#ifdef USE_PLUGIN
  std::cout << "using qaz plugin" << std::endl;
   /* Start QAT device, start QAT device at any
    time before compression job started */
    QZSTD_startQatDevice();
#endif
#ifdef COMPRESS_ONLY
  std::cout << "timing conpression only" << std::endl;
#endif

  sbar = false;
  std::cout << "running test once with sw-bk" << std::endl;
  doTest(1);

  std::cout << "\n\n\n" << std::endl;
  std::cout << "running test once with NO sw-bk" << std::endl;
  doTest(0);


#ifndef ONLY8
int nTH[] = {4,8,16,32,54,112,224};
#else
int nTH[] = {8,8,8,8,8,8,8};
#endif

std::cout << "\n\n\n" << std::endl;
for (auto nt : nTH) {
{
  std::cout << "running test in " << nt <<" threads with sw-bk" << std::endl;
  sbar = true;
  tid = 0;
  std::vector<std::thread> ts;
  for (int i=0; i<nt; ++i) ts.emplace_back(doTest,1);

  auto start = std::chrono::steady_clock::now();
  sbar = false;
  for (auto & t : ts) t.join();
  auto delta =  std::chrono::steady_clock::now() -start;
  std::cout << "total  duration " <<  std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << std::endl;
}
}  // loop on nTH

std::cout << "\n\n\n" << std::endl;
for (auto nt : nTH) {
{
  std::cout << "running test in " << nt <<" threads with NO sw-bk" << std::endl;
  sbar = true;
  tid = 0;
  std::vector<std::thread> ts;
  for (int i=0; i<nt; ++i) ts.emplace_back(doTest,0);

  auto start = std::chrono::steady_clock::now();
  sbar = false;
  for (auto & t : ts) t.join();
  auto delta =  std::chrono::steady_clock::now() -start;
  std::cout << "total  duration " <<  std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() << std::endl;

}

}  // loop on nTH

#ifdef USE_PLUGIN
    /* Please call QZSTD_stopQatDevice before
    QAT is no longer used or the process exits */
    QZSTD_stopQatDevice();
#endif
  return 0;
}

