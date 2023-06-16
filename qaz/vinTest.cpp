#include <qatzip.h>
#include "qatseqprod.h"
#include <zstd.h>

#include <iostream>
#include <cassert>
#include <cstdint>

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

#ifdef USE_ZSTD
#define USE_MALLOC
#endif

void doTest(int sw) {
   int me = tid++;
   while(sbar);

   auto start = std::chrono::steady_clock::now();
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
    }

   params.common_params.comp_algorithm = QZ_LZ4;  // this is '4' not int(4)
   params.common_params.sw_backup = sw;
   params.common_params.comp_lvl = 8;

#ifdef USE_ZSTD
    using Ctx_ptr = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>;
    Ctx_ptr fCtx{ZSTD_createCCtx(), &ZSTD_freeCCtx};
    ZSTD_CCtx_setParameter(fCtx.get(),ZSTD_c_compressionLevel,1);
    using Dtx_ptr = std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>;
    Dtx_ptr fDtx{ZSTD_createDCtx(), &ZSTD_freeDCtx};
#ifdef USE_PUGIN
   /* Start QAT device, start QAT device at any
    time before compression job started */
    QZSTD_startQatDevice();
    /* Create sequence producer state for QAT sequence producer */
    void *sequenceProducerState = QZSTD_createSeqProdState();
    /* register qatSequenceProducer */
    ZSTD_registerSequenceProducer(
        fCtx.get(),
        sequenceProducerState,
        qatSequenceProducer
    );
    /* Enable sequence producer fallback */
    ZSTD_CCtx_setParameter(fCtx.get(), ZSTD_c_enableSeqProducerFallback, 1);
#endif
#else
    QzSession_T sess = {0};
    status = qzInit(&sess,  params.common_params.sw_backup);
    if (status<0)  std::cout << me << " qzInit failed " << status << std::endl;
    assert(status>=0);

    status = qzSetupSessionLZ4(&sess, &params);
    if (status<0)  std::cout << me << " SetupSessionLZ4 failed " << status << std::endl;
    assert(status>=0);
#endif

#ifdef USE_MALLOC
    uint32_t  orig_sz = 4 * 1024 * 1024;
    uint8_t * orig_src = (uint8_t*)malloc(orig_sz);
    auto comp_sz = orig_sz;
    uint8_t *  comp_src = (uint8_t*)malloc(comp_sz);
    auto decomp_sz = orig_sz;
    uint8_t *  decomp_src = (uint8_t*)malloc(decomp_sz);
#else
    uint32_t  orig_sz = 4 * 1024 * 1024;
    uint8_t * orig_src = (uint8_t*)qzMalloc(orig_sz, 0, COMMON_MEM); // PINNED_MEM); // COMMON_MEM);
    auto comp_sz = orig_sz;
    uint8_t *  comp_src = (uint8_t*)qzMalloc(comp_sz, 0, COMMON_MEM); // PINNED_MEM); // COMMON_MEM);
    auto decomp_sz = orig_sz;
    uint8_t *  decomp_src = (uint8_t*)qzMalloc(decomp_sz, 0, COMMON_MEM); 
#endif

    uint8_t d=0;
    for (uint32_t k=0; k<orig_sz; ++k) 
       // orig_src[k]=d++;
     {  orig_src[k]= d; d+=7; if (0==(k%16))  d+=131;}


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

#ifdef USE_ZSTD
   delta -= (std::chrono::steady_clock::now() - start);
   auto rc = ZSTD_compress2(fCtx.get(),
                               comp_src,comp_sz,
                               orig_src, orig_sz);
   delta += (std::chrono::steady_clock::now() - start);
     if (ZSTD_isError(rc)) {
            std::cerr << "Error in zip ZSTD. Type = " << ZSTD_getErrorName(rc) <<
            " . Code = " << rc << std::endl;
     } else comp_sz =  rc;
#else
   delta -= (std::chrono::steady_clock::now() - start);
   auto rc = qzCompress(&sess, orig_src, &orig_sz, comp_src,
                        &comp_sz, 1);
   delta += (std::chrono::steady_clock::now() - start);
   if (rc !=QZ_OK) std::cout <<  me << " qzCompress failed " << rc << std::endl;
   assert(rc == QZ_OK);
#endif


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

#ifdef USE_ZSTD
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
#else
#ifndef COMPRESS_ONLY
   delta -= (std::chrono::steady_clock::now() - start);
#endif
   rc = qzDecompress(&sess, comp_src, &comp_sz, decomp_src,
                     &decomp_sz);
#ifndef COMPRESS_ONLY
   delta += (std::chrono::steady_clock::now() - start);
#endif
   if (rc !=QZ_OK) std::cout <<  me << " qzDecompress failed " << rc << std::endl;
   assert(rc == QZ_OK);
#endif

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

#ifndef USE_ZSTD
    qzFree(orig_src);
    qzFree(comp_src);
    qzFree(decomp_src);
    qzTeardownSession(&sess);
    qzClose(&sess);
#else
    free(orig_src);
    free(comp_src);
    free(decomp_src);
#ifdef USE_PUGIN
    /* Free sequence producer state */
    QZSTD_freeSeqProdState(sequenceProducerState);
    /* Please call QZSTD_stopQatDevice before
    QAT is no longer used or the process exits */
    QZSTD_stopQatDevice();
#endif
#endif
}


int main() {

#ifdef USE_ZSTD
  std::cout << "using zstd" << std::endl;
#endif
#ifdef USE_PLUGIN
  std::cout << "using qaz plugin" << std::endl;
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

  return 0;
}

