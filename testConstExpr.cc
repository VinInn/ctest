#include <string>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include "constHash64.h"

constexpr size_t SAMPLE_SIZE = 100000;
constexpr size_t REPETITIONS = 1000;
std::string testData[SAMPLE_SIZE];

static std::unordered_map<std::string, int> dmTranslator =
  {
          {"oneProng0Pi0", 0},
          {"oneProng1Pi0", 1},
          {"oneProng2Pi0", 2},
          {"oneProngOther", 3},
          {"threeProng0Pi0", 4},
          {"threeProng1Pi0", 5},
          {"threeProngOther", 6},
          {"electron", 7},
          {"muon", 8}
  };

int translateGenDecayModeToReco(const std::string &name) {
  auto found = dmTranslator.find(name);
  if (found != dmTranslator.end()) {
           return found->second;
  } else
  return -1;
}

struct entry {
  char const*label;
  int mode;
};

constexpr entry dmTranslatorMap[] = {
  {"oneProng0Pi0", 0},
  {"oneProng1Pi0", 1},
  {"oneProng2Pi0", 2},
  {"oneProngOther", 3},
  {"threeProng0Pi0", 4},
  {"threeProng1Pi0", 5},
  {"threeProngOther", 6},
  {"electron", 7},
  {"muon", 8},
  {nullptr, -1}
};

constexpr bool same(char const *x, char const *y) {
  return !*x && !*y     ? true 
         /* default */  : (*x == *y && same(x+1, y+1));
}
  
constexpr int decayModeStringToId(char const *label, entry const *entries) {
  return !entries->label ? -1                         
       : same(entries->label, label) ? entries->mode 
         /*default*/                 : decayModeStringToId(label, entries+1);
}

unsigned int constexpr hash(char const *input, size_t len) { 
  return hash64::hash64(input, len, 0);
    // really simple hash function...
  //    return !*input      ? 0 
  //     : /*default*/    static_cast<unsigned int>(*input) + hash(input+1);
}

struct hash_entry {
  size_t hash;
  int mode;
};

constexpr hash_entry dmTranslatorHashMap[] = {
  {hash("oneProng0Pi0",::strlen("oneProng0Pi0")), 0},
  {hash("oneProng1Pi0",::strlen("oneProng1Pi0")), 1},
  {hash("oneProng2Pi0",::strlen("oneProng2Pi0")), 2},
  {hash("oneProngOther",::strlen("oneProngOther")), 3},
  {hash("threeProng0Pi0",::strlen("threeProng0Pi0")), 4},
  {hash("threeProng1Pi0",::strlen("threeProng1Pi0")), 5},
  {hash("threeProngOther",::strlen("threeProngOther")), 6},
  {hash("electron",::strlen("electron")), 7},
  {hash("muon",::strlen("muon")), 8},
  {0, -1}
};

constexpr int decayModeHashToId(unsigned int hash, hash_entry const *entries) {
  return entries->hash == 0   ? -1
       : entries->hash == hash ? entries->mode
       : /*default*/             decayModeHashToId(hash, entries+1);
}

int oldResults[SAMPLE_SIZE];
void oldLookup_()
{
  for (size_t i = 0; i < SAMPLE_SIZE*REPETITIONS;  ++i)
    oldResults[i%SAMPLE_SIZE] = translateGenDecayModeToReco(testData[i%SAMPLE_SIZE]);
}

int newResults[SAMPLE_SIZE];
void newLookup_()
{
  for (size_t i = 0; i < SAMPLE_SIZE*REPETITIONS;  ++i)
    newResults[i%SAMPLE_SIZE] = decayModeStringToId(testData[i%SAMPLE_SIZE].c_str(), dmTranslatorMap);
}

int hashResults[SAMPLE_SIZE];
void hashLookup_()
{
  for (size_t i = 0; i < SAMPLE_SIZE*REPETITIONS;  ++i)
    hashResults[i%SAMPLE_SIZE] = decayModeHashToId(hash(testData[i%SAMPLE_SIZE].c_str(),testData[i%SAMPLE_SIZE].size()), dmTranslatorHashMap);
}

int oldResultsParallel[SAMPLE_SIZE];
void oldLookupParallel_()
{
  #pragma omp parallel for
  for (size_t i = 0; i < SAMPLE_SIZE*REPETITIONS;  ++i)
    oldResultsParallel[i%SAMPLE_SIZE] = translateGenDecayModeToReco(testData[i%SAMPLE_SIZE]);
}

int newResultsParallel[SAMPLE_SIZE];
void newLookupParallel_()
{
  #pragma omp parallel for
  for (size_t i = 0; i < SAMPLE_SIZE*REPETITIONS;  ++i)
    newResultsParallel[i%SAMPLE_SIZE] = decayModeStringToId(testData[i%SAMPLE_SIZE].c_str(), dmTranslatorMap);
}

int hashResultsParallel[SAMPLE_SIZE];
void hashLookupParallel_()
{
  #pragma omp parallel for
  for (size_t i = 0; i < SAMPLE_SIZE*REPETITIONS;  ++i)
  {
    hashResultsParallel[i%SAMPLE_SIZE] = decayModeHashToId(hash(testData[i%SAMPLE_SIZE].c_str(),testData[i%SAMPLE_SIZE].size()), dmTranslatorHashMap);
  }
}

#include <cassert>
#include <iostream>
std::string sampleData[] = {"electron", "oneProng0Pi0", 
                            "oneProng1Pi0", "oneProng2Pi0", "oneProngOther",
                            "threeProng0Pi0", "threeProng1Pi0", "threeProngOther", 
                            "electron", "nonexisting"};

void (*newLookup)() = newLookup_;
void (*oldLookup)() = oldLookup_;
void (*hashLookup)() = hashLookup_;
void (*newLookupParallel)() = newLookupParallel_;
void (*oldLookupParallel)() = oldLookupParallel_;
void (*hashLookupParallel)() = hashLookupParallel_;


#define RDTSC(v)                                                        \
  do { unsigned lo, hi;                                                 \
    __asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));                   \
    (v) = ((uint64_t) lo) | ((uint64_t) hi << 32);                      \
  } while (0)

int main (int argc, char **argv)
{
  /* Prepare some data. Poor-man random generator. */ 
  size_t i = 0;
  for (std::string &s : testData)
  {
    i = (i + 1234567891) % (sizeof(sampleData)/sizeof(void*));
    s = sampleData[i];
  }
  #pragma omp parallel
    printf("Hello, world.\n");

  uint64_t start, stop;
  RDTSC(stop);
  RDTSC(start);
  oldLookup();
  RDTSC(stop);
  std::cerr << "Old          :" << stop-start << std::endl;
  RDTSC(start);
  newLookup();
  RDTSC(stop);
  std::cerr << "New          :" << stop-start << std::endl;
  RDTSC(start);
  oldLookupParallel();
  RDTSC(stop);
  std::cerr << "Old parallel :" << stop-start << std::endl;
  RDTSC(start);
  newLookupParallel();
  RDTSC(stop);
  std::cerr << "New parallel :" << stop-start << std::endl;
  RDTSC(start);
  hashLookup();
  RDTSC(stop);
  std::cerr << "Hash         :" << stop-start << std::endl;
  RDTSC(start);
  hashLookupParallel();
  RDTSC(stop);
  std::cerr << "Hash parallel:" << stop-start << std::endl;
  
  for (hash_entry x : dmTranslatorHashMap)
  {
    std::cerr << x.hash << " " << x.mode << std::endl;
  }
  
  for (size_t i = 0; i < SAMPLE_SIZE; ++i)
  {
    assert(oldResults[i] == newResults[i]);
    assert(oldResultsParallel[i] == newResultsParallel[i]);
    assert(newResults[i] == newResultsParallel[i]);
    assert(hashResults[i] == newResults[i]);
    assert(hashResultsParallel[i] == hashResults[i]);
  }
}
