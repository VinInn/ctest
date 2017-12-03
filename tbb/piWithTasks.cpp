#include <iostream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include"tbb/tick_count.h"
#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include "cstdlib"


int main(int argc, char * argv[])
{

  auto numThreads = (argc>1) ? ::atoi(argv[1]) : tbb::task_scheduler_init::default_num_threads();
  auto numTasks = (argc>2) ? ::atoi(argv[2]) : numThreads;


  const int num_steps = 100000;

  double pi = 0.0;
  double step = 1.0/(double) num_steps;

  std::cerr << "default num of thread " << tbb::task_scheduler_init::default_num_threads() << std::endl;
  std::cerr << "actual num of thread " << numThreads << std::endl;


  //tbb::task_scheduler_init init;  // Automatic number of threads
   tbb::task_scheduler_init init(numThreads);  // Explicit number of threads

   tbb::task_group g;


  auto start = tbb::tick_count::now();

  std::cerr << "actual num of tasks " << numTasks << std::endl;



  std::vector<double> partialSums(numTasks,0.0);
  for (int id = 0; id < numTasks; id++) {
      g.run([=, &partialSums]()
      {
        double x;
        double sum = 0.;
        for (int i=id;i< num_steps; i=i+numThreads) {
          x = (i+0.5)*step;
          sum += 4.0/(1.0+x*x);
        }
        partialSums[id] = sum;
      });
  }

  g.wait();
  for (int id = 0; id < numThreads; id++) {
    pi += partialSums[id] * step;
  }

  auto stop = tbb::tick_count::now();
  std::cout << "result: " <<  std::setprecision (15) << pi << std::endl;
  std::cout << "time: " << (stop-start).seconds() << " seconds" << std::endl;

  return 0;

}

