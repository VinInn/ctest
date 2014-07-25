#include<atomic>
#include<thread>
#include<iostream>

struct GG {
  static __thread int a;
  static std::atomic<int> b;
};

__thread int GG::a=0;
std::atomic<int> GG::b(0);


namespace {
   struct GO {
   
  static bool stop;
  static void stopper() {
    GG::a++;
    std::cout << GG::a << std::endl;
    char s;
    std::cin >> s;
    stop=true;
    std::cout << GG::a << std::endl;
  }

  std::thread * s=nullptr;  
  GO() {
    s = new std::thread(stopper);
    s->detach();

// #pragma omp parallel for
      for (int i=0; i<100; ++i) {
       GG::a++; GG::b++;
      }
    }
   };

  bool GO::stop=false;

  GO go;
}

