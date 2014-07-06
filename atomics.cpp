#include <cstdlib>
#include<iostream>

#include<vector>

#include <thread>
#include <functional>
#include <algorithm>
#include<cmath>
#include<cstdatomic>

#include<iostream>

// convert gcc to c0xx
#define thread_local __thread;

typedef std::thread Thread;
typedef std::vector<std::thread> ThreadGroup;
typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> Guard;
typedef std::condition_variable Condition;





int main() {


  {

  volatile int x=11;

  int y = __sync_lock_test_and_set(&x,-3);

  std::cout << x <<", " << y << std::endl;

  y = __sync_add_and_fetch(&x,2);
  std::cout << x <<", " << y << std::endl;

  y = __sync_fetch_and_add(&x,2);
  std::cout << x <<", " << y << std::endl;

  int z = x;
  y = __sync_val_compare_and_swap(&x,z,z+1);

  std::cout << x <<", " << y <<", " << z << std::endl;

  z=x;
  std::cout <<  __sync_val_compare_and_swap(&x,z,z+1) << std::endl;


  z = x;
  x+=2;
  y = __sync_val_compare_and_swap(&x,z,z+1);

  std::cout <<  __sync_val_compare_and_swap(&x,&x,z,z+1) << std::endl;
  std::cout << x <<", " << y <<", " << z << std::endl;

  }

 {


   typedef std::atomic<int> Aint;
   
   Aint x(11);
   std::cout << "c++ atomics " << x.is_lock_free() << std::endl;
   
   //int y=0;
   //int y = x;
   int y = x; //.fetch_add(0); // bha
   x = -5;
   std::cout << x <<", " << y << std::endl;
   
   // y = x.load();
   // std::cout << x <<", " << y << std::endl;
   
   x = -3;
   y = x; //.fetch_add(0);
   // x.exchange(-3);
   std::cout << x <<", " << y << std::endl;
   
   y = x;
   x+=2;
   std::cout << x <<", " << y << std::endl;
   y =  x.fetch_add(2);
   std::cout << x <<", " << y << std::endl;
   
   int z = x;
   std::cout <<  x.compare_exchange_strong(z,z+1) << std::endl;
   std::cout << x <<", " << z << std::endl;
   
   
   y = z = x;
   x+=2;
   
   std::cout <<  x.compare_exchange_strong(z,z+1) << std::endl;
   std::cout << x <<", " << y <<", " << z << std::endl;
   
 }




  return 0;

}
