#include<mutex>
#include<iostream>


int a=0;
void bar() {
    std::once_flag first;
    std::call_once(first, [](){a++;});
}


int & foo() {
   static int l=4;
   return l;

}
