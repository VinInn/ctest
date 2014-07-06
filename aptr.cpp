#include <memory> 

struct A {}; 
struct B : A {}; 

int main(int argc, char* argv[]) 
{ 
    // direct initialization 
    std::auto_ptr<A> a0(std::auto_ptr<A>(0)); // okay 
    std::auto_ptr<A> b0(std::auto_ptr<B>(0)); // okay 
    // copy initialization 
    std::auto_ptr<A> a1 = std::auto_ptr<A>(0); // okay 
    std::auto_ptr<A> b1 = std::auto_ptr<B>(0); // error 

} 
