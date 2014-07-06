#include<vector>
struct bar {
   static void * foo();
   virtual void * getFoo() const { return bar::foo();}
};

