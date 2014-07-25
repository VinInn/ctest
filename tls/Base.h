#include<atomic>
struct Base {
  public:
   void here(Base * ime) { me = ime;}  
   virtual ~Base(){}
   virtual int go(int i) const =0;

   virtual int hi() const =0;

   static Base * me;
   static std::atomic<int> a;

};
