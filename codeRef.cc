class MyRef0;
class MyRef1;

#include <typeinfo>
#include <sstream>

struct CodeRef{

enum mask {  maskMyRef0=0,  maskMyRef1=(1<<28)   };

template<typename REF>
static unsigned int  code(REF const& ref) { std::stringstream ss; ss << typeid(REF).name() << " no supported"; throw ss.str().c_str();}  

};

template<>
unsigned int  CodeRef::code<MyRef0>(MyRef0 const& ref) {  return ref.index() | maskMyRef0;}

template<>
unsigned int  CodeRef::code<MyRef1>(MyRef1 const& ref) {  return ref.index() | maskMyRef1;}

