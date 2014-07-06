#include<iostream>
#include<vector>
#include<string>
#include<boost/any.hpp>
#include<typeinfo>

namespace {

  struct Property {
    virtual ~Property(){}
    // by value???
    virtual boost::any getAny(const std::string& iname, const std::type_info & type) const=0;

    // return by value?
    template<typename T>
    T get(const std::string& iname) const {
       try {
	 return boost::any_cast<T>(getAny(iname,typeid(T)));
       }catch (const boost::bad_any_cast &) {
	 throw;
       }
    }

  };

}


struct Trivial : public Property {
  virtual boost::any getAny(const std::string& iname, const std::type_info & type) const {
    if (type==typeid(std::string)) return boost::any(getString(iname));
    if (type==typeid(int)) return boost::any(getInt(iname));
    
  }

  std::string getString(const std::string& iname) const {
    if (iname=="S") return "hello";
    else throw "notS";
  }

  int getInt(const std::string& iname) const {
    if (iname=="I") return 3;
    else throw "notI";
  }

};

int main() {

  const std:: string h = "hello";
  const boost::any a = h;
  boost::any p = &h;

  try {

    std::cout << boost::any_cast<std::string>(a) << std::endl;
  } catch (const boost::bad_any_cast &) {
    std::cout << " not a string???" << std::endl;
  }

  try {

    std::cout << boost::any_cast<const std::string>(a) << std::endl;
  } catch (const boost::bad_any_cast &) {
    std::cout << " not a string???" << std::endl;
  }

  try {

    std::cout << *boost::any_cast<std::string const *>(p) << std::endl;
  } catch (const boost::bad_any_cast &) {
    std::cout << " not a string pointer???" << std::endl;
  }

  try{
    std::cout << boost::any_cast<int>(a) << std::endl;
  } catch (const boost::bad_any_cast &) {
    
    std::cout << " not an int!!!" << std::endl;
  }

  try {
    std::cout << *boost::any_cast<std::string>(&a) << std::endl;
  } catch (const boost::bad_any_cast &) {
    
    std::cout << " not a string???" << std::endl;
  }

  Trivial pset;


  try {
   std::cout << pset.get<std::string>("S") <<  std::endl;
   std::cout << pset.get<int>("I") <<  std::endl;
   std::cout << pset.get<std::string>("I") <<  std::endl;
  } catch ( const char * c) {
    std::cout << "error " << c << std::endl;
  }



  return 0;


}
