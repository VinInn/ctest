#include<map>
#include<string>
#include<boost/shared_ptr.hpp>
#include<boost/any.hpp>
#include<iostream>


struct Base {
  virtual ~Base(){}
  void hi() {
     std::cout << "Base" << std::endl;
  }
};

struct Foo : public Base {
  Foo() {
    std::cout << "c  Foo" << std::endl;
  }
  Foo(Foo const &) {
    std::cout << "cc Foo" << std::endl;
  }
  Foo & operator=(Foo const &) {
    std::cout << "=  Foo" << std::endl;
    return *this;
  }
  ~Foo() {
    std::cout << "d  Foo" << std::endl;
  }

};



int main() {

  {
    std::cout << "a" << std::endl;
    boost::any a = Foo();
    std::cout << "b" << std::endl;
    boost::any b = a;
    std::cout << "c" << std::endl;
    boost::any c; c.swap(a);
    std::cout << "b2" << std::endl;
    b = Foo();
    std::cout << "e" << std::endl;
    Base * base = boost::any_cast<Base>(&b);
    if (!base)  std::cout << "no Base...." << std::endl;
    base = boost::unsafe_any_cast<Base>(&b);
    if (!base)  std::cout << "no Base???" << std::endl;
  }

  typedef std::map<int, boost::any> Map;	

  Map m;
  {
    boost::any & f = m[1];
    
    if (f.empty()) 
      f = boost::shared_ptr<double>(new double(3.3));
    if (f.empty()) std::cout << "error" << std::endl;
    std::cout <<  boost::any_cast<boost::shared_ptr<double> >(f).use_count()
	      << std::endl;
  }
  {
    boost::any f = m[1];
    if (f.empty()) std::cout << "error" << std::endl;
    std::cout <<  boost::any_cast<boost::shared_ptr<double> >(&f)->use_count()
	      << std::endl;
    boost::shared_ptr<double> d = *boost::any_cast<boost::shared_ptr<double> >(&f);
    std::cout << *d << std::endl;
  }
  {
    boost::any f = m[1];
    if (f.empty()) std::cout << "error" << std::endl;
    boost::shared_ptr<double> d = *boost::any_cast<boost::shared_ptr<double> >(&f);
    *d = 4.4;
    std::cout << *d << std::endl;
    std::cout <<  boost::any_cast<boost::shared_ptr<double> >(&f)->use_count()
	      << std::endl;
  }
  {
    boost::any f = m[1];
    if (f.empty()) std::cout << "error" << std::endl;
    std::cout <<  boost::any_cast<boost::shared_ptr<double> >(&f)->use_count()
	      << std::endl;
    boost::shared_ptr<double> d = *boost::any_cast<boost::shared_ptr<double> >(&f);
    std::cout << *d << std::endl;
  }
  {
    boost::any & f = m[1];
    if (f.empty()) std::cout << "error" << std::endl;
    boost::shared_ptr<double> d = *boost::any_cast<boost::shared_ptr<double> >(&f);
    std::cout << *d << std::endl;
    std::cout <<  boost::any_cast<boost::shared_ptr<double> >(&f)->use_count()
	      << std::endl;
    f = boost::shared_ptr<std::string>(new std::string("hello"));
    std::cout <<  d.use_count()
	      << std::endl;

  }
  return 0;

}
