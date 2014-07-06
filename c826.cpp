#include<iostream>
#include<string>
#include<vector>
#include <iomanip>
#include <cstdlib>
#include <iterator>

namespace C826 {

 struct C32 {
#ifdef _powerpc__
    unsigned char w4 :2;
#endif
    unsigned char c0 :6;    
    unsigned char w1 :2;    
    unsigned char c1 :6;   
    unsigned char w2 :2;    
    unsigned char c2 :6;    
    unsigned char w3 :2;    
    unsigned char c3 :6;
#ifdef __linux__
    unsigned char w4 :2;
#endif    
  };

  union FourBytes {
    C32 c32;
    char c[4];
    unsigned int i;
  };

  struct C24 {
    unsigned int c0 :6;    
    unsigned int c1 :6;   
    unsigned int c2 :6;    
    unsigned int c3 :6;    
  };

 
  union ThreeBytes {
    C24 c24;
    char c[3];
  };

  const std::string letters("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"); 

 
  std::string encode(const std::vector<unsigned char> & s) {
    std::string res; res.reserve(4*s.size()/3);
    for (int i=0; i<int(s.size());i+=3) {
      ThreeBytes const & t = (ThreeBytes const &)(s[i]);
      res+=letters[t.c24.c0];
      res+=letters[t.c24.c1];
      res+=letters[t.c24.c2];
      res+=letters[t.c24.c3];
    }
    return res;
  }

  std::vector<unsigned char> decode(const std::string & s) {
    std::vector<unsigned char> res; res.reserve(3*s.size()/4);
    ThreeBytes t;
    // ok this is slow....
    for (int i=0; i<int(s.size());i+=4) {
      size_t p = letters.find(s[i]);
      if (p==std::string::npos) throw std::string("error");
      t.c24.c0=p;
      p = letters.find(s[i+1]);
      if (p==std::string::npos) throw std::string("error");
      t.c24.c1=p;
      p = letters.find(s[i+2]);
      if (p==std::string::npos) throw std::string("error");
      t.c24.c2=p;
      p = letters.find(s[i+3]);
      if (p==std::string::npos) throw std::string("error");
      t.c24.c3=p;
      copy(t.c,t.c+3,std::back_insert_iterator< std::vector<unsigned char> >(res));
    }
    return res;    
  }
  
};

int main() {
   std::vector<unsigned char> s;
  C826::FourBytes r1;  r1.i=::rand();
  C826::FourBytes r2;  r2.i=::rand();
  std::copy(r1.c,r1.c+4,std::back_insert_iterator< std::vector<unsigned char> >(s));
  std::copy(r2.c,r2.c+4,std::back_insert_iterator< std::vector<unsigned char> >(s));
  r1.i=::rand(); 
  r2.i=::rand(); 
  std::copy(r1.c,r1.c+4,std::back_insert_iterator< std::vector<unsigned char> >(s));
  std::copy(r2.c,r2.c+4,std::back_insert_iterator< std::vector<unsigned char> >(s));
  int i = s.size()%3;
  if(i!=0) for (;i<3;i++) s.push_back('-');
  std::cout << "size in " << s.size() << " "<< std::endl;
  // std::cout << s << std::endl;
  
  for (int j=0; j<int(s.size());j++) std::cout << (unsigned int)(s[j]) << "."; 
  std::cout << std::endl;

  std::string c = C826::encode(s);
  std::cout << "size out " << c.size() << std::endl;
  std::cout << c << std::endl;
  for (int j=0; j<int(c.size());j++) std::cout << int(c[j]) << "."; 
  std::cout << std::endl;

  std::vector<unsigned char> sn = C826::decode(c);
  std::cout << "size new " << sn.size() << " "<< std::endl;
  // std::cout << s << std::endl;
  
  for (int j=0; j<int(sn.size());j++) std::cout << (unsigned int)(sn[j]) << "."; 
  std::cout << std::endl;
  
 

  return 0;
  }

