#include "PPCRealTime.h"
#include <string>
#include <iostream>


inline void byValue(std::string s) {

  char p = s[0];
  if (p=='a') std::cout << "never" << std::endl;
}

inline void byRef(const std::string & s) {

  char p = s[0];
  if (p=='a') std::cout << "never" << std::endl;
}

inline void array(const std::string & a, std::string * s, int n, bool * l) {
  // static const std::string b("longish);
  for (int i=0;i<n;i++) {
    if (l[i]) s[i]=a;
    else s[i] = "bhabha";
  }
}

inline void array_st(const std::string & a, std::string * s, int n, bool * l) {
  static const std::string b("longish");
  for (int i=0;i<n;i++) {
    if (l[i]) s[i]=a;
    else s[i] = b;
  }

}

int main() {



  typedef  unsigned long long int TimeT;
  typedef  long long int TimeD;

  { 
    TimeT before = rdtsc();
    
    std::string s1("short string");
    
    TimeD d =  rdtsc()-before;
    
    std::cout << "small constr " << d << std::endl;

    before = rdtsc();
    
    std::string s2(s1);
    
    d =  rdtsc()-before;
    
    std::cout << "small copy - constr " << d << std::endl;

    before = rdtsc();
    
    std::string s3 = s1;
    
    d =  rdtsc()-before;
    
    std::cout << "small assign " << d << std::endl;

    ////
    before = rdtsc();
    
    s3[456]='w';
    
    d =  rdtsc()-before;
    
    std::cout << "small mod " << d << std::endl;

    ///
    before = rdtsc();
    
    byValue(s3);
    
    d =  rdtsc()-before;
    
    std::cout << "small by value " << d << std::endl;

    ///
    before = rdtsc();

    byRef(s3);
    
    d =  rdtsc()-before;
    
    std::cout << "small by ref " << d << std::endl;

    ///
    before = rdtsc();
    
    std::string s4;

    for (int i=0; i!=s3.size();i++)
      s4+=s3[i];

    d =  rdtsc()-before;
    
    std::cout << "small loop " << d << std::endl;
    ///

    before = rdtsc();
    
    std::string s5; s5.reserve(s3.size());

    for (int i=0; i!=s3.size();i++)
      s5+=s3[i];

    d =  rdtsc()-before;
    
    std::cout << "small loop reserve " << d << std::endl;


   std::string sq[10];
   std::string sa[10];
   std::string sb[10];
   std::string sc[10];
   std::string sd[10];
   bool t[10]; std::fill(t,t+10,true);
   bool f[10]; std::fill(f,f+10,false);
   // bild static
   array(s3,sq,1,t);
   array_st(s3,sq,1,t);
   before = rdtsc();
   array(s3,sa,10,t);
   d =  rdtsc()-before;
   std::cout << "small assign array " << d << std::endl;
   before = rdtsc();
   array(s3,sb,10,f);
   d =  rdtsc()-before;
   std::cout << "small assign array char " << d << std::endl;
   before = rdtsc();
   array_st(s3,sc,10,t);
   d =  rdtsc()-before;
   std::cout << "small assign array 2 " << d << std::endl;
   before = rdtsc();
   array_st(s3,sd,10,f);
   d =  rdtsc()-before;
   std::cout << "small assign array static " << d << std::endl;

  }

  { 
    TimeT before = rdtsc();
    
    std::string big1(400000,'q');
    
    TimeD d =  rdtsc()-before;
    
    std::cout << "big constr " << d << std::endl;

    before = rdtsc();
    
    std::string big2(big1);
    
    d =  rdtsc()-before;
    
    std::cout << "big copy - constr " << d << std::endl;

    before = rdtsc();
    
    std::string big3 = big1;
    
    d =  rdtsc()-before;
    
    std::cout << "big assign " << d << std::endl;

    before = rdtsc();
    
    { 
      std::string bigl = big1;
      std::string p = bigl.substr(0,1);
      if (p== "a") std::cout << "never" << std::endl;
    }
    
    d =  rdtsc()-before;
    
    std::cout << "big assign and destroy " << d << std::endl;

    ////
    before = rdtsc();
    
    big3[456]='w';
    
    d =  rdtsc()-before;
    
    std::cout << "big [] " << d << std::endl;

   ////
    before = rdtsc();
    
    big3[456]='w';
    
    d =  rdtsc()-before;
    
    std::cout << "big mod " << d << std::endl;

    ///
    before = rdtsc();
    
    byValue(big3);
    
    d =  rdtsc()-before;
    
    std::cout << "big by value " << d << std::endl;


   ///
    before = rdtsc();
    
    byRef(big3);
    
    d =  rdtsc()-before;
    
    std::cout << "big by ref " << d << std::endl;

    std::string big4;

    for (int i=0; i!=big3.size();i++)
      big4+=big3[i];

    d =  rdtsc()-before;
    
    std::cout << "small loop " << d << std::endl;
    ///

    before = rdtsc();
    
    std::string big5; big5.reserve(big3.size());

    for (int i=0; i!=big3.size();i++)
      big5+=big3[i];

    d =  rdtsc()-before;
    
    std::cout << "small loop reserve " << d << std::endl;


  }


  return 0;

}



