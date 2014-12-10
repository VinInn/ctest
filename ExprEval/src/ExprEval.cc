#include "ExprEval.h"
#include "popenCPP.h"
#include <fstream>
#include <iostream>
#include <dlfcn.h>

namespace {
  std::string generateName() {
    auto s1 = popenCPP("uuidgen | sed 's/-//g'");
    char c; std::string n1;
    while (s1->get(c)) n1+=c;
    n1.pop_back();
    return n1;
  }

}

ExprEval::ExprEval(const char * iname, const char * iexpr) :
  m_name("VI_"+generateName())
{

  
  std::string factory = "factory" + m_name;

  std::string quote("\"");
  std::string source = std::string("#include ")+quote+iname+".h"+quote+"\n";
  source+="struct "+m_name+" final : public "+iname + "{\n";
  source+=iexpr;
  source+="\n};\n";


  source += "extern " + quote+'C'+quote+' ' + std::string(iname) + "* "+factory+"() {\n";
  source += "static "+m_name+" local;\n";
  source += "return &local;\n}\n";


  std::cout << source << std::endl;

  std::string sfile = "/tmp/"+m_name+".cc";
  std::string ofile = "/tmp/"+m_name+".so";

  {
    std::ofstream tmp(sfile.c_str());
    tmp<<source << std::endl;
  }

  std::string cpp = "c++ -H -std=c++14 -O3 -Wall -Ilib -shared -fPIC -Winvalid-pch -o ";
  cpp += ofile + ' ' + sfile+" 2>&1\n";

  std::cout << cpp << std::endl;

  try{
    auto ss = popenCPP(cpp);
    char c;
    while (ss->get(c)) std::cout << c;
    std::cout << std::endl;
  }catch(...) { std::cout << "error in popen" << cpp << std::endl;}

  void * dl = dlopen(ofile.c_str(),RTLD_LAZY);
  if (!dl) {
    std::cout << dlerror() <<std::endl;
    return;
  }

  m_expr = dlsym(dl,factory.c_str());

}


ExprEval::~ExprEval(){
  std::string sfile = "/tmp/"+m_name+".cc";
  std::string ofile = "/tmp/"+m_name+".so";

  std::string rm="rm -f "; rm+=sfile+' '+ofile;

  system(rm.c_str());

}
