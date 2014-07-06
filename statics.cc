#include <map>
#include <vector>
#include <string>

struct  EvtId {
  constexpr EvtId(){}
  constexpr EvtId(int l1, int l2) : i1(l1), i2(l2){}
  int i1=-1;
  int i2=-1;

};

struct EvtPartProp {
  EvtId id;
  constexpr EvtId getId() const { return id;}
};

namespace EvtPDL {

  ///std::vector<EvtPartProp> _partlist;
  EvtPartProp _partlist[14];
  std::map<std::string, int> _particleNameLookup;

  EvtId getId(const std::string& name ){
    
    std::map<std::string,int>::iterator it=
      _particleNameLookup.find(std::string(name));
    if (it==_particleNameLookup.end()) return EvtId(-1,-1);
    
    return _partlist[it->second].getId();
    
  }
  
  struct entry {
    const char * label;
    int index;
  };

  constexpr entry theMap[] = {
    {"D-", 0},
    {"D+", 1},
    {"D0", 2},
    {"anti-D0", 3},
    {"K-", 4},
    {"K+", 5},
    {"k0", 6},
    {"anti-K0", 7},
    {"K_L0", 8},
    {"K_S0", 9},
    {"pi-",10},
    {"pi+",11},
    {"pi0",12},
    {nullptr,13}
  };
  
  inline constexpr bool  same(char const *x, char const *y)   {
    return !*x && !*y ? true : (*x == *y && same(x+1, y+1));
  }
  
  inline constexpr int getIndex(char const *label, entry const *entries)   {
    return !entries->label ? entries->index  : same(entries->label, label) ? entries->index : getIndex(label, entries+1);
  }
  
  inline  EvtId __attribute__((always_inline)) cgetId(const char * name )  {
    return  _partlist[getIndex(name,theMap)].getId();
  }


};  

#define SetEvtId(X,NAME) \
 constexpr int i_##X = EvtPDL::getIndex(NAME, EvtPDL::theMap);\
 const EvtId X =  EvtPDL::_partlist[i_##X].getId()

namespace a1{

#ifdef BAR
  int bar(int j) {
    
    static const EvtId DM=EvtPDL::getId("D-");
    static const EvtId DP=EvtPDL::getId("D+");
    static const EvtId D0=EvtPDL::getId("D0");
    static const EvtId D0B=EvtPDL::getId("anti-D0");
    static const EvtId KM=EvtPDL::getId("K-");
    static const EvtId KP=EvtPDL::getId("K+");
    static const EvtId K0=EvtPDL::getId("K0");
    static const EvtId KB=EvtPDL::getId("anti-K0");
    static const EvtId KL=EvtPDL::getId("K_L0");
    static const EvtId KS=EvtPDL::getId("K_S0");
    static const EvtId PIM=EvtPDL::getId("pi-");
    static const EvtId PIP=EvtPDL::getId("pi+");
    static const EvtId PI0=EvtPDL::getId("pi0");
    
    // use all 
    int ret=0;
    if ( DM.i1==j) ++ret;
    if ( DP.i1==j) ++ret;
    if ( D0.i1==j) ++ret;
    if ( D0B.i1==j) ++ret;
    if ( KM.i1==j) ++ret;
    if ( KP.i1==j) ++ret;
    if ( K0.i1==j) ++ret;
    if ( KB.i1==j) ++ret;
    if ( KL.i1==j) ++ret;
    if ( KS.i1==j) ++ret;
    if ( PIM.i1==j) ++ret;
    if ( PIP.i1==j) ++ret;
    if ( PI0.i1==j) ++ret;
    
    return ret;
    
  }
#endif
  
#ifdef FOO
  int foo(int j) {
    
    static struct {
      const EvtId DM=EvtPDL::getId("D-");
      const EvtId DP=EvtPDL::getId("D+");
      const EvtId D0=EvtPDL::getId("D0");
      const EvtId D0B=EvtPDL::getId("anti-D0");
      const EvtId KM=EvtPDL::getId("K-");
      const EvtId KP=EvtPDL::getId("K+");
      const EvtId K0=EvtPDL::getId("K0");
      const EvtId KB=EvtPDL::getId("anti-K0");
      const EvtId KL=EvtPDL::getId("K_L0");
      const EvtId KS=EvtPDL::getId("K_S0");
      const EvtId PIM=EvtPDL::getId("pi-");
      const EvtId PIP=EvtPDL::getId("pi+");
      const EvtId PI0=EvtPDL::getId("pi0");
    } const parts;
    
    // use all 
    int ret=0;
    if ( parts.DM.i1==j) ++ret;
    if ( parts.DP.i1==j) ++ret;
    if ( parts.D0.i1==j) ++ret;
    if ( parts.D0B.i1==j) ++ret;
    if ( parts.KM.i1==j) ++ret;
    if ( parts.KP.i1==j) ++ret;
    if ( parts.K0.i1==j) ++ret;
    if ( parts.KB.i1==j) ++ret;
    if ( parts.KL.i1==j) ++ret;
    if ( parts.KS.i1==j) ++ret;
    if ( parts.PIM.i1==j) ++ret;
    if ( parts.PIP.i1==j) ++ret;
    if ( parts.PI0.i1==j) ++ret;
    
    return ret;
    
  }
#endif


#ifdef KET

  int bra(int j) {
    using namespace EvtPDL;
    constexpr int a1= getIndex("anti-D0",theMap);
    constexpr int a2= getIndex("D0",theMap);
 
    if (j>0) 
      return a1;
    return a2;
  }


  int ket(int j) {
    
     SetEvtId(DM, "D-");

#ifndef ONE
     SetEvtId(DP, "D+");
     SetEvtId(D0, "D0");
     SetEvtId(D0B, "anti-D0");
     SetEvtId(KM, "K-");
     SetEvtId(KP, "K+");
     SetEvtId(K0, "K0");
     SetEvtId(KB, "anti-K0");
     SetEvtId(KL, "K_L0");
     SetEvtId(KS, "K_S0");
     SetEvtId(PIM, "pi-");
     SetEvtId(PIP, "pi+");
     SetEvtId(PI0, "pi0");
#endif
    
    // use all 
    int ret=0;
    if ( DM.i1==j) ++ret;
#ifndef ONE
    if ( DP.i1==j) ++ret;
    if ( D0.i1==j) ++ret;
    if ( D0B.i1==j) ++ret;
    if ( KM.i1==j) ++ret;
    if ( KP.i1==j) ++ret;
    if ( K0.i1==j) ++ret;
    if ( KB.i1==j) ++ret;
    if ( KL.i1==j) ++ret;
    if ( KS.i1==j) ++ret;
    if ( PIM.i1==j) ++ret;
    if ( PIP.i1==j) ++ret;
    if ( PI0.i1==j) ++ret;
#endif
    return ret;
    
  }
#endif

}
