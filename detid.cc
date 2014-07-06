
typedef unsigned int uint32_t;

class DetId {
public:
  static const int kDetOffset          = 28;
  static const int kSubdetOffset       = 25;


  enum Detector { Tracker=1,Muon=2,Ecal=3,Hcal=4,Calo=5 };
  /// Create an empty or null id (also for persistence)
  DetId()  : id_(0) { }
  /// Create an id from a raw number
  DetId(uint32_t id) : id_(id) { }
  /// Create an id, filling the detector and subdetector fields as specified
  DetId(Detector det, int subdet)  {
    id_=((det&0xF)<<28)|((subdet&0x7)<<25);
  }

  /// get the detector field from this detid
  Detector det() const { return Detector((id_>>kDetOffset)&0xF); }
  /// get the contents of the subdetector field (not cast into any detector's numbering enum)
  int subdetId() const { return ((id_>>kSubdetOffset)&0x7); }

  uint32_t operator()() const { return id_; }
  operator uint32_t() const { return id_; }

  /// get the raw id 
  uint32_t rawId() const { return id_; }
  /// is this a null id ?
  bool null() const { return id_==0; }
  
  /// equality
  bool operator==(DetId id) const { return id_==id.id_; }
  /// inequality
  bool operator!=(DetId id) const { return id_!=id.id_; }
  /// comparison
  bool operator<(DetId id) const { return id_<id.id_; }

protected:
  uint32_t id_;
};



class PixelSubdetector { 
 public:

static const unsigned int PixelBarrel=1;
static const unsigned int PixelEndcap=2; 
};
class PXBDetId : public DetId {
 public:
  /** Constructor of a null id */
  PXBDetId(){}
  /** Constructor from a raw value */
  PXBDetId(uint32_t rawid) : DetId(rawid){}
  /**Construct from generic DetId */
  PXBDetId(const DetId& id) : DetId(id){}
  
  PXBDetId(uint32_t layer,
	   uint32_t ladder,
	   uint32_t module) : DetId(DetId::Tracker,PixelSubdetector::PixelBarrel){
    id_ |= (layer& layerMask_) << layerStartBit_     |
      (ladder& ladderMask_) << ladderStartBit_  |
      (module& moduleMask_) << moduleStartBit_;
  }
  
  
  /// layer id
  unsigned int layer() const{
    return int((id_>>layerStartBit_) & layerMask_);}
  
  /// ladder  id
  unsigned int ladder() const
    { return ((id_>>ladderStartBit_) & ladderMask_) ;}
  
  /// det id
  unsigned int module() const 
    { return ((id_>>moduleStartBit_)& moduleMask_) ;}
  
 private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int layerStartBit_=   16;
  static const unsigned int ladderStartBit_=   8;
  static const unsigned int moduleStartBit_=   2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int layerMask_=       0xF;
  static const unsigned int ladderMask_=      0xFF;
  static const unsigned int moduleMask_=      0x3F;
};

namespace detId {
  struct Base {
    Base(uint32_t rawId) { (uint32_t&)(*this) = rawId;}
    uint32_t rawId() const { return (uint32_t&)(*this);}

    unsigned int det:    4;
    unsigned int subdet: 3;
    unsigned int pad: 25;
  };

  struct PXB {
    PXB(uint32_t ilayer,
	uint32_t iladder,
	uint32_t imodule) : 
      det(DetId::Tracker),
      subdet(PixelSubdetector::PixelBarrel),
      highpad(0),
      layer(ilayer),
      ladder(iladder),
      module(imodule),
      lowpad(0){}
    PXB(uint32_t rawId) { (uint32_t&)(*this) = rawId;}
    uint32_t rawId() const { return (uint32_t&)(*this);}
    operator uint32_t() const { return rawId(); }

 
    /*
    unsigned int det:    4;
    unsigned int subdet: 3;
     
    unsigned int highpad: 5;
    unsigned int layer:   4;
    unsigned int ladder:  8;
    unsigned int module:  6;
    unsigned int lowpad:  2;
    */

    unsigned int lowpad:  2;
    unsigned int module:  6;
    unsigned int ladder:  8;
    unsigned int layer:   4;
    unsigned int highpad: 5;

    unsigned int subdet: 3;
    unsigned int det:    4;
    
  };

}


#include"RealTime.h"
#include<iostream>

int bdigi[10][10][10];
int edigi[10][10][10];


inline void fill( PXBDetId id, int k) {
  if (id.subdetId()==PixelSubdetector::PixelBarrel)
    bdigi[id.layer()][id.ladder()][id.module()]=k;
  //  else if(id.subdetId()==PixelSubdetector::PixelEndcap)
  //  edigi[id.layer()][digi.ladder()][digi.module()]=k;
}

inline void fill( detId::PXB id, int k) {
  if (id.subdet==PixelSubdetector::PixelBarrel)
    bdigi[id.layer][id.ladder][id.module]=k;
  //  else if(id.subdetId()==PixelSubdetector::PixelEndcap)
  //  edigi[id.layer()][digi.ladder()][digi.module()]=k;
}

using namespace perftools;

int main() {

  TimeType a = realTime();
  PXBDetId id(1,2,3);
  TimeType b = realTime();
  detId::PXB idf(1,2,3);
  TimeType c = realTime();
  detId::PXB idf2(id);

  std::cout << "shift "<< b-a << std::endl;
  std::cout << "fields "<< c-b << std::endl;

  fill(id,4);
  fill(idf,4);
 
  TimeType a2 = realTime();
  fill(id,6);
  TimeType b2 = realTime();
  fill(idf,6);
  TimeType c2 = realTime();

  std::cout << "shift "<< b2-a2 << std::endl;
  std::cout << "fields "<< c2-b2 << std::endl;


  std:: cout << "( " 
	     << id.det()  << ',' 
	     << id.subdetId() << ',' 
	     << id.layer() << ',' 
	     << id.ladder() << ',' 
	     << id.module() << ')'
	     << std::endl;

  std:: cout << "( " 
	     << idf.det  << ',' 
	     << idf.subdet << ',' 
	     << idf.layer << ',' 
	     << idf.ladder << ',' 
	     << idf.module << ')'
	     << std::endl;

  std:: cout << "( " 
	     << idf2.det  << ',' 
	     << idf2.subdet << ',' 
	     << idf2.layer << ',' 
	     << idf2.ladder << ',' 
	     << idf2.module << ')'
	     << std::endl;


  return 0;

}
