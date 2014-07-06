
#include<cmath>
inline float __attribute__((always_inline)) __attribute__ ((pure))
eta(float x, float y, float z) { float t(z/std::sqrt(x*x+y*y)); return ::asinhf(t);} 



class EtaCut {
public:
  static constexpr float convert(float eta) {  return std::copysign(powf(::sinhf(eta),2.f),eta); }
  constexpr EtaCut(float icut) : cut(convert(icut)){}

  // v.eta < cut
  template<typename V>
  bool less(V const & v) {
    return std::copysign(v.z()*v.z(),v.z()) < cut*v.perp2();
  } 
  // abs(v.eta) < cut
  template<typename V>
  bool absless(V const & v) {
    return v.z()*v.z() < std::abs(cut)*v.perp2();
  } 

  float cut;

};


constexpr float chord(float x) {return 2.f*std::sin(0.5f*x);}
constexpr float phiFromChord(float x) {return 2.f*std::asin(0.5f*x);}

// valid for cut<pi
class DPhiCut{
public:
  constexpr DPhiCut(float icut, float ir) : cut(std::pow(ir*chord(icut),2.f)){}
   //  (v1.phi-v2-phi) < cut ...   asumme identical r
  template<typename V>
  bool less(V const & v1,V const & v2) {
    float d = (v1.x()-v2.x())*(v1.x()-v2.x())+
      (v1.y()-v2.y())*(v1.y()-v2.y());
    return d<cut;
  }

  float cut;

};


// cut in eta phi around a vector
class ConeCut {
  static constexpr float convert(float eta) {  return std::copysign(powf(::sinhf(eta),2.f),eta); }
  template<typename V, typename F>
  ConeCut(V const & v, F idR) : 
    vx(v.x()),vy(v.y()),vz(v.z()),
    veta(eta(x,y,z)), vphi(::atanf(vy,vx)),vr2(vx*vx+vy*vy),ovr(1/std::sqrt(vr2));
    dR2(idr*idr) { 
    zmin = convert(veta-idr);
    zmax = convert(veta+idr);
    pmax = cos(std::min(idr,M_PI))*vr2;
  }

 
  template<typename V> 
  bool square(V const & v) {
    float x=v.x(),y=v.y(),z=v.z();
    float r2 = x*x+y*y;
    float z2 = z*z;
    float dot = x*vx+y*vy;
    return 
      z2 > zmin*r2
      && z2 < zmax*r2
	       && dot>pmax*r2;
  }
  
  template<typename V> 
  bool cone(V const & v) {
    float x=v.x(),y=v.y(),z=v.z();
    float e = eta(x,y,z);
    float dp = std:acos((x*vx+y*vy)*ovr/std::sqrt(x*x+y*y));
    return dp*dp+(e-veta)*(e-veta) < dr2;

  }

  float vx,vy,vz, veta, vphi,vr2,ovr
  float dR2;
  float zmin, zmax;
  float pmax;

};



struct Vector {
  float a,b,c;

  float x() const { return a;}
  float y() const { return b;}
  float z() const { return c;}
  float perp2() const { return a*a+b*b;}
  float eta() const { return ::eta(a,b,c);}

};

#include<vector>
#include<cstdio>
#include<iostream>
int main() {

  for (float e=-8.; e<8.1; e+=0.5) {
    EtaCut cut(e);
    printf("%f %f\n",e, cut.cut);
    for (float x=-1000.; x<=1010.; x+=100) 
      for (float y=-1000.; y<=1010.; y+=100) 
	for (float z=-1000.; z<=1010.; z+=100) {
	  Vector v={x,y,z};
	  if  ( (v.eta()<e)!=(cut.less(v) ) )
	    printf("< %f: %f %f,%f,%f\n",e, v.eta(),x,y,z);
	  if  ( (std::abs(v.eta())<std::abs(e))!=(cut.absless(v) ) )
	    printf("abs %f: %f %f,%f,%f\n",e, v.eta(),x,y,z);
	}
  }

  for (float dp=0; dp<3.14; dp+=0.2)
    for (float r=1.; r<=1010.; r+=100) { 
      DPhiCut cut(dp,r);
      for (float p1=0; p1<6.28; p1+=0.2)  
	for (float d2=-3.14; d2<3.14; d2+=0.2) {
	  float p2=p1+d2;
	  Vector v1= {r*std::cos(p1),r*std::sin(p1),30.f};
	  Vector v2={r*std::cos(p2),r*std::sin(p2),30.f};
	  if ( (std::abs(d2)<dp) != cut.less(v1,v2))
	    printf("< %f %f: %f,%f\n",dp, r ,p1,p2);
	}
    }


  std::vector<Vector> vs;
   for (float x=-1000.; x<=1010.; x+=100) 
      for (float y=-1000.; y<=1010.; y+=100) 
	for (float z=-1000.; z<=1010.; z+=100)
	  vs.emplace_back(x,y,z);
   std::cout << "testing " << v.size() << " vectors" << std::endl;

   for (float dr=0.1; dr<3; dr+=0.2)
     for (auto const & v1 : vs)
       for (auto const & v2 : vs) {
	 ConeCut cut(v1,dr);
	 
       }
}


  return 0;
}
