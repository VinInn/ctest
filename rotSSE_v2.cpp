#include <stdlib.h>
#include <stdio.h>
#include <mmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <cmath>

using std::sin;
using std::cos;

static __inline__ unsigned long long rdtsc(void)
{
 unsigned hi, lo;
 __asm__ __volatile__ ("rdtsc":"=a"(lo),"=d"(hi));
 return ((unsigned long long)lo)|(((unsigned long long)hi)<<32);
}

class Fvec4
{
 private:
  union
  {
   float __attribute__ ((aligned(16))) values[4]; 
   struct
   {
    float _v0, _v1, _v2, _v3;
   };
  };
 public:
  Fvec4(float x, float y, float z, float t) : 
        _v0(x), _v1(y), _v2(z), _v3(t) {}
  Fvec4(__m128 v)
  {
   _mm_store_ps(values, v);
  }
  float v0() {return _v0;}
  float v1() {return _v1;}
  float v2() {return _v2;}
  float v3() {return _v3;}
  void to_screen()
  {
   printf("\n%f\n%f\n%f\n%f\n", _v0, _v1, _v2, _v3);
  }
};

class Fmat43
{
 private:
  union
  {
   float __attribute__ ((aligned(16))) values[12];
   struct
   {
    float _m00, _m01, _m02, _p0;
    float _m10, _m11, _m12, _p1;
    float _m20, _m21, _m22, _p2;
   };
  };

 public:
  
  void to_screen()
  {
   printf("%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n%f\t%f\t%f\t%f\n", _m00, _m01, _m02, _p0, _m10, _m11, _m12, _p1, _m20, _m21, _m22, _p2);
  }

  Fmat43(float xx, float xy, float xz, float px, 
         float yx, float yy, float yz, float py,
         float zx, float zy, float zz, float pz) :
         _m00(xx), _m01(xy), _m02(xz), _p0(px),
         _m10(yx), _m11(yy), _m12(yz), _p1(py),
         _m20(zx), _m21(zy), _m22(zz), _p2(pz) {}

  float m00() {return _m00;}
  float m01() {return _m01;}
  float m02() {return _m02;}
  float m10() {return _m10;}
  float m11() {return _m11;}
  float m12() {return _m12;}
  float m20() {return _m20;}
  float m21() {return _m21;}
  float m22() {return _m22;}
  float p0() {return _p0;}
  float p1() {return _p1;}
  float p2() {return _p2;}

  inline __m128 toGlobalP(Fvec4 & lp)
  {
   __m128 m0 = _mm_set_ps(0, _m02, _m01, _m00);
   __m128 m1 = _mm_set_ps(0, _m12, _m11, _m10);
   __m128 m2 = _mm_set_ps(0, _m22, _m21, _m20);

   __m128 t0 = _mm_set1_ps(lp.v0());
   __m128 t1 = _mm_set1_ps(lp.v1());
   __m128 t2 = _mm_set1_ps(lp.v2());

   t0 = _mm_mul_ps(m0, t0);
   t1 = _mm_mul_ps(m1, t1);
   t2 = _mm_mul_ps(m2, t2);

   __m128 rs = _mm_add_ps(t0, _mm_add_ps(t1, t2));
   __m128 p = _mm_set_ps(0, _p2, _p1, _p0);
   return _mm_add_ps(rs, p);
  }

  inline __m128 toGlobalV(Fvec4 & lv) 
  {
   __m128 m0 = _mm_set_ps(0, _m02, _m01, _m00);
   __m128 m1 = _mm_set_ps(0, _m12, _m11, _m10);
   __m128 m2 = _mm_set_ps(0, _m22, _m21, _m20);

   __m128 t0 = _mm_set1_ps(lv.v0());
   __m128 t1 = _mm_set1_ps(lv.v1());
   __m128 t2 = _mm_set1_ps(lv.v2());

   t0 = _mm_mul_ps(m0, t0);
   t1 = _mm_mul_ps(m1, t1);
   t2 = _mm_mul_ps(m2, t2);
   return _mm_add_ps(t0, _mm_add_ps(t1, t2)); 
  }

  inline __m128 toLocalP(Fvec4 & gp) 
  {
   __m128 m0 = _mm_set_ps(0, _m20, _m10, _m00);
   __m128 m1 = _mm_set_ps(0, _m21, _m11, _m01);
   __m128 m2 = _mm_set_ps(0, _m22, _m12, _m02);

   __m128 t0 = _mm_set1_ps(gp.v0()-_p0); 
   __m128 t1 = _mm_set1_ps(gp.v1()-_p1); 
   __m128 t2 = _mm_set1_ps(gp.v2()-_p2);

   t0 = _mm_mul_ps(m0, t0);
   t1 = _mm_mul_ps(m1, t1);
   t2 = _mm_mul_ps(m2, t2);
   return _mm_add_ps(t0, _mm_add_ps(t1, t2)); 
  }

  inline __m128 toLocalV(Fvec4 & gv) 
  {
   __m128 m0 = _mm_set_ps(0, _m20, _m10, _m00);
   __m128 m1 = _mm_set_ps(0, _m21, _m11, _m01);
   __m128 m2 = _mm_set_ps(0, _m22, _m12, _m02);

   __m128 t0 = _mm_set1_ps(gv.v0()); 
   __m128 t1 = _mm_set1_ps(gv.v1()); 
   __m128 t2 = _mm_set1_ps(gv.v2());

   t0 = _mm_mul_ps(m0, t0);
   t1 = _mm_mul_ps(m1, t1);
   t2 = _mm_mul_ps(m2, t2);
   return _mm_add_ps(t0, _mm_add_ps(t1, t2));
  }
};

Fvec4 do_rotate(float theta_degrees, Fvec4 axis_to_norm, Fvec4 p, Fvec4 vect)
{
 //consersion from degrees to radiants
 float theta = theta_degrees*M_PI/180;
 //axis vector normalization
 float norm = sqrt((axis_to_norm.v0() * axis_to_norm.v0()) + (axis_to_norm.v1() * axis_to_norm.v1()) + (axis_to_norm.v2() * axis_to_norm.v2()));
 Fvec4 axis(axis_to_norm.v0()/norm, axis_to_norm.v1()/norm, axis_to_norm.v2()/norm, 0);
 Fmat43 matr(
 axis.v0()*axis.v0()+(1-axis.v0()*axis.v0())*cos(theta),axis.v0()*axis.v1()*(1-cos(theta))-axis.v2()*sin(theta),axis.v0()*axis.v2()*(1-cos(theta))+axis.v1()*sin(theta),p.v0(),
 axis.v0()*axis.v1()*(1-cos(theta))+axis.v2()*sin(theta),axis.v1()*axis.v1()+(1-axis.v1()*axis.v1())*cos(theta),axis.v1()*axis.v2()*(1-cos(theta))-axis.v0()*sin(theta),p.v1(),
 axis.v0()*axis.v2()*(1-cos(theta))-axis.v1()*sin(theta),axis.v1()*axis.v2()*(1-cos(theta))+axis.v0()*sin(theta),axis.v2()*axis.v2()+(1-axis.v2()*axis.v2())*cos(theta),p.v2()
 );
 printf("\nBefore Rotation:\n");
 vect.to_screen();
 Fvec4 glob(matr.toGlobalP(vect));
 printf("\nAfter Rotation:\n");
 glob.to_screen();
 return glob;
}

Fvec4 undo_rotate(float theta_degrees, Fvec4 axis_to_norm, Fvec4 p, Fvec4 vect)
{
 //consersion from degrees to radiants
 float theta = theta_degrees*M_PI/180;
 //axis vector normalization
 float norm = sqrt((axis_to_norm.v0() * axis_to_norm.v0()) + (axis_to_norm.v1() * axis_to_norm.v1()) + (axis_to_norm.v2() * axis_to_norm.v2()));
 Fvec4 axis(axis_to_norm.v0()/norm, axis_to_norm.v1()/norm, axis_to_norm.v2()/norm, 0);
 Fmat43 matr(
 axis.v0()*axis.v0()+(1-axis.v0()*axis.v0())*cos(theta),axis.v0()*axis.v1()*(1-cos(theta))-axis.v2()*sin(theta),axis.v0()*axis.v2()*(1-cos(theta))+axis.v1()*sin(theta),p.v0(),
 axis.v0()*axis.v1()*(1-cos(theta))+axis.v2()*sin(theta),axis.v1()*axis.v1()+(1-axis.v1()*axis.v1())*cos(theta),axis.v1()*axis.v2()*(1-cos(theta))-axis.v0()*sin(theta),p.v1(),
 axis.v0()*axis.v2()*(1-cos(theta))-axis.v1()*sin(theta),axis.v1()*axis.v2()*(1-cos(theta))+axis.v0()*sin(theta),axis.v2()*axis.v2()+(1-axis.v2()*axis.v2())*cos(theta),p.v2()
 );
 printf("\nBefore Rotation:\n");
 vect.to_screen();
 Fvec4 loc(matr.toLocalP(vect));
 printf("\nAfter Rotation:\n");
 loc.to_screen();
 return loc;
}

int main(void)
{
 float theta_1 = -90.0;
 Fvec4 vect   (-1,  1,  2,  0);
 Fvec4 axis_x ( 1,  0,  0,  0);
 Fvec4 p_1    ( 2,  3,  4,  0);
 Fvec4 rot1 = do_rotate(theta_1, axis_x, p_1, vect); //rotation about x-axis of -90 degrees (+traslation)
 float theta_2 = 30.0;
 Fvec4 axis_y ( 0,  1,  0,  0);
 Fvec4 p_2    ( -1,  5,  -1.2,  0);
 Fvec4 rot2 = do_rotate(theta_2, axis_y, p_2, rot1); //rotation about y-axis of -30 degrees (+traslation)
 float theta_3 = 0.0001;
 Fvec4 axis_z ( 0,  0,  1,  0);
 Fvec4 p_3    ( 0.0345,  -3,  40,  0);
 Fvec4 rot3 = do_rotate(theta_3, axis_z, p_3, rot2); //rotation about z-axis of 0.0001 degrees (+traslation)
 float theta_4 = 10.0001;
 Fvec4 axis_c1 ( 12,  -3.5,  1.7,  0);
 Fvec4 p_4     ( 120.0345,  -3.15,  -21,  0);
 Fvec4 rot4 = do_rotate(theta_4, axis_c1, p_4, rot3); //rotation about a custom axis of 10.0001 degrees (+traslation)
 Fvec4 rot5 = undo_rotate(theta_4, axis_c1, p_4, rot4); //undo rotation about a custom axis of 10.0001 degrees (+traslation)
 Fvec4 rot6 = undo_rotate(theta_3, axis_z, p_3, rot5); //undo rotation about z-axis of 0.0001 degrees (+traslation)
 Fvec4 rot7 = undo_rotate(theta_2, axis_y, p_2, rot6); //undo rotation about y-axis of -30 degrees (+traslation)
 Fvec4 rot8 = undo_rotate(theta_1, axis_x, p_1, rot7); //undo rotation about x-axis of -90 degrees (+traslation)
 return 0;
}
