#include<cmath>



/**
 * fully inlined specialized code to perform the inversion of a
 * positive defined matrix of rank up to 6.
 *
 * originally by
 * @author Manuel Schiller
 * @date Aug 29 2008
 *
 *
 */
namespace choleskyInversion {

  template<typename M1, typename M2> 
  inline constexpr
  void invert11(M1 const & src, M2 & dst) {
    using F = decltype(src(0,0));
    dst(0,0) = F(1.0) / src(0,0);
  }
  
  template<typename M1, typename M2> 
  inline constexpr
  void invert22(M1 const & src, M2 & dst) {
    using F = decltype(src(0,0));
    auto luc0 = F(1.0) / src(0,0);
    auto luc1 = src(1,0)*src(1,0) * luc0;
    auto luc2 = F(1.0) / (src(1,1)-luc1);

    auto li21 = luc1 * luc0 * luc2;

    dst(0,0) = li21 + luc0;
    dst(1,0) = - src(1,0)*luc0*luc2;
    dst(1,1) = luc2;
  }
  
  template<typename M1, typename M2> 
  inline constexpr
  void invert33(M1 const & src, M2 & dst) {
    using F = decltype(src(0,0));
    auto luc0 = F(1.0) / std::sqrt(src(0,0));
    auto luc1 = src(1,0) * luc0;
    auto luc2 = src(1,1) - luc1 * luc1;
    luc2 = F(1.0) / std::sqrt(luc2);
    auto luc3 = src(2,0) * luc0;
    auto luc4 = (src(2,1) - luc1 * luc3) * luc2;
    auto luc5 = src(2,2) - (luc3 * luc3 + luc4 * luc4);
    luc5 = F(1.0) / std::sqrt(luc5);
    
    auto li21 = -luc1 * luc0 * luc2;
    auto li32 = -luc4 * luc2 * luc5;
    auto li31 = (luc1 * luc4 * luc2 - luc3) * luc0 * luc5;

    dst(0,0) = li31*li31 + li21*li21 + luc0*luc0;
    dst(1,0) = li31*li32 + li21*luc2;
    dst(1,1) = li32*li32 + luc2*luc2;
    dst(2,0) = li31*luc5;
    dst(2,1) = li32*luc5;
    dst(2,2) = luc5*luc5;
    
  }

  template<typename M1, typename M2> 
  inline constexpr
  void invert44(M1 const & src, M2 & dst) {
    using F = decltype(src(0,0));
    auto luc0 = F(1.0) / std::sqrt(src(0,0));
    auto luc1 = src(1,0) * luc0;
    auto luc2 = src(1,1) - luc1 * luc1;
    luc2 = F(1.0) / std::sqrt(luc2);
    auto luc3 = src(2,0) * luc0;
    auto luc4 = (src(2,1) - luc1 * luc3) * luc2;
    auto luc5 = src(2,2) - (luc3 * luc3 + luc4 * luc4);
    luc5 = F(1.0) / std::sqrt(luc5);
    auto luc6 = src(3,0) * luc0;
    auto luc7 = (src(3,1) - luc1 * luc6) * luc2;
    auto luc8 = (src(3,2) - luc3 * luc6 - luc4 * luc7) * luc5;
    auto luc9 = src(3,3) - (luc6 * luc6 + luc7 * luc7 + luc8 * luc8);
    luc9 = F(1.0) / std::sqrt(luc9);
    
    auto li21 = -luc1 * luc0 * luc2;
    auto li32 = -luc4 * luc2 * luc5;
    auto li31 = (luc1 * luc4 * luc2 - luc3) * luc0 * luc5;
    auto li43 = -luc8 * luc9 * luc5;
    auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2 * luc9;
    auto li41 = (-luc1 * luc4 * luc8 * luc2 * luc5 +
		 luc1 * luc7 * luc2 + luc3 * luc8 * luc5 - luc6) * luc0 * luc9;

    dst(0,0) = li41*li41 + li31*li31 + li21*li21 + luc0*luc0;
    dst(1,0) = li41*li42 + li31*li32 + li21*luc2;
    dst(1,1) = li42*li42 + li32*li32 + luc2*luc2;
    dst(2,0) = li41*li43 + li31*luc5;
    dst(2,1) = li42*li43 + li32*luc5;
    dst(2,2) = li43*li43 + luc5*luc5;
    dst(3,0) = li41*luc9;
    dst(3,1) = li42*luc9;
    dst(3,2) = li43*luc9;
    dst(3,3) = luc9*luc9;  
    
  }

  
  template<typename M1, typename M2> 
  inline constexpr
  void invert55(M1 const & src, M2 & dst) {
    using F = decltype(src(0,0));
    auto luc0 = F(1.0) / std::sqrt(src(0,0));
    auto luc1 = src(1,0) * luc0;
    auto luc2 = src(1,1) - luc1 * luc1;
    luc2 = F(1.0) / std::sqrt(luc2);
    auto luc3 = src(2,0) * luc0;
    auto luc4 = (src(2,1) - luc1 * luc3) * luc2;
    auto luc5 = src(2,2) - (luc3 * luc3 + luc4 * luc4);
    luc5 = F(1.0) / std::sqrt(luc5);
    auto luc6 = src(3,0) * luc0;
    auto luc7 = (src(3,1) - luc1 * luc6) * luc2;
    auto luc8 = (src(3,2) - luc3 * luc6 - luc4 * luc7) * luc5;
    auto luc9 = src(3,3) - (luc6 * luc6 + luc7 * luc7 + luc8 * luc8);
    luc9 = F(1.0) / std::sqrt(luc9);
    auto luc10 = src(4,0) * luc0;
    auto luc11 = (src(4,1) - luc1 * luc10) * luc2;
    auto luc12 = (src(4,2) - luc3 * luc10 - luc4 * luc11) * luc5;
    auto luc13 = (src(4,3) - luc6 * luc10 - luc7 * luc11 - luc8 * luc12) * luc9;
    auto luc14 = src(4,4) - (luc10*luc10+luc11*luc11+luc12*luc12+luc13*luc13);
    luc14 = F(1.0) / std::sqrt(luc14);
    
    
    auto li21 = -luc1 * luc0 * luc2;
    auto li32 = -luc4 * luc2 * luc5;
    auto li31 = (luc1 * luc4 * luc2 - luc3) * luc0 * luc5;
    auto li43 = -luc8 * luc9 * luc5;
    auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2 * luc9;
    auto li41 = (-luc1 * luc4 * luc8 * luc2 * luc5 +
		 luc1 * luc7 * luc2 + luc3 * luc8 * luc5 - luc6) * luc0 * luc9;
    auto li54 = -luc13 * luc14 * luc9;
    auto li53 = (luc13 * luc8 * luc9 - luc12) * luc5 * luc14;
    auto li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 +
		 luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2 * luc14;
    auto li51 = (luc1*luc4*luc8*luc13*luc2*luc5*luc9 -
		 luc13*luc8*luc3*luc9*luc5 - luc12*luc4*luc1*luc2*luc5 - luc13*luc7*luc1*luc9*luc2 +
		 luc11*luc1*luc2 + luc12*luc3*luc5 + luc13*luc6*luc9 -luc10) * luc0 * luc14;
    
    dst(0,0) = li51*li51 + li41*li41 + li31*li31 + li21*li21 + luc0*luc0;
    dst(1,0) = li51*li52 + li41*li42 + li31*li32 + li21*luc2;
    dst(1,1) = li52*li52 + li42*li42 + li32*li32 + luc2*luc2;
    dst(2,0) = li51*li53 + li41*li43 + li31*luc5;
    dst(2,1) = li52*li53 + li42*li43 + li32*luc5;
    dst(2,2) = li53*li53 + li43*li43 + luc5*luc5;
    dst(3,0) = li51*li54 + li41*luc9;
    dst(3,1) = li52*li54 + li42*luc9;
    dst(3,2) = li53*li54 + li43*luc9;
    dst(3,3) = li54*li54 + luc9*luc9;
    dst(4,0) = li51*luc14;
    dst(4,1) = li52*luc14;
    dst(4,2) = li53*luc14;
    dst(4,3) = li54*luc14;
    dst(4,4) = luc14*luc14;
  }

  template<typename M1, typename M2> 
  inline __attribute__((always_inline)) constexpr
  void invert66(M1 const & src, M2 & dst) {
    using F = decltype(src(0,0));
    auto luc0 = F(1.0) / std::sqrt(src(0,0));
    auto luc1 = src(1,0) * luc0;
    auto luc2 = src(1,1) - luc1 * luc1;
    luc2 = F(1.0) / std::sqrt(luc2);
    auto luc3 = src(2,0) * luc0;
    auto luc4 = (src(2,1) - luc1 * luc3) * luc2;
    auto luc5 = src(2,2) - (luc3 * luc3 + luc4 * luc4);
    luc5 = F(1.0) / std::sqrt(luc5);
    auto luc6 = src(3,0) * luc0;
    auto luc7 = (src(3,1) - luc1 * luc6) * luc2;
    auto luc8 = (src(3,2) - luc3 * luc6 - luc4 * luc7) * luc5;
    auto luc9 = src(3,3) - (luc6 * luc6 + luc7 * luc7 + luc8 * luc8);
    luc9 = F(1.0) / std::sqrt(luc9);
    auto luc10 = src(4,0) * luc0;
    auto luc11 = (src(4,1) - luc1 * luc10) * luc2;
    auto luc12 = (src(4,2) - luc3 * luc10 - luc4 * luc11) * luc5;
    auto luc13 = (src(4,3) - luc6 * luc10 - luc7 * luc11 - luc8 * luc12) * luc9;
    auto luc14 = src(4,4) - (luc10*luc10+luc11*luc11+luc12*luc12+luc13*luc13);
    luc14 = F(1.0) / std::sqrt(luc14);
    auto luc15 = src(5,0) * luc0;
    auto luc16 = (src(5,1) - luc1 * luc15) * luc2;
    auto luc17 = (src(5,2) - luc3 * luc15 - luc4 * luc16) * luc5;
    auto luc18 = (src(5,3) - luc6 * luc15 - luc7 * luc16 - luc8 * luc17) * luc9;
    auto luc19 = (src(5,4) - luc10 * luc15 - luc11 * luc16 - luc12 * luc17 - luc13 * luc18) * luc14;
    auto luc20 = src(5,5) - (luc15*luc15+luc16*luc16+luc17*luc17+luc18*luc18+luc19*luc19);
    luc20 = F(1.0) / std::sqrt(luc20);

    auto li21 = -luc1 * luc0 * luc2;
    auto li32 = -luc4 * luc2 * luc5;
    auto li31 = (luc1 * luc4 * luc2 - luc3) * luc0 * luc5;
    auto li43 = -luc8 * luc9 * luc5;
    auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2 * luc9;
    auto li41 = (-luc1 * luc4 * luc8 * luc2 * luc5 +
		 luc1 * luc7 * luc2 + luc3 * luc8 * luc5 - luc6) * luc0 * luc9;
    auto li54 = -luc13 * luc14 * luc9;
    auto li53 = (luc13 * luc8 * luc9 - luc12) * luc5 * luc14;
    auto li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 +
		 luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2 * luc14;
    auto li51 = (luc1*luc4*luc8*luc13*luc2*luc5*luc9 -
		 luc13*luc8*luc3*luc9*luc5 - luc12*luc4*luc1*luc2*luc5 - luc13*luc7*luc1*luc9*luc2 +
		 luc11*luc1*luc2 + luc12*luc3*luc5 + luc13*luc6*luc9 -luc10) * luc0 * luc14;
    auto li65 = -luc19 * luc20 * luc14;
    auto li64 = (luc19 * luc13 * luc14 - luc18) * luc9 * luc20;
    auto li63 = (-luc8 * luc13 * luc19 * luc9 * luc14 +
		 luc8 * luc18 * luc9 + luc12 * luc19 * luc14 - luc17) * luc5 * luc20;
        auto li62 = (luc4*luc8*luc13*luc19*luc5*luc9*luc14 -
		 luc18*luc8*luc4*luc9*luc5 - luc19*luc12*luc4*luc14*luc5 -luc19*luc13*luc7*luc14*luc9 +
		 luc17*luc4*luc5 + luc18*luc7*luc9 + luc19*luc11*luc14 - luc16) * luc2 * luc20;
    auto li61 = (-luc19*luc13*luc8*luc4*luc1*luc2*luc5*luc9*luc14 +
		 luc18*luc8*luc4*luc1*luc2*luc5*luc9 + luc19*luc12*luc4*luc1*luc2*luc5*luc14 +
		 luc19*luc13*luc7*luc1*luc2*luc9*luc14 + luc19*luc13*luc8*luc3*luc5*luc9*luc14 -
		 luc17*luc4*luc1*luc2*luc5 - luc18*luc7*luc1*luc2*luc9 - luc19*luc11*luc1*luc2*luc14 -
		 luc18*luc8*luc3*luc5*luc9 - luc19*luc12*luc3*luc5*luc14 - luc19*luc13*luc6*luc9*luc14 +
		 luc16*luc1*luc2 + luc17*luc3*luc5 + luc18*luc6*luc9 + luc19*luc10*luc14 - luc15) * luc0 * luc20;    

    dst(0,0) = li61*li61 + li51*li51 + li41*li41 + li31*li31 + li21*li21 + luc0*luc0;
    dst(1,0) = li61*li62 + li51*li52 + li41*li42 + li31*li32 + li21*luc2;
    dst(1,1) = li62*li62 + li52*li52 + li42*li42 + li32*li32 + luc2*luc2;
    dst(2,0) = li61*li63 + li51*li53 + li41*li43 + li31*luc5;
    dst(2,1) = li62*li63 + li52*li53 + li42*li43 + li32*luc5;
    dst(2,2) = li63*li63 + li53*li53 + li43*li43 + luc5*luc5;
    dst(3,0) = li61*li64 + li51*li54 + li41*luc9;
    dst(3,1) = li62*li64 + li52*li54 + li42*luc9;
    dst(3,2) = li63*li64 + li53*li54 + li43*luc9;
    dst(3,3) = li64*li64 + li54*li54 + luc9*luc9;
    dst(4,0) = li61*li65 + li51*luc14;
    dst(4,1) = li62*li65 + li52*luc14;
    dst(4,2) = li63*li65 + li53*luc14;
    dst(4,3) = li64*li65 + li54*luc14;
    dst(4,4) = li65*li65 + luc14*luc14;
    dst(5,0) = li61*luc20;
    dst(5,1) = li62*luc20;
    dst(5,2) = li63*luc20;
    dst(5,3) = li64*luc20;
    dst(5,4) = li65*luc20;
    dst(5,5) = luc20*luc20;
      
  }

  

  template<typename M>
  inline constexpr
  void symmetrize11(M & dst) {}
  template<typename M>
  inline constexpr
  void symmetrize22(M & dst) {
    dst(0,1) = dst(1,0);
  }
  template<typename M>
  inline constexpr
  void symmetrize33(M & dst) {
    symmetrize22(dst);
    dst(0,2) = dst(2,0);
    dst(1,2) = dst(2,1);
  }
  template<typename M>
  inline constexpr
  void symmetrize44(M & dst) {
    symmetrize33(dst);
    dst(0,3) = dst(3,0);
    dst(1,3) = dst(3,1);
    dst(2,3) = dst(3,2);
  }
  template<typename M>
  inline constexpr
  void symmetrize55(M & dst) {
    symmetrize44(dst);
    dst(0,4) = dst(4,0);
    dst(1,4) = dst(4,1);
    dst(2,4) = dst(4,2);
    dst(3,4) = dst(4,3);
  }
  template<typename M>
  inline constexpr
  void symmetrize66(M & dst) {
    symmetrize55(dst);
    dst(0,5) = dst(5,0);
    dst(1,5) = dst(5,1);
    dst(2,5) = dst(5,2);
    dst(3,5) = dst(5,3);
    dst(4,5) = dst(5,4);
  }

  
  template<typename M1, typename M2, int N>
  struct Inverter {
    static constexpr void eval(M1 const & src, M2 & dst){ dst=src.inverse();}
  };
  template<typename M1, typename M2>
  struct Inverter<M1,M2,1> {
    static constexpr void eval(M1 const & src, M2 & dst) {
      invert11(src,dst);
    }
  };
  template<typename M1, typename M2>
  struct Inverter<M1,M2,2> {
    static constexpr void eval(M1 const & src, M2 & dst) {
      invert22(src,dst);
      symmetrize22(dst);
    }
  };
  template<typename M1, typename M2>
  struct Inverter<M1,M2,3> {
    static constexpr void eval(M1 const & src, M2 & dst) {
      invert33(src,dst);
      symmetrize33(dst);
    }
  };
  template<typename M1, typename M2>
  struct Inverter<M1,M2,4> {
    static constexpr void eval(M1 const & src, M2 & dst) {
      invert44(src,dst);
      symmetrize44(dst);
    }
  };
  template<typename M1, typename M2>
  struct Inverter<M1,M2,5> {
    static constexpr void eval(M1 const & src, M2 & dst) {
      invert55(src,dst);
      symmetrize55(dst);
    }
  };
  template<typename M1, typename M2>
  struct Inverter<M1,M2,6> {
    static constexpr void eval(M1 const & src, M2 & dst) {
      invert66(src,dst);
      symmetrize66(dst);
    }
  };

  // Eigen interface (MatrixBase<Derived> ??)
  template<typename M1, typename M2> 
  inline constexpr
  void invert(M1 const & src, M2 & dst) {
    Inverter<M1,M2,M2::ColsAtCompileTime>::eval(src,dst);
  }

  
}
