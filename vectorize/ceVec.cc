typedef float __attribute__( ( vector_size( 4*sizeof(float) ) ) ) V4;
constexpr V4 build(float x,float y, float z) { return (V4){x,y,z,0};}
constexpr V4 x = build(1,0,0);

constexpr V4 w = {1,1,1,0};
constexpr V4 m[3] = { (V4){1,0,0,0}, (V4){0,1,0,0}, (V4){0,0,1,0}};
// constexpr float xx = w[0];
// constexpr V4 r = w[0]*m[0] +  w[1]*m[1] + w[2]*m[2];


V4  build(float x) { 
  constexpr V4 zero{0,0,0,0};
  return x+zero;
}

constexpr float a[4] = {1,1,1,0};
constexpr float b = a[0];

struct Rot3 {
  typedef float T;
  typedef V4 Vec;
  Vec  axis[3];
  constexpr Rot3( V4 ix,  V4 iy,  V4 iz) :
    axis{ix,iy,iz}{}

  constexpr Rot3(T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) :
    Rot3( (Vec){xx,xy,xz,0},
	  (Vec){yx,yy,yz,0},
	  (Vec){zx,zy,zz,0}
	  ){}

  constexpr Rot3 transpose() const {
    return Rot3( axis[0][0], axis[1][0], axis[2][0],
		 axis[0][1], axis[1][1], axis[2][1],
		 axis[0][2], axis[1][2], axis[2][2]
		 );
  }
  
  constexpr V4 rotateBack(V4 v) const {
    return v[0]*axis[0] +  v[1]*axis[1] + v[2]*axis[2];
  }

    constexpr V4 rotate(V4 v) const {
      return transpose().rotateBack(v);
    }

  
};

constexpr V4  v = {1,1,1,1};
constexpr Rot3 r2( (V4){0, 1 ,0,0}, (V4){0, 0, 1,0},(V4){1, 0, 0,0});
constexpr Rot3 r1( 0, 1 ,0, 0, 0, 1,  1, 0, 0);
constexpr Rot3 r3 = r1.transpose();
constexpr V4  v1 = {1,1,1,1};
constexpr V4  v2 = r1.rotateBack(v1);
constexpr V4  v3 = r1.rotate(v1);
