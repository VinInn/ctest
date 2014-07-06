struct V {
  float x,y,z,w;
};

V a;
V b;

float dot() {
  return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
}

float dot2() {
  V v=a;
  v.x*=b.x; v.y*=b.y; v.z*=b.z; v.w*=b.w;
  return v.x+v.y+v.z+v.w;
}




V sum() {
  V v=a;
  v.x+=b.x; v.y+=b.y; v.z+=b.z; v.w+=b.w;
  return v; 
}

V prod() {
  V v=a;
  v.x*=b.x; v.y*=b.y; v.z*=b.z; v.w*=b.w;
  return v;
}

struct W {
  float x[4];
};

W wa;
W wb;

float dotW() {
  float sum=0;
  for (int i=0; i!=4; ++i)
  sum += wa.x[i]*wb.x[i];
  return sum;
}

W prodW() {
  W v;
  for (int i=0; i!=4; ++i)
  v.x[i]= wa.x[i]*wb.x[i];
  return v;
}


