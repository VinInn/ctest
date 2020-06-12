// sed 's?<<<?/*?g' removeBraKet.cc |   sed 's?>>>?*/?g'      | c++ -O3 -c -x c++ -

#define __global__

  __global__
  void go(int,int,float);

void bha() {

  go<<<256,32,0,0>>>(7,5,4.3);

}
