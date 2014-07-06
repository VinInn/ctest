inline float bar(float x) {
    
 float y = x-2.f;
   
 constexpr float zero_threshold_ftz =-float(0x5.75628p4);

 y = (x<zero_threshold_ftz) ?  0.f : y;
 return y;

};

 float a[1024], b[1024];
  void foo() {
    for (int i=0; i!=1024; ++i) {
      b[i] = bar(a[i]);
    }
  }

