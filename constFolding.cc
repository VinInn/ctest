struct PFLayer {
enum Layer {PS2          = -12, 
              PS1          = -11,
              ECAL_ENDCAP  = -2,
              ECAL_BARREL  = -1,
	      NONE         = 0,
              HCAL_BARREL1 = 1,
              HCAL_BARREL2 = 2,
              HCAL_ENDCAP  = 3,
              HF_EM        = 11,
              HF_HAD       = 12,
              HGCAL        = 13  // HGCal, could be EM or HAD
  };
};


bool isBarrel(int cell_layer)
{  return (cell_layer == PFLayer::HCAL_BARREL1 ||
          cell_layer == PFLayer::HCAL_BARREL2 ||
          cell_layer == PFLayer::ECAL_BARREL);
}


struct A {
  
  void bar(float * x, float * y, int N);
  void foo(float * x, float * y, int N);

    
  bool q,p;
  
};

float sa(float) __attribute__((const));

bool qq(bool) __attribute__((const));

 void A::bar(float * x, float * y, int N) {
    for (int i=0; i<N;++i)
      if (qq(q&p)) x[i]+=y[i];
      else x[i]-=sa(y[i]);
  }
 

void A::foo(float * x, float * y, int N) {
   auto  w= qq(q&p);
    for (int i=0; i<N;++i)
      if (w) x[i]+=y[i];
      else x[i]-=sa(y[i]);
  }
 



