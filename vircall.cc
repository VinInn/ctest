struct B {
 virtual ~B(){}
 float d() const;
 virtual float vd() const;
 virtual float vd2() const;

  float i;
};

struct D : public B {

 virtual float vd() const;

};


B b[1024];
D d[1024];

B * bs[1024];


float sumB() {
  float s=0;
  for (int i=0; i!=1024; ++i)
    s+=b[i].vd()+b[i].d();
    return s;
}


float sumBS() {
  float s=0;
  for (int i=0; i!=1024; ++i)
    s+=bs[i]->vd()+bs[i]->d();
  return s;
}



