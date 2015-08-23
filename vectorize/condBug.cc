float v0[1024];
float v1[1024];
float v2[1024];
float v3[1024];

void condAssign1() {
  for(int i=0; i<1024; ++i)
    v0[i] = (v2[i]>v1[i]) ? v2[i]*v3[i] : v0[i];
}


void condAssign2() {
  for(int i=0; i<1024; ++i)
    v0[i] = (v2[i]>v1[i]) ? v2[i]*v1[i] : v0[i];
}


