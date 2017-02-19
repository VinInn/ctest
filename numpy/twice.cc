#include<iostream>
extern "C" {
  void twice(const float * vi, float * vo, long n) { 
    for (int i=0;i<n;++i) { vo[i]=2.f*vi[i]; }
  }
}
