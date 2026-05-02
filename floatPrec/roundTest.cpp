#include <cstdio>
#include <cmath>

int main() {

  float two = 2.f;
  auto twon = std::nextafter(two,4.f);
  auto ulp = twon - two;
  auto ulp2 = ulp*0.5f;
  auto ulp4 = ulp*0.25f;
  auto ulp8 = ulp2*0.25f;
  printf("%a %a\n",two,twon);
  printf("%a %a %a %a %a\n",ulp,ulp4,ulp+ulp2,ulp+ulp4,ulp+ulp8);
  printf("%a %a\n",two+ulp2,twon+ulp2);
  printf("%a %a\n",two+(ulp2+ulp4),twon+(ulp2+ulp4));
  printf("%a %a\n",two+(ulp2+ulp8),twon+(ulp2+ulp8));

  return 0;

}
