#include "FakeCPP.h"



int main() {
  cppUnit::Dump a;

  CPPUNIT_ASSERT(1==0);
  CPPUNIT_ASSERT(1==1);

  return 0;
}
