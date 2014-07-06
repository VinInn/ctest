/* This is a demo for STT_GNU_IFUNC.  See ifunc.txt at

   http://groups.google.com/group/generic-abi
 */

#include <stdio.h>
#include <cpuid.h>

class cpuid
{
private:
  static void x86 (void)
    {
      printf ("I am an x86\n");
    }

  static void sse (void)
    {
      printf ("I have SSE\n");
    }

  static void sse2 (void)
    {
      printf ("I have SSE2\n");
    }

  static void sse3 (void)
    {
      printf ("I have SSE3\n");
    }

  static void ssse3 (void)
    {
      printf ("I have SSSE3\n");
    }

  static void sse4_1 (void)
    {
      printf ("I have SSE4.1\n");
    }

  static void sse4_2 (void)
    {
      printf ("I have SSE4.2\n");
    }

public:
  /* feature is an indirec function.  */
  static void __attribute__ ((ifunc)) feature (void)
    {
      unsigned int eax, ebx, ecx, edx;

      __cpuid (1, eax, ebx, ecx, edx);

      if (ecx & bit_SSE4_2)
	return cpuid::sse4_2;
      if (ecx & bit_SSE4_1)
	return cpuid::sse4_1;
      if (ecx & bit_SSSE3)
	return cpuid::ssse3;
      if (ecx & bit_SSE3)
	return cpuid::sse3;
      if (edx & bit_SSE2)
	return cpuid::sse2;
      if (edx & bit_SSE)
	return cpuid::sse;
      return cpuid::x86;
    }
};

class long_mode
{
private:
  void lm_yes (void)
    {
      printf ("I support 64bit.\n");
    }

  void lm_no (void)
    {
      printf ("I don't support 64bit.\n");
    }

public:
  /* lm is an indirec function.  */
  void __attribute__ ((ifunc)) lm (void)
    {
      unsigned int eax, ebx, ecx, edx;

      __cpuid (0x80000001, eax, ebx, ecx, edx);

      if (edx & bit_LM)
	return &long_mode::lm_yes;
      else
	return &long_mode::lm_no;
    }
};

class Foo
{
private:
  virtual void foo1 ()
    {
      printf ("I am %s\n", __PRETTY_FUNCTION__);
    }
public:
  virtual void  __attribute__ ((ifunc)) foo ()
    {
      return &Foo::foo1;
    }
};

class Bar : public Foo
{
private:
  void foo1 ()
    {
      printf ("I am %s\n", __PRETTY_FUNCTION__);
    }
public:
  void foo ()
    {
      printf ("I am %s\n", __PRETTY_FUNCTION__);
    }
};

class X
{
private:
  virtual void foo1 ()
    {
      printf ("I am %s\n", __PRETTY_FUNCTION__);
    }
public:
  virtual void foo ();
};

void
__attribute__ ((ifunc))
X::foo ()
{
  return &X::foo1;
}

class Y : public X
{
private:
  void foo1 ()
    {
      printf ("I am %s\n", __PRETTY_FUNCTION__);
    }
public:
  void foo ()
    {
      printf ("I am %s\n", __PRETTY_FUNCTION__);
    }
};

int
main ()
{
  int i;
  class cpuid c;
  class long_mode l;

  /* Run it under gdb to see how feature and lm are called.  */
  for (i = 0; i < 5; i++)
    {
      c.feature ();
      l.lm ();
    }

  Foo f;
  f.foo ();

  Bar b;
  b.foo ();

  X x;
  x.foo ();

  Y y;
  y.foo ();

  return 0;
}
