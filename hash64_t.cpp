#include "constHash64.h"


#define mix64(a,b,c) \
{ \
  a -= b; a -= c; a ^= (c>>43); \
  b -= c; b -= a; b ^= (a<<9); \
  c -= a; c -= b; c ^= (b>>8); \
  a -= b; a -= c; a ^= (c>>38); \
  b -= c; b -= a; b ^= (a<<23); \
  c -= a; c -= b; c ^= (b>>5); \
  a -= b; a -= c; a ^= (c>>35); \
  b -= c; b -= a; b ^= (a<<49); \
  c -= a; c -= b; c ^= (b>>11); \
  a -= b; a -= c; a ^= (c>>12); \
  b -= c; b -= a; b ^= (a<<18); \
  c -= a; c -= b; c ^= (b>>22); \
}

typedef  unsigned long  long ub8;   /* unsigned 8-byte quantities */
typedef  unsigned long  int  ub4;   /* unsigned 4-byte quantities */
typedef         char ub1;

inline ub8 hashsize(ub8 n) { return (ub8)1<<(n);}
inline ub8 hashmask(ub8 n) { return hashsize(n)-1; }

#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>

inline
ub8 hash(const char * k, ub8 length, ub8 level) {
  return hash64::hash64(k, length, level);
}

/* used for timings */
void driver1()
{
  ub1 buf[256];
  ub8 i;
  ub8 h=0;

  constexpr char const * ls = "a literal string";
  constexpr char const * ls1 = " with trailer";
  constexpr char const * ls2 = "a literal string with trailer";
  constexpr ub8 hq = hash64::hash64(ls,::strlen(ls),0);
  constexpr ub8 hq1 = hash64::hash64(ls1,::strlen(ls1),0);
  constexpr ub8 hq2 = hash64::hash64(ls2,::strlen(ls2),0);
  constexpr ub8 hq3 = hash64::hash64(ls1,::strlen(ls1),hq);
  printf("%.8lx%.8lx\n", (ub4)hq, (ub4)(hq>>32));
  printf("%.8lx%.8lx\n", (ub4)hq1, (ub4)(hq1>>32));
  printf("%.8lx%.8lx\n", (ub4)hq2, (ub4)(hq2>>32));
  printf("%.8lx%.8lx\n", (ub4)hq3, (ub4)(hq3>>32));

  for (i=0; i<256; ++i) 
  {
    h = hash(buf,i,h);
    if (i%12==0) printf("%d %.8lx%.8lx\n", i, (ub4)h, (ub4)(h>>32));

 }

  ub8 a=0,b=0,c=0x9e3779b97f4a7c13LL;mix64(a,b,c);
  printf("%.8lx%.8lx %.8lx%.8lx %.8lx%.8lx\n", (ub4)a, (ub4)(a>>32), (ub4)b, (ub4)(b>>32), (ub4)c, (ub4)(c>>32));

  hash64::w iw = hash64::mix(hash64::w(0,0,0x9e3779b97f4a7c13LL));
  printf("%.8lx%.8lx %.8lx%.8lx %.8lx%.8lx\n", (ub4)iw.a, (ub4)(iw.a>>32), (ub4)iw.b, (ub4)(iw.b>>32), (ub4)iw.c, (ub4)(iw.c>>32));
}

/* check that every input bit changes every output bit half the time */
#define HASHSTATE 1
#define HASHLEN   1
#define MAXPAIR 80
#define MAXLEN 5
void driver2()
{
  ub1 qa[MAXLEN+1], qb[MAXLEN+2], *a = &qa[0], *b = &qb[1];
  ub8 c[HASHSTATE], d[HASHSTATE], i=0, j=0, k=0, l=0, m=0, z=0;
  ub8 e[HASHSTATE],f[HASHSTATE],g[HASHSTATE],h[HASHSTATE];
  ub8 x[HASHSTATE],y[HASHSTATE];
  ub8 hlen=0;

  printf("No more than %d trials should ever be needed \n",MAXPAIR/2);
  for (hlen=0; hlen < MAXLEN; ++hlen)
  {
    z=0;
    for (i=0; i<hlen; ++i)  /*----------------------- for each byte, */
    {
      for (j=0; j<8; ++j)   /*------------------------ for each bit, */
      {
	for (m=0; m<8; ++m) /*-------- for serveral possible levels, */
	{
	  for (l=0; l<HASHSTATE; ++l) e[l]=f[l]=g[l]=h[l]=x[l]=y[l]=~((ub8)0);

      	  /*---- check that every input bit affects every output bit */
	  for (k=0; k<MAXPAIR; k+=2)
	  { 
	    ub8 finished=1;
	    /* keys have one bit different */
	    for (l=0; l<hlen+1; ++l) {a[l] = b[l] = (ub1)0;}
	    /* have a and b be two keys differing in only one bit */
	    a[i] ^= (k<<j);
	    a[i] ^= (k>>(8-j));
	     c[0] = hash(a, hlen, m);
	    b[i] ^= ((k+1)<<j);
	    b[i] ^= ((k+1)>>(8-j));
	     d[0] = hash(b, hlen, m);
	    /* check every bit is 1, 0, set, and not set at least once */
	    for (l=0; l<HASHSTATE; ++l)
	    {
	      e[l] &= (c[l]^d[l]);
	      f[l] &= ~(c[l]^d[l]);
	      g[l] &= c[l];
	      h[l] &= ~c[l];
	      x[l] &= d[l];
	      y[l] &= ~d[l];
	      if (e[l]|f[l]|g[l]|h[l]|x[l]|y[l]) finished=0;
	    }
	    if (finished) break;
	  }
	  if (k>z) z=k;
	  if (k==MAXPAIR) 
	  {
	     printf("Some bit didn't change: ");
	     printf("%.8lx %.8lx %.8lx %.8lx %.8lx %.8lx  ",
	            (unsigned long)e[0],(unsigned long)f[0],(unsigned long)g[0],(unsigned long)h[0],(unsigned long)x[0],(unsigned long)y[0]);
	     printf("i %ld j %ld m %ld len %ld\n",
	            (ub4)i,(ub4)j,(ub4)m,(ub4)hlen);
	  }
	  if (z==MAXPAIR) goto done;
	}
      }
    }
   done:
    if (z < MAXPAIR)
    {
      printf("Mix success  %2ld bytes  %2ld levels  ",(ub4)i,(ub4)m);
      printf("required  %ld  trials\n",(ub4)(z/2));
    }
  }
  printf("\n");
}

/* Check for reading beyond the end of the buffer and alignment problems */
void driver3()
{
  ub1 buf[MAXLEN+20], *b;
  ub8 len;
  ub1 q[] = "This is the time for all good men to come to the aid of their country";
  ub1 qq[] = "xThis is the time for all good men to come to the aid of their country";
  ub1 qqq[] = "xxThis is the time for all good men to come to the aid of their country";
  ub1 qqqq[] = "xxxThis is the time for all good men to come to the aid of their country";
  ub1 o[] = "xxxxThis is the time for all good men to come to the aid of their country";
  ub1 oo[] = "xxxxxThis is the time for all good men to come to the aid of their country";
  ub1 ooo[] = "xxxxxxThis is the time for all good men to come to the aid of their country";
  ub1 oooo[] = "xxxxxxxThis is the time for all good men to come to the aid of their country";
  ub8 h,i,j,ref,x,y;

  printf("Endianness.  These should all be the same:\n");
  h = hash(q+0, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  h = hash(qq+1, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  h = hash(qqq+2, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  h = hash(qqqq+3, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  h = hash(o+4, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  h = hash(oo+5, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  h = hash(ooo+6, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  h = hash(oooo+7, (ub8)(sizeof(q)-1), (ub8)0);
  printf("%.8lx%.8lx\n", (ub4)h, (ub4)(h>>32));
  printf("\n");
  for (h=0, b=buf+1; h<8; ++h, ++b)
  {
    for (i=0; i<MAXLEN; ++i)
    {
      len = i;
      for (j=0; j<i; ++j) *(b+j)=0;

      /* these should all be equal */
      ref = hash(b, len, (ub8)1);
      *(b+i)=(ub1)~0;
      *(b-1)=(ub1)~0;
      x = hash(b, len, (ub8)1);
      y = hash(b, len, (ub8)1);
      if ((ref != x) || (ref != y)) 
      {
	printf("alignment error: %.8lx %.8lx %.8lx %ld %ld\n",(unsigned long)ref,(unsigned long)x,(unsigned long)y,(long)h,(long)i);
      }
    }
  }
}

/* check for problems with nulls */
 void driver4()
{
  ub1 buf[1];
  ub8 h,i,state[HASHSTATE];


  buf[0] = ~0;
  for (i=0; i<HASHSTATE; ++i) state[i] = 1;
  printf("These should all be different\n");
  for (i=0, h=0; i<8; ++i)
  {
    h = hash(buf, (ub8)0, h);
    printf("%2ld  0-byte strings, hash is  %.8lx%.8lx\n", (ub4)i,
      (ub4)h,(ub4)(h>>32));
  }
}


int main()
{



  driver1();   /* test that the key is hashed: used for timings */
  driver2();   /* test that whole key is hashed thoroughly */
  driver3();   /* test that nothing but the key is hashed */
  driver4();   /* test hashing multiple buffers (all buffers are null) */
  return 0;
}


