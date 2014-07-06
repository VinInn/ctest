
struct entry {
  const char * label;
  int index;
};

constexpr entry theMap[] = {
  {"a", 0},
  {"b", 1},
  {nullptr,2}
};

// filled at run time 
double v[3];


constexpr bool  same(char const *x, char const *y)   {
  return !*x && !*y ? true : (*x == *y && same(x+1, y+1));
}

constexpr int getIndex(char const *label, entry const *entries)   {
  return !entries->label ? entries->index  : same(entries->label, label) ? entries->index : getIndex(label, entries+1);
}


inline  double __attribute__((always_inline)) getV(const char * name )  {
  return  v[getIndex(name,theMap)];
}

  

#define SetV(X,NAME) \
 constexpr int i_##X = getIndex(NAME, theMap);\
 const double X = v[i_##X]


int foo() {
  const double a = getV("a");
  const double b = getV("b");

  if (a==b) return 1;
  return 0;

}

int foo2() {
  const double a = v[getIndex("a",theMap)];
  const double b = v[getIndex("b",theMap)];

  if (a==b) return 1;
  return 0;

}


int bar() {
  SetV(a,"a");
  SetV(b,"b");

  if (a==b) return 1;
  return 0;

}
