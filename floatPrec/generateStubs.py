# !/usr/bin/env python3
import tempfile
import random
import sys
# echo acos acosh asin asinh atan atanh cbrt cos cospi cosh erf erfc exp exp10 exp2 expm1 j0 j1 log log10 log1p log2 rsqrt sin sinpi sinh tan tanpi tanh y0 y1 lgamma tgamma | sed 's/ /\" \"/g'
if __name__ == '__main__':
    print("// auto generated")
    print('extern "C"')
    print("{")
    functions = ["acos","acosh","asin","asinh","atan","atanh","cbrt","cos","cospi","cosh","erf","erfc","exp","exp10","exp2","expm1","j0","j1","log","log10","log1p","log2","rsqrt","sin","sinpi","sinh","tan","tanpi","tanh","y0","y1","lgamma","tgamma"]
#    print(functions)
    index = 14 # 0-13 reserved for sincos and 2x(atan2,hypot,pow)
    for fun in functions:
      print("")
      print("funfSym orig"+fun+"f = nullptr;")
      print("float "+fun+"f(float x) {")
      print("  if (!orig"+fun+"f) orig"+fun+'f = (funfSym)dlsym(RTLD_NEXT,"'+fun+'f");')
      print("  float ret  = orig"+fun+"f(x);")
      print("  count(x,",index,");")
      print("  return ret;")
      print("}")
      print("")
      print("fundSym orig"+fun+"d = nullptr;")
      print("double "+fun+"(double x) {")
      print("  if (!orig"+fun+"d) orig"+fun+'d = (fundSym)dlsym(RTLD_NEXT,"'+fun+'");')
      print("  double ret  = orig"+fun+"d(x);")
      print("  count(x,",index+1,");")
      print("  return ret;")
      print("}")
      print("")
      index +=2
    print("} // C")
