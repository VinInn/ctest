typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
float a = 3.14f;
float32x4_t va; 
float sa;
void bar() {
  va = (va==va) ? a : va;
}
void foo() {
  sa = (sa==sa) ? a : sa;
}

