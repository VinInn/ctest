#ifdef __AVX__
#define NATIVE_VECT_LENGH 32
#else
#define NATIVE_VECT_LENGH 16
#endif


template<typename T>
struct NativeVectorTraits {
  typedef T __attribute__( ( vector_size(  NATIVE_VECT_LENGH ) ) ) type;
};



template<typename T> using NativeVector =  typename NativeVectorTraits<T>::type;



template<typename V> V neg(V a) { return -a;}

// template<typename T>
// typename NativeVectorTraits<T>::type neg(typename NativeVectorTraits<T>::type a) { return -a;}


// template<typename V> V  zero() { return V{0}; }


int main() {

NativeVector<float> b = {0}; b+=1.f;

auto a = neg(b);


return a[0];

}
