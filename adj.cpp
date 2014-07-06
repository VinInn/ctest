#include<algorithm>
#include<numeric>
#include<iterator>
#include<iostream>

struct mean {
  float operator()(float a, float b) const {
    return 0.5*(b+a);
  }
};

struct addMean {
  float operator()(float a, float m) const {
    return 2*m-a;
  }
};

int main()
{
  float A[] = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
  const int N = sizeof(A) / sizeof(int);
  float B[N];

  std::cout << "A[]:         ";
  std::copy(A, A + N, std::ostream_iterator<float>( std::cout, " "));
  std::cout << std::endl;
  
  std::copy(A, A + N,B);
  std::adjacent_difference(B, B + N, B,mean());
  std::cout << "Differences: ";
  std::copy(B, B + N,  std::ostream_iterator<float>( std::cout, " "));
  std::cout << std::endl;
  
  std::cout << A[std::upper_bound(B,B+N,19)-B] << std::endl;

  std::cout << "Reconstruct: ";
  std::partial_sum(B, B + N,  std::ostream_iterator<float>( std::cout, " "), addMean());
  std::cout << std::endl;
}
