#include<iostream>
int main(int argv,char** argc) {

std::cout << argv << ' ' << argc[0] << std::endl;

float a[argv-1];

if (argc[0][0]!= 'a') a[0]=1;
else a[0]=1.2;
if (argc[0][1]!= 'a') a[1]=1;
else a[1]=1.2;

std::cout << a << std::endl;

std::cout << a[0] << std::endl;


if (argc[0][0]!= 'c') return a[0];

  return a[1];
}
