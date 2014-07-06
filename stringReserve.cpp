#include <cassert> 

#include <string>
#include <vector>
#include <iostream>
int main()
{
{
std::string foo,bar;
foo.reserve(100);
std::cout << foo.capacity() << std::endl;
std::cerr << std::hex << (size_t) foo.c_str() << std::endl;
foo.reserve(10);
std::cout << foo.capacity()<< std::endl;
std::cerr << (size_t) foo.c_str() << std::endl;
bar.reserve(100);
foo.reserve(50);
std::cout << foo.capacity() << std::endl;
std::cerr << (size_t) foo.c_str() << std::endl;
}

{
std::vector<char> foo,bar;
foo.reserve(100);
std::cout << foo.capacity() << std::endl;
std::cerr << std::hex << (size_t) (&foo.front()) << std::endl;
foo.reserve(10);
std::cout << foo.capacity() << std::endl;
std::cerr << (size_t) (&foo.front()) << std::endl;
bar.reserve(100);
foo.reserve(50);
std::cout << foo.capacity() << std::endl;
std::cerr << (size_t) (&foo.front()) << std::endl;
}

}
