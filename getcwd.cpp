#include <iostream>
#include <unistd.h>

int main()
{
 std::string fileName("muonNtuple.root");
 std::string fullName;
 fullName.reserve(1024);
 std::cout << fullName << std::endl;
 // fullName = getcwd(&fullName[0],1024);
 getcwd(&fullName[0],1024);
 std::cout << fullName << std::endl;
 fullName += "/" + fileName;
 std::cout << fullName << std::endl;
}

