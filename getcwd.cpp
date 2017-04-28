#include <iostream>
#include <unistd.h>

int main()
{
 std::string fileName("muonNtuple.root");
 std::string fullName(1024,'\0');
 std::cout << fullName << std::endl;
 getcwd(&fullName[0],1024);
 std::cout << fullName << std::endl;
 fullName += "/" + fileName;
 std::cout << fullName << std::endl;

 char buff[1024];
 buff = getcwd(buff,1024);
std::cout << buff << std::endl;

}

