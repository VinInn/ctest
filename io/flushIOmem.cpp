#include "fcntl.h"
#include<cstdio>
#include <iostream>

int main(int n, char * pars[]) {

if (n<2) {
    std::cout << "please provide a file name" << std::endl;
    return 1;
}

   int fd = open(pars[1],0);

   if (fd<0) return 1;

   posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

return 0;

}

