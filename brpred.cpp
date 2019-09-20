#include <algorithm>
#include <vector>
#include <iostream>

int main( int argc, char**)
{
    // generate data
    const size_t arraySize = 32768;
    std::vector<int> test(arraySize);
    std::vector<int> data(arraySize);

    for (unsigned c = 0; c < arraySize; ++c) {
        test[c] = std::rand() % 256;
        data[c] = std::rand() % 256;
    }

    // If the data are sorted like shown here the program runs about
   // 6x faster (on my test machine, with -O2)
   if (argc>1)  std::sort(test.begin(), test.end());

    long long sum = 0;

    for (unsigned i = 0; i < 100000; ++i)
    {
        for (unsigned c = 0; c < arraySize; ++c)
        {
            if (test[c] >= 128)
                sum += data[c];
        }
    }
    std::cout << "sum = " << sum << ' ' << argc << std::endl;

    return 0;
}
