#include <random>
#include "Xoshiro.h"
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstddef>

extern "C" {
    #include "TestU01.h"
}

// GENERATOR SECTION
//
// This is the only part of the code that needs to change to switch to
// a new gnerator.

const char* gen_name = "XoshiroPP jump"; // TestU01 doesn't like colons!!?!

const int MAX_SEEDS = 1;
uint64_t seed_data[MAX_SEEDS];

uint64_t gen64()
{
    static XoshiroPP  rng(seed_data[0]);

    return rng();
}

// END OF GENERATOR SECTION

inline uint32_t rev32(uint32_t v)
{
    // https://graphics.stanford.edu/~seander/bithacks.html
    // swap odd and even bits
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    // swap consecutive pairs
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    // swap nibbles ...
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    // swap bytes
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    // swap 2-byte-long pairs
    v = ( v >> 16             ) | ( v               << 16);
    return v;
}

inline uint32_t high32(uint64_t v)
{
    return uint32_t(v >> 32);
}

inline uint32_t low32(uint64_t v)
{
    return uint32_t(v);
}

uint32_t gen32_high()
{
    return high32(gen64());
}

uint32_t gen32_high_rev()
{
    return rev32(high32(gen64()));
}

uint32_t gen32_low()
{
    return low32(gen64());
}

uint32_t gen32_low_rev()
{
    return rev32(low32(gen64()));
}

const char* progname;

void usage()
{
    printf("%s: [-v] [-r] [seeds...]\n", progname);
    exit(1);
}

int main (int argc, char** argv)
{
    progname = argv[0];

    // Config options for TestU01
    swrite_Basic = FALSE;  // Turn of TestU01 verbosity by default
                           // reenable by -v option.

    // Config options for generator output
    bool reverseBits = false;
    bool highBits = false;

    // Config options for tests
    bool testSmallCrush = false;
    bool testCrush = false;
    bool testBigCrush = false;
    bool testLinComp = false;

    // Handle command-line option switches
    while (1) {
        --argc; ++argv;
        if ((argc == 0) || (argv[0][0] != '-'))
            break;
        if ((argv[0][1]=='\0') || (argv[0][2]!='\0'))
            usage();
        switch(argv[0][1]) {
        case 'r':
            reverseBits = true;
            break;      
        case 'h':
            highBits = true;
            break;      
        case 's':
            testSmallCrush = true;
            break;
        case 'm':
            testCrush = true;
            break;
        case 'b':
            testBigCrush = true;
            break;
        case 'l':
            testLinComp = true;
            break;
        case 'v':
            swrite_Basic = TRUE;
            break;
        default:
            usage();
        }
    }

    // Name of the generator

    std::string genName = gen_name;
    genName += highBits ? " [High bits]" : " [Low bits]";
    if (reverseBits)
        genName += " [Reversed]";

    // Determine a default test if need be

    if (!(testSmallCrush || testCrush || testBigCrush || testLinComp)) {
        testCrush = true;
    }

    // Initialize seed-data array, either using command-line arguments
    // or std::random_device.

    printf("Testing %s:\n", genName.c_str());
    printf("- seed_data[%u] = { ", MAX_SEEDS);
    std::random_device rdev;
    for (int i = 0; i < MAX_SEEDS; ++i) {
        if (argc >= i+1) {
            seed_data[i] = strtoull(argv[i],0,0);
        } else {
            seed_data[i] = (uint64_t(rdev()) << 32) | rdev();
        }
        printf("%s0x%016lx", i == 0 ? "" : ", ", seed_data[i]);
    }
    printf("}\n");
    fflush(stdout);

    // Create a generator for TestU01.

    unif01_Gen* gen =
        unif01_CreateExternGenBits((char*) genName.c_str(),
          reverseBits ? (highBits ? gen32_high_rev : gen32_low_rev)
                      : (highBits ? gen32_high     : gen32_low));

    // Run tests.

    if (testSmallCrush) {
        bbattery_SmallCrush(gen);
        fflush(stdout);
    }
    if (testCrush) {
        bbattery_Crush(gen);
        fflush(stdout);
    }
    if (testBigCrush) {
        bbattery_BigCrush(gen);
        fflush(stdout);
    }
    if (testLinComp) {
        scomp_Res* res = scomp_CreateRes();
        swrite_Basic = TRUE;
        for (int size : {250, 500, 1000, 5000, 25000, 50000, 75000})
            scomp_LinearComp(gen, res, 1, size, 0, 1);
        scomp_DeleteRes(res);
        fflush(stdout);
    }

    // Clean up.

    unif01_DeleteExternGenBits(gen);

    return 0;
}

