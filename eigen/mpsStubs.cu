#include <Eigen/Dense>

    constexpr int SampleVectorSize = 10;
    constexpr int FullSampleVectorSize = 19;
    constexpr int PulseVectorSize = 12;
    constexpr int NGains = 3;

    using data_type = double;

    typedef Eigen::Matrix<data_type, SampleVectorSize, 1> SampleVector;
    typedef Eigen::Matrix<data_type, SampleVectorSize, SampleVectorSize> SampleMatrix;
    typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic, 0, PulseVectorSize, PulseVectorSize> PulseMatrix;

    typedef Eigen::LLT<SampleMatrix> SampleDecompLLT;
    typedef Eigen::LLT<PulseMatrix> PulseDecompLLT;
    typedef Eigen::LDLT<PulseMatrix> PulseDecompLDLT;

__device__ bool inplace_fnnls(SampleMatrix const& A,
                              SampleVector const& b,
                              PulseMatrix & pulse_matrix) {

}
