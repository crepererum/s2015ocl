#include <cstdlib>

#include <exception>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#define BACKWARD_HAS_DW 1
#include <backward.hpp>


// check some assumptions made while programming
static_assert(sizeof(cl_float) == sizeof(float), "sizeof(cl_float) == sizeof(float)");


// install backward handler
namespace backward {
backward::SignalHandling sh;
}


class MyException : public std::exception {
    public:
        MyException(const std::string& msg) : msg(msg) {}
        virtual const char* what() const throw() {
            return msg.c_str();
        }

    private:
        std::string msg;
};


cl::Program buildProgramFromFile(const std::string& fname, const cl::Context& context, const std::vector<cl::Device>& devices) {
    std::ifstream file(fname.c_str());
    if (file.fail()) {
        throw MyException("Cannot open file " + fname);
    }

    std::string sourceCode;
    file.seekg(0, std::ios::end);
    sourceCode.reserve(static_cast<std::size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    sourceCode.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    cl::Program program(context, sourceCode);
    try {
        program.build(devices);
    } catch (const cl::Error& e) {
        std::cout << "Build erros:" << std::endl;
        for (const auto& dev : devices) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            if (!buildLog.empty()) {
                std::cout << buildLog << std::endl;
            }
        }
        throw;
    }
    return program;
}

int main() {
    // config
    std::size_t n = 16;
    std::size_t m = 4;


    // host storage
    // place data on heap to avoid stack overflows
    std::cout << "Allocate host memory ..." << std::flush;
    std::vector<float> hState(n * n * m, 0.f);
    std::vector<float> hRules(9 * m, 0.f);
    std::vector<float> hFrequencies(m, 0.f);
    std::cout << "OK" << std::endl;


    std::cout << "Set up OpenCL..." << std::flush;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cout << "no platforms found" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        std::cout << "no devices found" << std::endl;
        return EXIT_FAILURE;
    }

    cl::Context context(devices);

    cl::Program programAutomaton = buildProgramFromFile("automaton.cl", context, devices);
    cl::Kernel kernelAutomaton(programAutomaton, "automaton");
    cl::Buffer dState(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * hState.size(), hState.data());
    cl::Buffer dRules(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * hRules.size(), hRules.data());
    kernelAutomaton.setArg(0, dState);
    kernelAutomaton.setArg(1, dRules);
    std::cout << "OK" << std::endl;


    return EXIT_SUCCESS;
}
