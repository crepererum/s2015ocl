#include <cstdlib>

#include <chrono>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <thread>

#define BACKWARD_HAS_DW 1
#include <backward.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include "common.hpp"
#include "audio.hpp"
#include "gui.hpp"


// check some assumptions made while programming
static_assert(sizeof(cl_float) == sizeof(float), "sizeof(cl_float) != sizeof(float)");
static_assert(sizeof(cl_float4) == 4 * sizeof(cl_float), "sizeof(cl_float4) != 4 * sizeof(cl_float)");
static_assert(sizeof(cl_uchar) == sizeof(unsigned char), "sizeof(cl_uchar) != sizeof(unsigned char)");
static_assert(sizeof(cl_uchar4) == 4 * sizeof(cl_uchar), "sizeof(cl_uchar4) != 4 * sizeof(cl_uchar)");


// install backward handler
namespace backward {
backward::SignalHandling sh;
}


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
    // set up logging
    auto log = spdlog::stdout_logger_mt("main");
    log->info() << "s2015ocl booting";

    // config
    std::size_t n = 16;
    std::size_t m = 4;
    std::size_t reduction_size = 1;
    std::size_t sample_rate = 44100;
    std::size_t nsamples = 1000;


    // shard host storage
    // place data on heap to avoid stack overflows
    log->info() << "allocate host memory";
    auto mGlobal = std::make_shared<std::mutex>();
    auto shutdown = std::make_shared<std::atomic<bool>>(false);
    auto hState = std::make_shared<std::vector<float>>(n * n * m, 0.f);
    auto hRules = std::make_shared<std::vector<float>>(9 * m * m + m, 0.f);
    auto hFrequencies = std::make_shared<std::vector<float>>(m, 0.f);
    auto hTexture = std::make_shared<std::vector<unsigned char>>(n * n * 4, 0);
    auto hColors = std::make_shared<std::vector<float>>(m * 4, 0.f);
    auto hBuffer = std::vector<float>(nsamples, 0.f);
    auto audiobuffer = std::make_shared<std::queue<std::vector<float>>>();

    log->info() << "prefill data";
    (*hColors)[0] = 1.f;
    (*hColors)[1] = 0.f;
    (*hColors)[2] = 0.f;
    (*hColors)[3] = 1.f;
    (*hColors)[4] = 0.f;
    (*hColors)[5] = 1.f;
    (*hColors)[6] = 0.f;
    (*hColors)[7] = 1.f;
    (*hColors)[8] = 0.f;
    (*hColors)[9] = 0.f;
    (*hColors)[10] = 1.f;
    (*hColors)[11] = 1.f;
    (*hColors)[12] = 0.5f;
    (*hColors)[13] = 0.5f;
    (*hColors)[14] = 0.5f;
    (*hColors)[15] = 1.f;
    (*hState)[0] = 1.f;

#define RIDX_OTHER(m, dx, dy, ltarget, lsource) ((m) * (m) * (((dx) + 1) + 3 * ((dy) + 1)) + (m) * (ltarget) + (lsource))
#define RIDX_BASE(m, l) ((m) * (m) * 9 + (l))

    // copy level 0 to down right
    (*hRules)[RIDX_OTHER(m, 0, -1, 0, 0)] = 0.00001f;
    (*hRules)[RIDX_OTHER(m, -1, 0, 0, 0)] = 0.00004f;
    (*hRules)[RIDX_OTHER(m, 0, 0, 0, 0)] = 1.f;

    // if there is some level 1 in my neighborhood, wipe level 0
    (*hRules)[RIDX_OTHER(m, -1, -1, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, -1, 0, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, -1, 1, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, 0, -1, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, 0, 0, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, 0, 1, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, 1, -1, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, 1, 0, 0, 1)] = -10.f;
    (*hRules)[RIDX_OTHER(m, 1, 1, 0, 1)] = -10.f;

    // create level 1 dots if there is a bunch (sum>0.8) level 0 around
    (*hRules)[RIDX_OTHER(m, -1, -1, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, -1, 0, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, -1, 1, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, 0, -1, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, 0, 0, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, 0, 1, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, 1, -1, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, 1, 0, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, 1, 1, 1, 0)] = 0.1f;
    (*hRules)[RIDX_OTHER(m, 0, 0, 1, 1)] = 100.f;
    (*hRules)[RIDX_BASE(m, 1)] = -0.8f;

    (*hFrequencies)[0] = 400.f;

    log->info() << "set up OpenCL";

    log->debug() << "get platform data";
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        log->error() << "no platforms found";
        return EXIT_FAILURE;
    }

    log->debug() << "get device data";
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        log->error() << "no devices found";
        return EXIT_FAILURE;
    }

    log->debug() << "create context";
    cl::Context context(devices);

    log->debug() << "build program";
    cl::Program programAutomaton = buildProgramFromFile("automaton.cl", context, devices);
    cl::Program programVisualize = buildProgramFromFile("visualize.cl", context, devices);
    cl::Program programRender = buildProgramFromFile("render.cl", context, devices);
    cl::Kernel kernelAutomaton(programAutomaton, "automaton");
    cl::Kernel kernelVisualize(programVisualize, "visualize");
    cl::Kernel kernelRender(programRender, "render");
    cl::Kernel kernelReduce(programRender, "reduce");

    log->debug() << "allocate buffers";
    // TODO: lock global mutex to enable running this while other components are already active
    cl::Buffer dState0(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * hState->size(), hState->data());
    cl::Buffer dState1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * hState->size(), hState->data());
    cl::Buffer dRules(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * hRules->size(), hRules->data());
    cl::Buffer dFrequencies(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * hFrequencies->size(), hFrequencies->data());
    cl::Buffer dTexture(context, CL_MEM_WRITE_ONLY, sizeof(cl_char) * hTexture->size());
    cl::Buffer dColors(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * hColors->size(), hColors->data());
    cl::Buffer dBuffer0(context, CL_MEM_READ_WRITE, sizeof(cl_float) * n * n * nsamples / reduction_size);
    cl::Buffer dBuffer1(context, CL_MEM_READ_WRITE, sizeof(cl_float) * n * n * nsamples / reduction_size);

    log->debug() << "set kernel args";
    kernelAutomaton.setArg(2, dRules);
    kernelVisualize.setArg(1, dTexture);
    kernelVisualize.setArg(2, dColors);
    kernelVisualize.setArg(3, static_cast<cl_uint>(m));
    kernelRender.setArg(1, dFrequencies);
    kernelRender.setArg(3, static_cast<cl_uint>(m));
    kernelRender.setArg(5, static_cast<cl_uint>(nsamples));
    kernelRender.setArg(6, static_cast<cl_uint>(sample_rate));
    kernelRender.setArg(7, static_cast<cl_uint>(reduction_size));
    kernelRender.setArg(8, sizeof(cl_float) * nsamples, nullptr);
    kernelReduce.setArg(2, static_cast<cl_uint>(nsamples));
    kernelReduce.setArg(3, static_cast<cl_uint>(2));
    kernelReduce.setArg(4, sizeof(cl_float) * nsamples, nullptr);

    log->debug() << "create command queue";
    cl::CommandQueue queue(context, devices[0]);

    log->info() << "spawn GUI thread";
    std::thread thread_gui(main_gui, n, m, mGlobal, hTexture, shutdown);

    log->info() << "spawn audio thread";
    std::thread thread_audio(main_audio, sample_rate, mGlobal, shutdown, audiobuffer);

    log->info() << "run kernel loop";
    bool flipflop = false;
    float t = 0.f;
    while (!(*shutdown)) {
        // check if the audiobuffer is not overfull
        bool place_in_buffer;
        {
            log->debug() << "queck audiobuffer status";
            std::lock_guard<std::mutex> guard(*mGlobal);
            place_in_buffer = audiobuffer->size() * nsamples < sample_rate * 0.5;
        }

        if (place_in_buffer) {
            log->debug() << "set kernel args";
            if (flipflop) {
                kernelAutomaton.setArg(0, dState0);
                kernelAutomaton.setArg(1, dState1);
                kernelVisualize.setArg(0, dState1);
                kernelRender.setArg(0, dState1);
            } else {
                kernelAutomaton.setArg(0, dState1);
                kernelAutomaton.setArg(1, dState0);
                kernelVisualize.setArg(0, dState0);
                kernelRender.setArg(0, dState0);
            }
            kernelRender.setArg(4, t);

            log->debug() << "run automaton kernel";
            queue.enqueueNDRangeKernel(kernelAutomaton, cl::NullRange, cl::NDRange(n, n, m));

            log->debug() << "run visualization kernel";
            queue.enqueueNDRangeKernel(kernelVisualize, cl::NullRange, cl::NDRange(n, n));

            log->debug() << "run render kernel";
            std::size_t to_reduce = n * n / reduction_size;
            bool flipflop2 = true;
            kernelRender.setArg(2, dBuffer0);
            queue.enqueueNDRangeKernel(kernelRender, cl::NullRange, cl::NDRange(to_reduce), cl::NDRange(1, 1));

            log->debug() << "run reduction kernel";
            // TODO: too slow, reimplement
            /*while (to_reduce > 16 * 8) {
                if (flipflop2) {
                    kernelReduce.setArg(0, dBuffer0);
                    kernelReduce.setArg(1, dBuffer1);
                } else {
                    kernelReduce.setArg(0, dBuffer1);
                    kernelReduce.setArg(1, dBuffer0);
                }
                std::size_t to_reduce_new = to_reduce / 2;
                queue.enqueueNDRangeKernel(kernelReduce, cl::NullRange, cl::NDRange(to_reduce_new), cl::NDRange(1, 1));

                to_reduce = to_reduce_new;
                flipflop2 = !flipflop2;
            }*/

            log->debug() << "sync with device";
            queue.finish();

            {
                log->debug() << "download visualization and rendered audio data";
                std::lock_guard<std::mutex> guard(*mGlobal);
                queue.enqueueReadBuffer(dTexture, false, 0, sizeof(char) * hTexture->size(), hTexture->data());
                if (flipflop2) {
                    queue.enqueueReadBuffer(dBuffer0, false, 0, sizeof(float) * hBuffer.size(), hBuffer.data());
                } else {
                    queue.enqueueReadBuffer(dBuffer1, false, 0, sizeof(float) * hBuffer.size(), hBuffer.data());
                }

                queue.finish();
                audiobuffer->push(hBuffer);
            }

            flipflop = !flipflop;
            t += static_cast<float>(nsamples) / static_cast<float>(sample_rate);
        } else {
            log->debug() << "audiobuffer full -> sleep";
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    log->info() << "join threads";
    thread_gui.join();
    thread_audio.join();

    log->info() << "done, goodbye!";
    return EXIT_SUCCESS;
}
