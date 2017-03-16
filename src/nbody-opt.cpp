#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <random>

#define POINTS (500 * 64)
#define SPACE (1000.0f)
#define BINS_PER_DIM (10)

#define DEBUG_PRINT(str, ...) /**/
//#define DEBUG_PRINT(str, ...) printf(str, ##__VA_ARGS__)

#define ASSERT(x, str, ...) \
{ \
    if ((x) == 0) \
    { \
        printf("**Assertion Error in function [%s] in file [%s:%d]: " str, __FUNCTION__, \
            __FILE__, __LINE__, ##__VA_ARGS__); \
        exit(0); \
    } \
}

cl_float4 * initializePositions ()
{
    int i;
    cl_float4 * pts;

    //
    // For deterministic results;
    //
    srand(42L);

    pts = (cl_float4 *) malloc(sizeof(cl_float4) * POINTS);
    ASSERT(pts, "PTR NOT VALID\n");


    for (i = 0; i < POINTS; ++i)
    {
        pts[i].x = (((float) std::rand()) / RAND_MAX) * SPACE;
        pts[i].y = (((float) std::rand()) / RAND_MAX) * SPACE;
        pts[i].z = (((float) std::rand()) / RAND_MAX) * SPACE;

        //
        // Size is 1.0f for simplicity.
        //
        pts[i].w = 1.0f;
    }

    return pts;
}

cl_float4 * initializeAccelerations ()
{
    cl_float4 * pts;

    pts = (cl_float4 *) malloc(POINTS * sizeof(cl_float4));
    ASSERT(pts, "PTR NOT VALID\n");

    return pts;
}

void calculate_nbody (
    cl::CommandQueue &queue,
    cl::Kernel &nbody_kernel,
    cl::Buffer &x_buffer,
    cl::Buffer &cm_buffer,
    cl::Buffer &bin_pts_buffer,
    cl::Buffer &bin_pts_offsets_buffer,
    cl::Buffer &a_buffer,
    cl::Buffer &points_buffer,
    cl_float4 * a
    )
{
    cl_int err;

    //
    // Set Args
    //
    DEBUG_PRINT("Set nbody_kernel args\n");
    err = nbody_kernel.setArg(0, x_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = nbody_kernel.setArg(1, cm_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = nbody_kernel.setArg(2, bin_pts_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = nbody_kernel.setArg(3, bin_pts_offsets_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = nbody_kernel.setArg(4, a_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = nbody_kernel.setArg(5, points_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Run Kernel
    //
    DEBUG_PRINT("Run nbody_kernel\n");
    err = queue.enqueueNDRangeKernel(nbody_kernel, cl::NDRange(0), cl::NDRange(POINTS), cl::NullRange);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Read buffer(s)
    //
    DEBUG_PRINT("Read buffers after nbody_kernel\n");
    err = queue.enqueueReadBuffer(a_buffer, CL_TRUE, 0, POINTS * sizeof(cl_float4), a);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);
}

void calculate_bins_cm (
    cl::CommandQueue &queue,
    cl::Kernel &calculate_bins_cm_kernel,
    cl::Buffer &cm_buffer,
    cl::Buffer &x_buffer,
    cl::Buffer &points_buffer
    )
{
    cl_int err;

    //
    // Set arguments to kernel
    //
    DEBUG_PRINT("Set args for calculate_bins_cm kernel\n");
    err = calculate_bins_cm_kernel.setArg(0, cm_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = calculate_bins_cm_kernel.setArg(1, x_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = calculate_bins_cm_kernel.setArg(2, points_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Run the nbody_kernel on specific ND range
    //
    DEBUG_PRINT("Run calculate_bins_cm_kernel\n");
    err = queue.enqueueNDRangeKernel(calculate_bins_cm_kernel, cl::NDRange(0, 0, 0), cl::NDRange(BINS_PER_DIM, BINS_PER_DIM, BINS_PER_DIM), cl::NullRange);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);
}

void construct_bin_pts (
    cl::CommandQueue &queue,
    cl::Kernel &construct_bin_pts_kernel,
    cl::Buffer &bin_pts_buffer,
    cl::Buffer &bin_pts_offsets_buffer,
    cl::Buffer &x_buffer,
    cl::Buffer &points_buffer,
    cl::Buffer &cm_buffer
    )
{
    cl_int err;

    //
    // Set arguments to kernel
    //
    DEBUG_PRINT("Set args for construct bin pts kernel\n");
    err = construct_bin_pts_kernel.setArg(0, bin_pts_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = construct_bin_pts_kernel.setArg(1, bin_pts_offsets_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = construct_bin_pts_kernel.setArg(2, x_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = construct_bin_pts_kernel.setArg(3, points_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = construct_bin_pts_kernel.setArg(4, cm_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Run the nbody_kernel on specific ND range
    //
    DEBUG_PRINT("Run construct_bin_pts_kernel\n");
    err = queue.enqueueNDRangeKernel(construct_bin_pts_kernel, cl::NDRange(0, 0, 0), cl::NDRange(BINS_PER_DIM, BINS_PER_DIM, BINS_PER_DIM), cl::NullRange);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);
}

int main() {
    try {
    // Get available platforms
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    // Select the default platform and create a context using this platform and the GPU
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(),
        0
    };
    cl::Context context(CL_DEVICE_TYPE_GPU, cps);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // Create a command queue and use the first device
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

    // Read source file
    std::ifstream sourceFile("src/nbody_kernel-opt.cl");

    if(!sourceFile.is_open()) {
        std::cerr << "Cannot find kernel file" << std::endl;
        throw;
    }

    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

    // Make program of the source code in the context
    cl::Program program = cl::Program(context, source);

    // Build program for these specific devices
    try {
        program.build(devices);
    } catch(cl::Error error) {
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        throw;
    }

    // Make kernel
    cl::Kernel nbody_kernel(program, "nbody");
    cl::Kernel calculate_bins_cm_kernel(program, "calculate_bins_cm");
    cl::Kernel construct_bin_pts_kernel(program, "construct_bin_pts");

    // Create buffers
    cl_int err = 0;

    DEBUG_PRINT("Create buffers\n");
    cl_float4 * x = initializePositions();
    cl_float4 * a = initializeAccelerations();
    int points = POINTS;

    //
    // Buffer for positions array
    //
    cl::Buffer x_buffer(context, CL_MEM_READ_ONLY, POINTS * sizeof(cl_float4), &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Buffer for acceleration array
    //
    cl::Buffer a_buffer(context, CL_MEM_WRITE_ONLY, POINTS * sizeof(cl_float4), &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Buffer for points value (need to have all values in a buffer)
    //
    cl::Buffer points_buffer(context, CL_MEM_READ_ONLY, sizeof(int), &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Buffer for center of masses for bins
    //
    cl::Buffer cm_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * BINS_PER_DIM * BINS_PER_DIM * BINS_PER_DIM, &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Buffer for bin pts
    //
    cl::Buffer bin_pts_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * POINTS, &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Buffer for bin pts offsets for each bin
    //
    cl::Buffer bin_pts_offsets_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * BINS_PER_DIM * BINS_PER_DIM * BINS_PER_DIM, &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    // Write buffers
    DEBUG_PRINT("Write buffers\n");
    err = queue.enqueueWriteBuffer(x_buffer, CL_TRUE, 0, POINTS * sizeof(cl_float4), x);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = queue.enqueueWriteBuffer(points_buffer,  CL_TRUE, 0, sizeof(int), &points);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Set args, run kernel and read buffers
    //
    calculate_bins_cm(queue, calculate_bins_cm_kernel, cm_buffer, x_buffer, points_buffer);
    construct_bin_pts(queue, construct_bin_pts_kernel, bin_pts_buffer, bin_pts_offsets_buffer, x_buffer, points_buffer, cm_buffer);
    calculate_nbody(queue, nbody_kernel, x_buffer, cm_buffer, bin_pts_buffer, bin_pts_offsets_buffer, a_buffer, points_buffer, a);

    for (int i = 0; i < POINTS; ++i)
    {
        printf("(%2.2f,%2.2f,%2.2f,%2.2f) (%2.3f,%2.3f,%2.3f)\n",
           x[i].x, x[i].y, x[i].z, x[i].w,
           a[i].x, a[i].y, a[i].z);
    }

    free(x);
    free(a);

    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
    }

    return EXIT_SUCCESS;
}
