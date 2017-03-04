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

#define DEBUG_PRINT(str, ...) /**/
//#define DEBUG_PRINT(str, ...) printf(str, ##__VA_ARGS__)

#define ASSERT(x, str, ...) \
{ \
    if ((x) == 0) \
    { \
        printf(str, ##__VA_ARGS__); \
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
    int i;
    cl_float4 * pts;

    /*
    pts = (cl_float4 *) malloc(sizeof(cl_float4) * POINTS);

    for (i = 0; i < POINTS; ++i)
    {
        pts[i].x = 0;
        pts[i].y = 0;
        pts[i].z = 0;
        pts[i].w = 0;
    }
    */

    pts = (cl_float4 *) calloc(POINTS, sizeof(cl_float4));

    return pts;
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
    std::ifstream sourceFile("src/nbody_kernel.cl");

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
    cl::Kernel kernel(program, "nbody");

    // Create buffers
    cl_int err = 0;

    DEBUG_PRINT("Create buffers\n");
    cl_float4 * x = initializePositions();
    cl_float4 * a = initializeAccelerations();
    int points = POINTS;

    //
    // Buffer for positions array
    //
    cl::Buffer x_buffer(context, CL_MEM_READ_WRITE, POINTS * sizeof(cl_float4), &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Buffer for acceleration array
    //
    cl::Buffer a_buffer(context, CL_MEM_READ_WRITE, POINTS * sizeof(cl_float4), &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    //
    // Buffer for points value (need to have all values in a buffer)
    //
    cl::Buffer points_buffer(context, CL_MEM_READ_ONLY, sizeof(int), &err);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    // Write buffers
    DEBUG_PRINT("Write buffers\n");
    err = queue.enqueueWriteBuffer(x_buffer, CL_TRUE, 0, POINTS * sizeof(cl_float4), x);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = queue.enqueueWriteBuffer(a_buffer, CL_TRUE, 0, POINTS * sizeof(cl_float4), a);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = queue.enqueueWriteBuffer(points_buffer,  CL_TRUE, 0, sizeof(int), &points);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    // Set arguments to kernel
    DEBUG_PRINT("Set args\n");
    err = kernel.setArg(0, x_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = kernel.setArg(1, a_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = kernel.setArg(2, points_buffer);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    // Run the kernel on specific ND range
    DEBUG_PRINT("Run\n");
    err = queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(POINTS), cl::NullRange);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    // Read buffer(s)
    DEBUG_PRINT("Read buffers\n");
    err = queue.enqueueReadBuffer(x_buffer, CL_TRUE, 0, POINTS * sizeof(cl_float4), x);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

    err = queue.enqueueReadBuffer(a_buffer, CL_TRUE, 0, POINTS * sizeof(cl_float4), a);
    ASSERT(err == CL_SUCCESS, "err was %d\n", err);

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
