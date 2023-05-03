#include <iostream>
#include <limits>
#include <CL/opencl.h>

constexpr size_t Count = 1024;

using namespace std;

const char *KernelSource =
  "__kernel void square(__global float* input,                             "
  "                     __global float* output,                            "
  "                     const unsigned long count) {                       "
  "   unsigned i = get_global_id(0);                                       "
  "   if(i < count)                                                        "
  "     output[i] = input[i] * input[i];                                   "
  "}                                                                       ";

int main() {
  int err;                            // error code returned from api calls

  float data[Count];                  // original data set given to device
  float results[Count];               // results returned from device

  size_t local;                       // local domain size for our calculation

  cl_device_id device_id;             // compute device id
  cl_context context;                 // compute context
  cl_command_queue commands;          // compute command queue
  cl_program program;                 // compute program
  cl_kernel kernel;                   // compute kernel

  cl_mem input;                       // device memory used for the input array
  cl_mem output;                      // device memory used for the output array

  // Fill our data set with random float values
  for(unsigned i = 0; i < Count; i++)
    data[i] = rand() / float(RAND_MAX);

  // Connect to a compute device
  //
  int gpu = 1;
  err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
  if (err != CL_SUCCESS) {
    cout << "Error: Failed to create a device group!" << endl;
    return EXIT_FAILURE;
  }

  // Create a compute context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context) {
    cout << "Error: Failed to create a compute context!" << endl;
    return EXIT_FAILURE;
  }

  // Create a command commands
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands) {
    cout <<"Error: Failed to create a command commands!\n" << endl;
    return EXIT_FAILURE;
  }

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  if (!program) {
    cout << "Error: Failed to create compute program!" << endl;
    return EXIT_FAILURE;
  }

  // Build the program executable
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    cout << "Error: Failed to build program executable!" << endl;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    cout << buffer << endl;
    return EXIT_FAILURE;
  }

  // Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "square", &err);
  if (!kernel || err != CL_SUCCESS) {
    cout << "Error: Failed to create compute kernel!" << endl;
    return EXIT_FAILURE;
  }

  // Create the input and output arrays in device memory for our calculation
  input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * Count, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * Count, NULL, NULL);
  if (!input || !output) {
    cout << "Error: Failed to allocate device memory!" << endl;
    return EXIT_FAILURE;
  }

  // Write our data set into the input array in device memory
  err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(data), data, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    cout << "Error: Failed to write to source array!" << endl;
    return EXIT_FAILURE;
  }

  // Set the arguments to our compute kernel
  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  err |= clSetKernelArg(kernel, 2, sizeof(Count), &Count);
  if (err != CL_SUCCESS) {
    cout << "Error: Failed to set kernel arguments! " << err << endl;
    return EXIT_FAILURE;
  }

  // Get the maximum work group size for executing the kernel on the device
  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
  if (err != CL_SUCCESS) {
    cout << "Error: Failed to retrieve kernel work group info! " << err << endl;
    return EXIT_FAILURE;
  }

  // Execute the kernel over the entire range of our 1d input data set using the maximum number of work group items for this device
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &Count, &local, 0, NULL, NULL);
  if (err) {
    cout << "Error: Failed to execute kernel!" << endl;
    return EXIT_FAILURE;
  }

  // Wait for the command commands to get serviced before reading back results
  clFinish(commands);

  // Read back the results from the device to verify the output
  err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * Count, results, 0, NULL, NULL );
  if (err != CL_SUCCESS) {
    cout << "Error: Failed to read output array! " << err << endl;
    return EXIT_FAILURE;
  }

  // Validate our results
  auto correct = 0;  // number of correct results returned
  for (auto i = 0; i < Count; i++) {
    if (results[i] == data[i] * data[i])
      correct++;
  }

  // Print a brief summary detailing the results
  cout << "Computed " << correct << "/" << Count << " correct values!" << endl;

  // Shutdown and cleanup
  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}
