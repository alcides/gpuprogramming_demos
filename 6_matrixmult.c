#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

void randomInit(float* data, int size)
{
   int i;
   for (i = 0; i < size; ++i)
   data[i] = rand() / (float)RAND_MAX;
}

int main(void) {

#define WA 1024
#define HA 1024
#define WB 1024
#define HB WA
#define WC WB
#define HC HA

unsigned int size_A = WA * HA;
unsigned int mem_size_A = sizeof(float) * size_A;
float* h_A = (float*) malloc(mem_size_A);

unsigned int size_B = WB * HB;
unsigned int mem_size_B = sizeof(float) * size_B;
float* h_B = (float*) malloc(mem_size_B);


unsigned int size_C = WC * HC;
unsigned int mem_size_C = sizeof(float) * size_C;
float* h_C = (float*) malloc(mem_size_C);




randomInit(h_A, size_A);
randomInit(h_B, size_B);

cl_context clGPUContext;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_kernel clKernel;

size_t dataBytes;
size_t kernelLength;
cl_int errcode;

// OpenCL device memory for matrices
cl_mem d_A;
cl_mem d_B;
cl_mem d_C;

clGPUContext = clCreateContextFromType(0, 
               CL_DEVICE_TYPE_GPU, 
               NULL, NULL, &errcode);

// get the list of GPU devices associated 
// with context
errcode = clGetContextInfo(clGPUContext, 
          CL_CONTEXT_DEVICES, 0, NULL, 
          &dataBytes);
cl_device_id *clDevices = (cl_device_id *)
          malloc(dataBytes);
errcode |= clGetContextInfo(clGPUContext, 
          CL_CONTEXT_DEVICES, dataBytes, 
          clDevices, NULL);

//Create a command-queue
clCommandQue = clCreateCommandQueue(clGPUContext, 
              clDevices[0], 0, &errcode);

// Setup device memory
d_C = clCreateBuffer(clGPUContext, 
      CL_MEM_READ_WRITE, 
      mem_size_A, NULL, &errcode);
d_A = clCreateBuffer(clGPUContext, 
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
      mem_size_A, h_A, &errcode);
d_B = clCreateBuffer(clGPUContext, 
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
      mem_size_B, h_B, &errcode);


FILE* fp = fopen("6_matrixmult.cl", "r");
fseek (fp , 0 , SEEK_END);
const size_t lSize = ftell(fp);
rewind(fp);
unsigned char* buffer;
buffer = (unsigned char*) malloc (lSize);
fread(buffer, 1, lSize, fp);
fclose(fp);

cl_int status;
clProgram = clCreateProgramWithBinary(clGPUContext, 
            1, (const cl_device_id *)clDevices, 
            &lSize, (const unsigned char**)&buffer, 
            &status, &errcode);
errcode = clBuildProgram(clProgram, 0, NULL, NULL, 
            NULL, NULL);

errcode = clBuildProgram(clProgram, 0, 
          NULL, NULL, NULL, NULL);

clKernel = clCreateKernel(clProgram, 
           "matrixMul", &errcode);


// 7. Launch OpenCL kernel
size_t localWorkSize[2], globalWorkSize[2];

int wA = WA;
int wC = WC;
errcode = clSetKernelArg(clKernel, 0, 
          sizeof(cl_mem), (void *)&d_C);
errcode |= clSetKernelArg(clKernel, 1, 
          sizeof(cl_mem), (void *)&d_A);
errcode |= clSetKernelArg(clKernel, 2, 
          sizeof(cl_mem), (void *)&d_B);
errcode |= clSetKernelArg(clKernel, 3, 
          sizeof(int), (void *)&wA);
errcode |= clSetKernelArg(clKernel, 4, 
          sizeof(int), (void *)&wC);

localWorkSize[0] = 16;
localWorkSize[1] = 16;
globalWorkSize[0] = 1024;
globalWorkSize[1] = 1024;

errcode = clEnqueueNDRangeKernel(clCommandQue, 
          clKernel, 2, NULL, globalWorkSize, 
          localWorkSize, 0, NULL, NULL);

errcode = clEnqueueReadBuffer(clCommandQue, 
          d_C, CL_TRUE, 0, mem_size_C, 
          h_C, 0, NULL, NULL);
		  
		  


		  int i;
		  printf("\n\nMatrix C (Results)\n");
		  for(i = 0; i < size_C; i++)
		  {
		    printf("%f ", h_C[i]);
		    if(((i + 1) % WC) == 0)
		    printf("\n");
		  }
		  printf("\n");

		  

free(h_A);
free(h_B);
free(h_C);

clReleaseMemObject(d_A);
clReleaseMemObject(d_C);
clReleaseMemObject(d_B);

free(clDevices);
clReleaseContext(clGPUContext);
clReleaseKernel(clKernel);
clReleaseProgram(clProgram);
clReleaseCommandQueue(clCommandQue);

}