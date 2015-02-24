#include <stdio.h>

#define N 2048

__global__ void block_sum(const int *input,
                          int *per_block_results,
                          const size_t n)
{
  extern __shared__ int sdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // load input into __shared__ memory
  int x = 0;
  if(i < n)
  {
    x = input[i];
  }
  sdata[threadIdx.x] = x;
  __syncthreads();

  // contiguous range pattern
  for(int offset = blockDim.x / 2;
      offset > 0;
      offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      // add a partial sum upstream to our own
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }

    // wait until all threads in the block have
    // updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if(threadIdx.x == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
  }
}


int main( void ) {

	int host_a[N];
	
	for (int i=0; i<N; i++) {
		host_a[i] = 2;
	}
	
    const size_t block_size = 512;
    const size_t num_blocks = (N/block_size) + ((N%block_size) ? 1 : 0);
	

	int *dev_a;
	cudaMalloc(&dev_a, sizeof(int) * N);
	int *d_partial_sums_and_total = 0;
	cudaMalloc((void**)&d_partial_sums_and_total, sizeof(int) * (num_blocks + 1));
	
	cudaMemcpy(dev_a, host_a, sizeof(int) * N, cudaMemcpyHostToDevice);
		 
	block_sum<<<num_blocks,block_size,block_size * sizeof(int)>>>(dev_a, d_partial_sums_and_total, N);

	block_sum<<<1,num_blocks,num_blocks * sizeof(int)>>>(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);


	int device_result = 0;
	cudaMemcpy(&device_result, d_partial_sums_and_total + num_blocks, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", device_result);
	
}