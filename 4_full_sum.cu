#include <stdio.h>

#define N 3000

__global__ void sum(int *a, int *b, int *c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < N) {
    c[i] = a[i] + b[i];
    i += gridDim.x * blockDim.x;
  }
}


int main( void ) {

	int host_a[N];
	int host_b[N];
	int host_c[N];
	
	for (int i=0; i<N; i++) {
		host_a[i] = i;
		host_b[i] = i;
	}

	int *dev_a, *dev_b, *dev_c;
	cudaMalloc(&dev_a, sizeof(int) * N);
	cudaMalloc(&dev_b, sizeof(int) * N);
	cudaMalloc(&dev_c, sizeof(int) * N);
	cudaMemcpy(dev_a, host_a, sizeof(int) * N,
	         cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, sizeof(int) * N,
	         cudaMemcpyHostToDevice);
		 
	sum<<<20, 30>>>(dev_a, dev_b, dev_c);

	cudaMemcpy(host_c, dev_c, sizeof(int) * N,
	            cudaMemcpyDeviceToHost);
	for (int i=0; i<N; i++) {
		printf("%d ", host_c[i]);
	}
	printf("\n");


}


