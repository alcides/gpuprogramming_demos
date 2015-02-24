__kernel void matrixMul(__global float* C, 
          __global float* A, 
          __global float* B, 
          int wA, int wB)
{
  
   // Cuda Version
   //int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   //int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   float value = 0;
   for (int k = 0; k < wA; ++k)
   {
      float elementA = A[ty * wA + k];
      float elementB = B[k * wB + tx];
      value += elementA * elementB;
   }
   C[ty * wA + tx] = value;
}