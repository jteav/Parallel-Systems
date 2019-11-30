//Johnathan Teav
//Compile with nvcc proj3-jmteav.cu -arch=compute_52 -code=sm_52
#include <assert.h>
#include <stdio.h>
#include <math.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
void dataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

//define the histogram kernel here
__global__ void histogram(int* r_h, int rSize, unsigned long long *histo, int num_bins)
{
    unsigned int i, h;
    for(i = 0; i < rSize; i++){
            h = bfe(r_h[i], 0, log2f(num_bins));
            atomicAdd(&(histo[h]), 1);
    }
}

//prefix_scan kernal borrowed from CUDA sample
__global__ void shfl_scan_test(unsigned long long *data, int width, int *partial_sums = NULL) {
    extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;
  
    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = data[id];
  
    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.
  #pragma unroll
    for (int i = 1; i <= width; i *= 2) {
      int n = __shfl_up(value, i, width);
  
      if (lane_id >= i) value += n;
    }
  
    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp
  
    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize - 1) {
      sums[warp_id] = value;
    }
  
    __syncthreads();
  
    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
  
      int warp_sum = sums[lane_id];
  
      for (int i = 1; i <= width; i *= 2) {
        int n = __shfl_up(warp_sum, i, width);
  
        if (lane_id >= i) warp_sum += n;
      }
  
      sums[lane_id] = warp_sum;
    }
  
    __syncthreads();
  
    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;
  
    if (warp_id > 0) {
      blockSum = sums[warp_id - 1];
    }
  
    value += blockSum;
  
    // Now write out our result
    data[id] = value;
  
    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
      partial_sums[blockIdx.x] = value;
    }  
}

//define the reorder kernel here
__global__ void Reorder()
{

}

int main(int argc, char const *argv[])
{
    int rSize = atoi(argv[1]);
    int num_bins = atoi(argv[2]);
    int blockSize = 256;
    int gridSize = rSize / blockSize;
    int nWarps = blockSize / 32;
    int shmem_sz = nWarps * sizeof(int);

    int* r_h; //input array

    cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory in host so it copies to GPU faster
    
    dataGenerator(r_h, rSize, 0, 1);

    //Declaring histogram
    unsigned long long histo[num_bins];
    unsigned long long *d_histo;
    cudaMalloc(&d_histo, num_bins);

    //Calculating histogram
    histogram<<<1, 1>>>(r_h, rSize, d_histo, num_bins);

    //Performing prefix scan
    shfl_scan_test<<<gridSize, blockSize, shmem_sz>>>(d_histo, 32);

    cudaMemcpy(histo, d_histo, num_bins, cudaMemcpyDeviceToHost);
    int i;
    for(i = 0; i < num_bins; i++){
        printf("%d\n", histo[i]);
    }

    cudaFree(d_histo);
    cudaFreeHost(r_h);

    return 0;
}