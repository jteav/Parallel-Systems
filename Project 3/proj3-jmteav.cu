//Johnathan Teav
//Compile with nvcc proj3-jmteav.cu -arch=compute_52 -code=sm_52
#include <assert.h>
#include <stdio.h>

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
__global__ void histogram(int* r, int rSize, int *histo, int num_bins)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
  uint nbits = log2f(num_bins);
  uint i, h;
  for(i = index; i < rSize; i += stride){
    h = bfe(r[i], 0, nbits);
    atomicAdd(&(histo[h]), 1);
  }
}

//Prefix scan kernal borrowed from /apps/cuda/7.5/samples/6_Advanced/shfl_scans
__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL) {
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

//Uniform add borrowed from /apps/cuda/7.5/samples/6_Advanced/shfl_scans
__global__ void uniform_add(int *data, int *partial_sums, int len){
  __shared__ int buf;
  int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

  if(threadIdx.x == 0){
    buf = partial_sums[blockIdx.x];
  }

  __syncthreads();
  data[id] += buf;
}

//define the reorder kernel here
__global__ void Reorder(int* r, int rSize, int num_bins, int* prefixSum, int* output)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
  int i, h;
  uint nbits = log2f(num_bins);
  int offset;
  for(i = index; i < rSize; i += stride){
    h = bfe(r[i], 0, nbits);
    offset = atomicAdd(&prefixSum[h], 1);
    output[offset] = i;
  }
}

int main(int argc, char const *argv[])
{
  int rSize = atoi(argv[1]);
  int num_bins = atoi(argv[2]);
  int blockSize = 256;
  int gridSize = (rSize + blockSize -1) / blockSize;
  int nWarps = blockSize / 32;
  int shmem_sz = nWarps * sizeof(int);
  int n_prefixSums = rSize/blockSize;
  int prefixSize = n_prefixSums * sizeof(int);

  //Declaring input array
  int* r_h;
  int *r_d;
  cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory in host so it copies to GPU faster
  cudaMalloc((void**)&r_d, sizeof(int)*rSize);

  //dataGenerator(r_h, rSize, 0, 1);
  int k;
  for(k = 0; k < rSize; k++){
    r_h[k] = k;
  }
  cudaMemcpy(r_d, r_h, sizeof(int)*rSize, cudaMemcpyHostToDevice);

  //Declaring histogram
  int *h_histo, *d_histo;
  cudaMallocHost(&h_histo, sizeof(int)*num_bins);
  cudaMalloc(&d_histo, num_bins*sizeof(int));
  cudaMemcpy(d_histo, h_histo, num_bins*sizeof(int), cudaMemcpyHostToDevice);

  //Declaring prefix sum
  int *h_prefix_sums, *d_prefix_sums;
  cudaMallocHost(reinterpret_cast<void **>(&h_prefix_sums), prefixSize);
  cudaMalloc(reinterpret_cast<void **>(&d_prefix_sums), prefixSize);
  cudaMemset(d_prefix_sums, 0, prefixSize);

  //Declaring output array
  int *h_output, *d_output;
  cudaMallocHost((void**)&h_output, sizeof(int)*rSize);
  cudaMalloc((void**)&d_output, sizeof(int)*rSize);

  //Calculating histogram
  histogram<<<gridSize, blockSize>>>(r_d, rSize, d_histo, num_bins);
  cudaMemcpy(h_histo, d_histo, num_bins*sizeof(int), cudaMemcpyDeviceToHost);
  int j;
  for(j = 0; j < num_bins; j++){
    printf("histo[%d] = %d\n", j, h_histo[j]);
  }

  //Performing prefix scan
  shfl_scan_test<<<gridSize, blockSize, shmem_sz>>>(d_histo, 32, d_prefix_sums);
  shfl_scan_test<<<gridSize, blockSize, shmem_sz>>>(d_prefix_sums, 32);
  uniform_add<<<gridSize-1, blockSize>>>(d_histo+blockSize, d_prefix_sums, rSize);
  cudaMemcpy(h_prefix_sums, d_prefix_sums, prefixSize*sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < prefixSize; i++){
    printf("prefix: %d\n", h_prefix_sums[i]);
  }

  //Performing reorder
  Reorder<<<gridSize, blockSize>>>(r_d, rSize, num_bins, d_prefix_sums, d_output);
  cudaMemcpy(h_output, d_output, rSize*sizeof(int), cudaMemcpyDeviceToHost);
  int y;
  for(y = 0; y < rSize; y++){
    printf("%d\n", h_output[y]);
  }

  
  cudaFreeHost(r_h);
  cudaFreeHost(h_histo);
  cudaFreeHost(h_prefix_sums);
  cudaFreeHost(h_output);
  cudaFree(r_d);
  cudaFree(d_histo);
  cudaFree(d_prefix_sums);
  cudaFree(d_output);

  return 0;
}