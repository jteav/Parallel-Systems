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
__global__ void histogram(int* r, int rSize, unsigned long long *histo, int num_bins){
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;
  uint nbits = log2f(num_bins);
  uint i, h;

  //Privatized bins
  extern __shared__ unsigned long long histo_s[];
  for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x){
		histo_s[binIdx] = 0u;
	}
  __syncthreads();
  
  //Histogram calculations
  for(i = index; i < rSize; i += stride){
    h = bfe(r[i], 0, nbits);
    atomicAdd(&(histo_s[h]), 1);
  }
  __syncthreads();

  //Commit to global memory
  for(int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x){
		atomicAdd(&(histo[binIdx]), histo_s[binIdx]);
	}
}

//define the prefix scan kernel here
__global__ void prefixScan(unsigned long long *histo, int num_bins, int *sum){
  for(int i = 0; i < num_bins; i++){
    if(i == 0)
      sum[i] = 0;
    else
      sum[i] = histo[i-1]+sum[i-1];
  }
}

//define the reorder kernel here
__global__ void Reorder(int* r, int rSize, int num_bins, int* prefixSum, int* output){
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

  //Declaring input array
  int* r_h;
  cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory in host so it copies to GPU faster
  dataGenerator(r_h, rSize, 0, 1);

  //Declaring histogram
  unsigned long long *h_histo, *d_histo;
  cudaMallocHost(&h_histo, sizeof(unsigned long long)*num_bins);
  cudaMalloc(&d_histo, num_bins*sizeof(unsigned long long));
  cudaMemcpy(d_histo, h_histo, num_bins*sizeof(unsigned long long), cudaMemcpyHostToDevice);

  //Declaring prefix sum
  int *h_prefix_sums, *d_prefix_sums;
  cudaMallocHost(&h_prefix_sums, sizeof(int)*num_bins);
  cudaMalloc(&d_prefix_sums, sizeof(int)*num_bins);
  //cudaMemset(d_prefix_sums, 0, prefixSize);

  //Declaring output array
  int *h_output, *d_output;
  cudaMallocHost(&h_output, sizeof(int)*rSize);
  cudaMalloc(&d_output, sizeof(int)*rSize);

  //Measuring GPU running time
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  //Calculating histogram
  histogram<<<gridSize, blockSize, num_bins*sizeof(unsigned long long)>>>(r_h, rSize, d_histo, num_bins);
  
  //Performing prefix scan
  prefixScan<<<gridSize, blockSize>>>(d_histo, num_bins, d_prefix_sums);

  //Performing reorder
  Reorder<<<gridSize, blockSize>>>(r_h, rSize, num_bins, d_prefix_sums, d_output);

  cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  /*cudaMemcpy(h_histo, d_histo, num_bins*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  int j;
  for(j = 0; j < num_bins; j++){
    printf("histo[%d] = %d\n", j, h_histo[j]);
  }
  cudaMemcpy(h_prefix_sums, d_prefix_sums, num_bins*sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < num_bins; i++){
    printf("prefix[%d] = %d\n", i, h_prefix_sums[i]);
  }*/
  cudaMemcpy(h_output, d_output, rSize*sizeof(int), cudaMemcpyDeviceToHost);
  for(int y = 0; y < rSize; y++){
    printf("output[%d] = %d\n", y, h_output[y]);
  }
  printf("******Total Running Time of Kernal = %0.5f ms******\n", elapsedTime);

  
  cudaFreeHost(r_h);
  cudaFreeHost(h_histo);
  cudaFreeHost(h_prefix_sums);
  cudaFreeHost(h_output);
  cudaFree(d_histo);
  cudaFree(d_prefix_sums);
  cudaFree(d_output);

  return 0;
}