/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/
//Modified by Johnathan Teav
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

bucket *histo;			//Struct to copy CUDA data into
atom *list;				//Struct to copy CUDA data into

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime, start, end;

/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket *h){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", h[i].d_cnt);
		total_cnt += h[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
	printf("\n");
}

//Display the differences
void output_differences(bucket *h1, bucket *h2){
	int i; 
	long long total_cnt = 0;
	printf("CPU bucket minus GPU bucket:");
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", h1[i].d_cnt - h2[i].d_cnt);
		total_cnt += h1[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

/*******************************
*************Kernals************
*******************************/
//Kernal function to calculate distance
__device__
double d_distance(atom *list, int ind1, int ind2) {
	
	double x1 = list[ind1].x_pos;
	double x2 = list[ind2].x_pos;
	double y1 = list[ind1].y_pos;
	double y2 = list[ind2].y_pos;
	double z1 = list[ind1].z_pos;
	double z2 = list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

//Kernal function to brute force SDH solution
__global__
void d_baseline(atom *list, bucket *histo, long long PDH_acnt, double PDH_res, int num_bins) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int i, j, h_pos;
	double dist;

	//Privatized bins
	extern __shared__ int histo_s[];
	for(unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x){
		histo_s[binIdx] = 0u;
	}
	__syncthreads();

	//Histogram
	for(i = index; i < PDH_acnt; i += stride) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = d_distance(list, i, j);
			h_pos = (int) (dist / PDH_res);
			atomicAdd(&(histo_s[h_pos]), 1);
		}
	}
	__syncthreads();

	//Commit to global memory
	for(int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x){
		atomicAdd(&(histo[binIdx].d_cnt), histo_s[binIdx]);
	}
}

int main(int argc, char **argv)
{
	if(argc != 4){
		printf("Error. 3 arguments required.\n");
		return 0;
	}
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	int blockSize = atoi(argv[3]);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	histo = (bucket *)malloc(sizeof(bucket)*num_buckets);		//bucket to copy GPU histogram into

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	atom *d_atom_list;		//GPU atom list
	cudaMalloc(&d_atom_list, PDH_acnt*sizeof(atom));
	cudaMemcpy(d_atom_list, atom_list, PDH_acnt*sizeof(atom), cudaMemcpyHostToDevice);

	bucket *d_histogram;	//GPU histogram
	cudaMalloc(&d_histogram, num_buckets*sizeof(bucket));
	cudaMemcpy(d_histogram, histogram, num_buckets*sizeof(bucket), cudaMemcpyHostToDevice);
	//start counting time
	gettimeofday(&startTime, &Idunno);
	
	//call CPU single thread version to compute the histogram
	PDH_baseline();

	//check the total running time
	report_running_time();
	
	//print out the histogram
	output_histogram(histogram);

	//Calculating blocks
	int numBlocks = (PDH_acnt + blockSize -1) / blockSize;

	//Measuring GPU running time
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Calculating the histogram
	d_baseline<<<numBlocks, blockSize, num_buckets*sizeof(int)>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Copying the CUDA data
	cudaMemcpy(histo, d_histogram, num_buckets*sizeof(bucket), cudaMemcpyDeviceToHost);

	//Output GPU histogram
	printf("******Total Running Time of Kernal = %0.5f ms******\n", elapsedTime);
	output_histogram(histo);

	//Display the differences
	output_differences(histogram, histo);

	cudaFree(d_atom_list);
	cudaFree(d_histogram);
	
	return 0;
}