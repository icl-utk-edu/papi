//This is a matrix multiplication program in CUDA without any optimizations
//like tiling, using shared memory etc

#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<assert.h>


__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int width)
{

	//2D thread ID 
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tdx=threadIdx.x;
	int tdy=threadIdx.y;

	int tx=bx*blockDim.x+tdx;
	int ty=by*blockDim.y+tdy;
	
	//Pvalue stores the Pd element that is computed by the thread
	float Pvalue=0;
	for(int k=0;k<width;++k){
		float Mdelement=Md[ty*width+k];
		float Ndelement=Nd[k*width+tx];
		Pvalue += Mdelement*Ndelement;
	}
	//Write the matrix to device memory each thread writes one element
	Pd[ty*width+tx]=Pvalue;
}


int main(int argc, char** argv){

	int width;
	int BlockDim;
	int GridDim;

	if (argc == 3){
		width=atoi(argv[1]);
		BlockDim=atoi(argv[2]);
		GridDim=width/BlockDim;
		printf("Using matrix dimension %dx%d ,Block Dim %dx%d threads per block, Grid Dim %dx%d blocks per grid\n",width,width,BlockDim,BlockDim,GridDim,GridDim);
	}else{
		width=512;
		BlockDim=16;
		GridDim=width/BlockDim;
		printf("Using Default Parameters: matrix dimension %dx%d ,Block Dim %dx%d threads per block, Grid Dim %dx%d blocks per grid\n",width,width,BlockDim,BlockDim,GridDim,GridDim);		
	}
	dim3 dimBlock(BlockDim,BlockDim);
	dim3 dimGrid(GridDim,GridDim);
	cudaError_t error;
	cudaDeviceProp deviceProp;
	int devID=0;
	error=cudaGetDevice(&devID);
	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
	}

	error=cudaGetDeviceProperties(&deviceProp,devID);
	if (error != cudaSuccess){
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
	}else{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	int size=width*width*sizeof(float);
	float* M=(float*)malloc(size);
	float* N=(float*)malloc(size);
	float* P=(float*)malloc(size);

	float* Md,*Nd,*Pd;

	if(!(M&&N)){
		printf("Malloc failed\n");
		exit(-1);
	}

	 // initialization of host data
	for (int j = 0; j < width; j++) {
		for (int i = 0; i < width; i++) {
			M[j*width + i] = (float)(rand()%50);
			N[j*width + i] = (float)(rand()%50);
			P[j*width + i] = 0;
		}
	}

	error=cudaMalloc((void**)&Md,size);
	if(error!=cudaSuccess){
		printf("Device memory allocation for M failed \n");
		exit(-1);
	}
	error=cudaMalloc((void**)&Nd,size);
	if(error!=cudaSuccess){
		printf("Device memory allocation for N failed \n");
		exit(-1);
	}
	error=cudaMalloc((void**)&Pd,size);
	if(error!=cudaSuccess){
		printf("Device memory allocation for P failed \n");
		exit(-1);
	}

	error=cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
	if(error!=cudaSuccess){
		printf("Device memory copy for M failed \n");
		exit(-1);
	}
	
	error=cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);
	if(error!=cudaSuccess){
		printf("Device memory copy for N failed \n");
		exit(-1);
	}

	
	cudaEvent_t start;
	error=cudaEventCreate(&start);
	if(error!=cudaSuccess){
		printf("cuda event start failed \n");
		exit(-1);
	}

	cudaEvent_t stop;
	error=cudaEventCreate(&stop);
	if(error!=cudaSuccess){
		printf("cuda event stop failed \n");
		exit(-1);
	}

	error =cudaEventRecord(start,NULL);
	if(error!=cudaSuccess){
		printf("cuda event start record failed \n");
		exit(-1);
	}
	
	
	MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,width);

//	error=cudaDeviceSynchronize();
	error =cudaEventRecord(stop,NULL);
	if(error!=cudaSuccess){
		printf("cuda event stop record failed with error=%s\n",cudaGetErrorString(error));
		exit(-1);
	}

	error = cudaEventSynchronize(stop);
	if(error!=cudaSuccess){
		printf("cuda event sync failed :%s\n",cudaGetErrorString(error));
		exit(-1);
	}
	


	float msecTotal=0.0f;
	error = cudaEventElapsedTime(&msecTotal,start,stop);
	if(error!=cudaSuccess){
		printf("cuda elapsed time calculation failed \n");
		exit(-1);
	}

	float msecPerMatrixMul = msecTotal;
	double flopsPerMatrixMul = 2*width*width*width;
	double gigaFlops=(flopsPerMatrixMul*1.0e-9f)/(msecPerMatrixMul/1000.0f);
	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
		    gigaFlops,
			msecPerMatrixMul,
			flopsPerMatrixMul,
			width * width);



	error=cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);
	if(error!=cudaSuccess){
		printf("Device memoory copy back for Pd failed \n");
		exit(-1);
	}

	printf("Very slow Host Matrix Mult \n");
	float temp;
	// initialization of host data
	for (int i = 0; i < width; ++i) {
		for ( int j = 0; j < width; ++j) {
			temp=0;
			for(int k=0; k<width; ++k)
				temp+=M[i*width+k]*N[k*width+j];
			if(temp != P[i*width+j]){
				printf("Matrix Mult Screwed Up!! differ in values CPU:%f and GPU:%f \n",temp,P[i*width+j]);
				exit(-1);
			}
		}
		
	}
	
	
	free(M);
	free(N);
	free(P);
	cudaFree(Md);	
	cudaFree(Nd);	
	cudaFree(Pd);	
	cudaDeviceReset();
	return 1;

}
