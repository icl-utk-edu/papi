/*
 * Author: Sangamesh Ragate
 * Date : 18th Nov 2015
 * ICL-UTK
 * Description : This is the parent process that Preloads the libactivity.so
 * and launches the cuda application for performing PC-Sampling
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <sys/wait.h>

#include "path.h"




int main(int argc,char** argv){


	int pid;
	if(argc < 2){
		printf("please supply Cuda app to be profiled \n");
		exit(-1);
	}


	//get CUDA device ID and sampling rate from args
	char *samp, *device;
	extern char* optarg;
	extern int optind;
	int c,tmp,index;
	
	//set default samp and device
	device="0";
	samp="5";
	index=1;
	while((c=getopt(argc,argv,"d:s:"))!=-1){
		switch(c){
			case 'd':
				device = optarg;
				tmp=atoi(device);
				index = optind;
				if(tmp < 0){
					printf("GPU device ID not valid \n");
					exit(-1);
				}
				break;
			case 's':
				samp = optarg;
				tmp=atoi(samp);
				//record index for argument forming for cuda app
				index = optind;
				if(tmp < 0 || tmp > 5){
					printf("PC sampling rate not valid \n");
					exit(-1);
				}
				break;
			case '?':
				printf("Switch not recognized by papi_sampling_cuda utility \n");
				break;
		}
	}

	
	//form the arg list for the cuda app
	char** var;
	var=&argv[index];
	

	char* ld_lib;
	char env1[1024];
	char env2[256];
	char env3[256];
	int status;

	//get the shared library load path
	strcpy(env1,"LD_LIBRARY_PATH=");
	ld_lib=getenv("LD_LIBRARY_PATH");
	if(ld_lib == NULL){
		printf("Error loading CUDA shared libraries: LD_LIBRARY_PATH=NULL \n");
		exit(-1);
	}
	strcat(env1,ld_lib);

	strcpy(env2,"GPU_DEVICE_ID=");
	strcat(env2,device);
	
	strcpy(env3,"PC_SAMPLING_RATE=");
	strcat(env3,samp);

	//form the env variable
	char* env[]={env1,env2,env3,ld_prld, NULL};

	printf("\n\n\n\n");
	printf("***************** PAPI_SAMPLING_CUDA utility **********************\n");
	pid=fork();
	if(pid==0){
		execve(var[0],var,env);
	}else if(pid==-1){
		printf("Profile fork failed \n");
		exit(-1);
	}else{
		wait(&status);
	}
	return 0;
}
