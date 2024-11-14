/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	5
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            goto cleanup; \
        } \
    } while (0)
 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }    
}

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
  int imageW, int imageH, int filterR) {
  
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int y = index / imageW; // row
  int x = index % imageW; // col
  double sum = 0;
  // printf("Blockidx.x: %d, BlockDim.x: %d, threadIdx.x: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
  if(x < imageW && y < imageH){
    for(int k = -filterR; k <= filterR; k++){
      int d = x + k;
      if(d >= 0 && d < imageW)
        sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
    }
    d_Dst[index] = sum;
  }
}

__global__ void convolutionColGPU(double *d_Dst, double *d_Src, double *d_Filter, 
  int imageW, int imageH, int filterR) {
  
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int y = index / imageW; // row
  int x = index % imageW; // col
  double sum = 0;
  if(x < imageW && y < imageH){
    for(int k = -filterR; k <= filterR; k++){
      int d = y + k;
      if(d >= 0 && d < imageH)
        sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
    }
    d_Dst[index] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}




////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    double
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;
    double *d_Filter = NULL;
    double*d_Input = NULL;
    double *d_Buffer = NULL;
    double *d_Output = NULL;

    int imageW;
    int imageH;
    int extra_block;  // used to indicate if an extra block is needed (if image size is not divisible by 1024)
    unsigned int i;

    // CUDA Error Checking
    cudaError_t e;

    // CPU time measuring variables
    clock_t start, end;

    // CUDA measuring events
    cudaEvent_t startCuda, stopCuda;
    float millisecondsTransfers = 0, millisecondsKernelsandTransferBack = 0;

	  printf("Enter filter radius : ");
	  // scanf("%d", &filter_radius);
    if(!scanf("%d", &filter_radius)){
      printf("ERROR: scanf: FILE: %s, LINE: %d\n", __FILE__, __LINE__);
    }

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    // scanf("%d", &imageW);
    if(!scanf("%d", &imageW)){
      printf("ERROR: scanf: FILE: %s, LINE: %d\n", __FILE__, __LINE__);
    }
    imageH = imageW;
    extra_block = ((imageH*imageW)%1024 != 0);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
    h_Input     = (double *)malloc(imageW * imageH * sizeof(double));
    h_Buffer    = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputCPU = (double *)malloc(imageW * imageH * sizeof(double));
    h_OutputGPU = (double *)malloc(imageW * imageH * sizeof(double));

    if(!(h_Filter && h_Input && h_Buffer && h_OutputCPU && h_OutputGPU)){
      printf("ERROR on Host Allocation!\n");
      goto cleanup;
    }
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (double)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
    }

    printf("Allocating GPU memory...\n");

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(double)));


    // Start Measuring memory transfer times
    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda, 0);

    CHECK_CUDA_ERROR(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(double), cudaMemcpyHostToDevice));

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    start = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
    end = clock();
    printf("CPU time: %f msec\n", (double)((end - start)/(CLOCKS_PER_SEC/1000000))/1000);
    printf("GPU computation...\n");

    // 1st Kernel launch
    // No need for sync barrier because cudaMemCpy(..., hostToDevice) works as barrier
    //printf("1st Kernel Launch: Row Conolution: ");
    cudaEventRecord(startCuda, 0 );
    convolutionRowGPU<<<(imageH*imageW)/1024+extra_block, 1024>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if(e!=cudaSuccess){
      printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(e), __FILE__, __LINE__);
      goto cleanup;
    }
    //else{
    //  printf("cudaGetLastError() == cudaSuccess!\n");
    //}

    // 2nd Kernel launch
    // Synchronize between 2 kernels launch because Column kernel needs the d_Buffer as input.
    // d_Buffer works as intermediate result so we ensure it is completely written.

    //printf("2nd Kernel Launch: Col Conolution: ");
    convolutionColGPU<<<(imageH*imageW)/1024+extra_block, 1024>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if(e!=cudaSuccess){
      printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(e), __FILE__, __LINE__);
      goto cleanup;
    }
    //else{
    //  printf("cudaGetLastError() == cudaSuccess!\n");
    //}

    CHECK_CUDA_ERROR(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsKernelsandTransferBack, startCuda, stopCuda);
    printf("GPU Time: %f msec\n", millisecondsTransfers+millisecondsKernelsandTransferBack);
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    for (i = 0; i < imageW * imageH; i++) {
        if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) > accuracy){
          printf("Difference bigger than accuray in index i = %d, h_OutputGPU[%d] = %lf, h_OutputCPU[%d] = %lf\n", i, i, h_OutputGPU[i], i, h_OutputCPU[i]);
          goto cleanup;
        }
    }

cleanup:
    // free all the allocated memory
    printf("Deallocating host memory\n");
    if (h_OutputCPU) free(h_OutputCPU);
    if (h_Buffer) free(h_Buffer);
    if (h_Input) free(h_Input);
    if (h_Filter) free(h_Filter);
    if (h_OutputGPU) free(h_OutputGPU);

    printf("Deallocating device memory\n");
    if (d_Output) cudaFree(d_Output);
    if (d_Buffer) cudaFree(d_Buffer);
    if (d_Input) cudaFree(d_Input);
    if (d_Filter) cudaFree(d_Filter);

    printf("Destroying CUDA events\n");
    CHECK_CUDA_ERROR(cudaEventDestroy(startCuda));
    CHECK_CUDA_ERROR(cudaEventDestroy(stopCuda));
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    printf("Reset Device: ");

    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if(e!=cudaSuccess){
      printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(e), __FILE__, __LINE__);
    }
    else{
      printf("cudaGetLastError() == cudaSuccess!\n");
    }

    cudaDeviceReset();


    return 0;
}
