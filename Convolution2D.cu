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
#define accuracy  	1000000000000
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
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = filterR; y < imageH+filterR; y++) {
    for (x = filterR; x < imageW+filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        sum += h_Src[y * (imageW+2*filterR) + d] * h_Filter[filterR - k];
        h_Dst[y * (imageW+2*filterR) + x] = sum;
      }
    }
  }    
}

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
  int imageW, int imageH, int filterR) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int y = index / imageW; // row
  int x = index % imageW; // col
  y+= filterR;
  x+= filterR;
  float sum = 0;
    for(int k = -filterR; k <= filterR; k++){
      int d = x + k;
        sum += d_Src[y * (imageW+2*filterR) + d] * d_Filter[filterR - k];
    }
    d_Dst[y * (imageW + 2 * filterR) + x] = sum; 
}

__global__ void convolutionColGPU(float *d_Dst, float *d_Src, float *d_Filter, 
  int imageW, int imageH, int filterR) {
  
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int y = index / imageW; // row
  int x = index % imageW; // col
  y += filterR;
  x += filterR;
  float sum = 0;
    for(int k = -filterR; k <= filterR; k++){
      int d = y + k;
        sum += d_Src[d * (imageW+2*filterR) + x] * d_Filter[filterR - k];
    }
      d_Dst[(y-filterR) * (imageW) + (x-filterR)] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = filterR; y < imageH+filterR; y++) {
    for (x = filterR; x < imageW+filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;
          sum += h_Src[d * (imageW+2*filterR) + x] * h_Filter[filterR - k];
        h_Dst[(y-filterR) * imageW + (x-filterR)] = sum;
      }
    }
  }
    
}




////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;
    float *d_Filter = NULL;
    float*d_Input = NULL;
    float *d_Buffer = NULL;
    float *d_Output = NULL;

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
    if(!scanf("%d", &filter_radius)){
      printf("ERROR: scanf: FILE: %s, LINE: %d\n", __FILE__, __LINE__);
    }

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    if(!scanf("%d", &imageW)){
      printf("ERROR: scanf: FILE: %s, LINE: %d\n", __FILE__, __LINE__);
    }
    imageH = imageW;
    extra_block = ((imageH*imageW)%1024 != 0);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)calloc((imageW+2*filter_radius) * (imageH+2*filter_radius), sizeof(float)); // use calloc to initialize padded elements to 0
    h_Buffer    = (float *)calloc((imageW+2*filter_radius) * (imageH+2*filter_radius), sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));

    if(!(h_Filter && h_Input && h_Buffer && h_OutputCPU && h_OutputGPU)){
      printf("ERROR on Host Allocation!\n");
      goto cleanup;
    }
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

 for (i = filter_radius; i < filter_radius + imageH; i++) {
        int row = i*(2*filter_radius + imageW);
        for(int j = filter_radius; j < imageW + filter_radius; j++){
          h_Input[row+j] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
        }
    }

    printf("Allocating GPU memory...\n");

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Input, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Buffer, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

    cudaMemset(d_Input, 0, (imageW+2*filter_radius) * (imageH+2*filter_radius));
    cudaMemset(d_Buffer, 0, (imageW+2*filter_radius) * (imageH+2*filter_radius));
    cudaMemset(d_Output, 0, imageW * imageH);

    // Start Measuring memory transfer times
    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda, 0);

    CHECK_CUDA_ERROR(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Input, h_Input, (imageW+2*filter_radius) * (imageH+2*filter_radius) * sizeof(float), cudaMemcpyHostToDevice));

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

    CHECK_CUDA_ERROR(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsKernelsandTransferBack, startCuda, stopCuda);
    printf("GPU Time: %f msec\n", millisecondsTransfers+millisecondsKernelsandTransferBack);

    printf("CPU OUTPUT: \n");
      for (i = 0; i < imageW * imageH; i++) {
        printf("%f  ", h_OutputCPU[i]);
      }
    printf("\nDONE\n");
    
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    for (i = 0; i < imageW * imageH; i++) {
        if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) > accuracy){
          printf("Difference bigger than accuray in index i = %d, h_OutputGPU[%d] = %f, h_OutputCPU[%d] = %f\n", i, i, h_OutputGPU[i], i, h_OutputCPU[i]);
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
