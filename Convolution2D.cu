/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 
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
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
  int imageW, int imageH, int filterR) {
  
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int y = index / imageW; // row
  int x = index % imageW; // col
  float sum = 0;

  for(int k = -filterR; k <= filterR; k++){
    int d = x + k;

    if(d >= 0 && d < imageW)
      sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
  }

  d_Dst[index] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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

__global__ void convolutionColGPU(float *d_Dst, float *d_Src, float *d_Filter, 
  int imageW, int imageH, int filterR) {
  
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int y = index / imageW; // row
  int x = index % imageW; // col
  float sum = 0;

  for(int k = -filterR; k <= filterR; k++){
    int d = y + k;

    if(d >= 0 && d < imageH)
      sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
  }

  d_Dst[index] = sum;
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
    unsigned int i;

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

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
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

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }

    printf("Allocating GPU memory...\n");

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    printf("GPU computation...\n");

    convolutionRowGPU<<<1, imageH*imageW>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    convolutionColGPU<<<1, imageH*imageW>>>(d_Output, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
    


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
    
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    printf("Reset Device");

    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess){
      printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(e), __FILE__, __LINE__);
    }
    else{
      printf("cudaGetLastError() == cudaSuccess!\n");
    }

    cudaDeviceReset();


    return 0;
}
