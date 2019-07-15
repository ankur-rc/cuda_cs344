/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "float.h"

__global__ void local_min(const float *const d_logLuminance,
                          float *const d_min,
                          const size_t numRows,
                          const size_t numCols)
{
   extern __shared__ float s_patch[];

   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;

   int global_pos = r * numCols + c;
   int patch_pos = threadIdx.y * blockDim.x + threadIdx.y;

   s_patch[patch_pos] = (global_pos < numRows * numCols ? d_logLuminance[global_pos]
                                                        : FLT_MAX);
   __syncthreads();

   int start = (blockDim.x * blockDim.y >> 1);
   for (int mid = start; mid > 0; mid = mid >> 1)
   {
      if (patch_pos < mid)
      {
         s_patch[patch_pos] = s_patch[patch_pos] < s_patch[mid + patch_pos] ? s_patch[patch_pos] : s_patch[mid + patch_pos];
      }
      __syncthreads();
   }

   if (threadIdx.x == 0 && threadIdx.y == 0)
      d_min[blockIdx.y * blockDim.x + blockIdx.x] = s_patch[0];
}

__global__ void local_max(const float *const d_logLuminance,
                          float *const d_max,
                          const size_t numRows,
                          const size_t numCols)
{
   extern __shared__ float s_patch[];

   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;

   int global_pos = r * numCols + c;
   int patch_pos = threadIdx.y * blockDim.x + threadIdx.y;

   s_patch[patch_pos] = (global_pos < numRows * numCols ? d_logLuminance[global_pos]
                                                        : -FLT_MAX);
   __syncthreads();

   // if (blockIdx.x == 0 && blockIdx.y == 0)
   // {
   //    printf("Global Pos:%d \t Max:%f\n", blockIdx.y * blockDim.x + blockIdx.x, s_patch[patch_pos]);
   // }

   int start = (blockDim.x * blockDim.y >> 1);
   for (int mid = start; mid > 0; mid = mid >> 1)
   {
      if (patch_pos < mid)
      {
         s_patch[patch_pos] = s_patch[patch_pos] > s_patch[mid + patch_pos] ? s_patch[patch_pos] : s_patch[mid + patch_pos];
      }
      __syncthreads();
   }

   if (threadIdx.x == 0 && threadIdx.y == 0)
   {
      d_max[blockIdx.y * blockDim.x + blockIdx.x] = s_patch[0];
      // printf("Global Pos:%d \t Max:%f\n", blockIdx.y * blockDim.x + blockIdx.x, s_patch[0]);
   }
}

void global_extrema(const float *const d_min, const float *const d_max, float &min, float &max, int len)
{
   using namespace std;
   {
      min = FLT_MAX;
      max = -FLT_MAX;

      cout << "Length: " << len << "\nOriginal: \n";
      cout << "Min: \n";
      for (int i = 0; i < len; ++i)
         cout << d_min[i] << " ";

      cout << "\nMax: \n";
      for (int i = 0; i < len; ++i)
         cout << d_max[i] << " ";

      for (int i = 0; i < len; ++i)
      {
         min = d_min[i] < min ? d_min[i] : min;
         max = d_max[i] > max ? d_max[i] : max;
      }
   }
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
   //TODO
   /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

   float *arr;

   checkCudaErrors(cudaMallocManaged((void **)&arr, sizeof(float) * numCols * numRows));

   for (int i = 0; i < numRows * numCols; ++i)
      arr[i] = 5.0f;

   arr[0] = 10.0f;
   arr[1] = 3.0f;

   dim3 blockSize(32, 32, 1);
   dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y, 1);

   float *d_min, *d_max;
   checkCudaErrors(cudaMallocManaged((void **)&d_min, sizeof(float) * gridSize.x * gridSize.y));
   checkCudaErrors(cudaMallocManaged((void **)&d_max, sizeof(float) * gridSize.x * gridSize.y));

   local_min<<<gridSize, blockSize, sizeof(float) * blockSize.x * blockSize.y>>>(arr, d_min, numRows, numCols);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());

   local_max<<<gridSize, blockSize, sizeof(float) * blockSize.x * blockSize.y>>>(arr, d_max, numRows, numCols);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());

   global_extrema(d_min, d_max, min_logLum, max_logLum, gridSize.x * gridSize.y);
   checkCudaErrors(cudaFree(d_min));
   checkCudaErrors(cudaFree(d_max));

   std::cout << "\nMin: " << min_logLum << "\n"
             << "Max: " << max_logLum << "\n";
}
