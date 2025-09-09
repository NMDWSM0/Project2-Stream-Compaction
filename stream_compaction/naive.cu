#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int N, int* odata, const int* idata, int pow2d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N) {
                return;
            }
            if (index >= pow2d) {
                odata[index] = idata[index] + idata[index - pow2d];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // compute dims
            int depth = ilog2ceil(n);
            int N = 1 << depth;  // set N the smallest power of 2
            int fullBlocksPerGrid = (N + blockSize - 1) / blockSize;

            // memory allocation
            int *dev_dataA, * dev_dataB;
            cudaMalloc(&dev_dataA, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataA failed!", __LINE__);
            cudaMalloc(&dev_dataB, N * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataB failed!", __LINE__);

            // copy data to device
            cudaMemset(dev_dataA, 0, N * sizeof(int));
            cudaMemset(dev_dataB, 0, N * sizeof(int));
            cudaMemcpy(dev_dataA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // start kernel
            timer().startGpuTimer();
            int curBuffer = 0;
            int *inDataPtr, *outDataPtr;
            for (int i = 0; i < depth; ++i) {
                inDataPtr = (curBuffer == 0) ? dev_dataA : dev_dataB;
                outDataPtr = (curBuffer == 0) ? dev_dataB : dev_dataA;
                kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(N, outDataPtr, inDataPtr, (1 << i));
                curBuffer = (curBuffer + 1) % 2;
            }
            timer().endGpuTimer();

            // copy back data
            cudaMemcpy(odata, outDataPtr, n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            cudaFree(dev_dataA);
            cudaFree(dev_dataB);
        }
    }
}
