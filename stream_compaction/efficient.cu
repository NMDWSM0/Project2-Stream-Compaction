#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernExclusiveScanEachBlock(int N, int* odata, int* blocksumdata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            constexpr int maxdepth = ilog2ceil(blockSize);

            // upsweep
            for (int d = 0; d < maxdepth; ++d) {
                const int offset = 1 << d;
                if ((threadIdx.x + 1) % (2 * offset) == 0) {
                    odata[index] += odata[index - offset];
                }
                __syncthreads();
            }

            // save the last data(total sum of a block) 
            if (threadIdx.x == blockSize - 1) {
                blocksumdata[blockIdx.x] = odata[index];
                odata[index] = 0; // and set the last data to 0
            }
            __syncthreads();

            // downsweep
            for (int d = maxdepth - 1; d >= 0; --d) {
                const int offset = 1 << d;
                if ((threadIdx.x + 1) % (2 * offset) == 0) {
                    int temp = odata[index - offset];
                    odata[index - offset] = odata[index];
                    odata[index] += temp;
                }
                __syncthreads();
            }
        }

        __global__ void kernExclusiveScanOneBlock(int N, int* odata) {
            int index = threadIdx.x;
            int maxdepth = ilog2ceil(N);

            // upsweep
            for (int d = 0; d < maxdepth; ++d) {
                const int offset = 1 << d;
                if ((threadIdx.x + 1) % (2 * offset) == 0) {
                    odata[index] += odata[index - offset];
                }
                __syncthreads();
            }

            // save the last data(total sum of a block) 
            if (threadIdx.x == N - 1) {
                odata[index] = 0; // and set the last data to 0
            }
            __syncthreads();

            // downsweep
            for (int d = maxdepth - 1; d >= 0; --d) {
                const int offset = 1 << d;
                if ((threadIdx.x + 1) % (2 * offset) == 0) {
                    int temp = odata[index - offset];
                    odata[index - offset] = odata[index];
                    odata[index] += temp;
                }
                __syncthreads();
            }
        }

        __global__ void kernAddBlockSum(int N, int* odata, int* scanned_blocksumdata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            odata[index] += scanned_blocksumdata[blockIdx.x];
        }

        __global__ void kernValueToBool(int N, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N) {
                return;
            }
            odata[index] = (idata[index] > 0);
        }

        __global__ void kernDoCompaction(int N, int* odata, const int* idata, const int* indexdata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N) {
                return;
            }
            if (idata[index] > 0) {
                odata[indexdata[index]] = idata[index];
            }
        }

        void recursiveExclusiveScan(int N, const std::vector<int*>& odatas, int bufferlvl) {
            if (N <= blockSize) 
            {
                // I'm sure that N is power of 2
                kernExclusiveScanOneBlock<<<1, N>>>(N, odatas[bufferlvl]);
                return;
            }
            else
            {
                int fullBlocksPerGrid = ((N + blockSize - 1) / blockSize);
                kernExclusiveScanEachBlock<<<fullBlocksPerGrid, blockSize>>>(N, odatas[bufferlvl], odatas[bufferlvl + 1]);
                recursiveExclusiveScan(fullBlocksPerGrid, odatas, bufferlvl + 1);
                kernAddBlockSum<<<fullBlocksPerGrid, blockSize>>>(N, odatas[bufferlvl], odatas[bufferlvl + 1]);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // compute dims
            int depth = ilog2ceil(n);
            int N = 1 << depth;  // set N the smallest power of 2

            // memory allocation
            std::vector<int*> dev_datas;
            int bufferSize = N;
            do {
                int* dev_data;
                cudaMalloc(&dev_data, bufferSize * sizeof(int));
                cudaMemset(dev_data, 0, bufferSize * sizeof(int));
                dev_datas.push_back(dev_data);
                bufferSize /= blockSize;
            } while (bufferSize > 1);

            // copy data to device
            cudaMemcpy(dev_datas[0], idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // start kernel
            timer().startGpuTimer();
            // do exclusive scan
            recursiveExclusiveScan(N, dev_datas, 0);
            timer().endGpuTimer();

            // copy back data
            cudaMemcpy(odata, dev_datas[0] + 1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[n - 1] = odata[n - 2] + idata[n - 1];  // make inclusive

            // free memory
            for (auto dev_data : dev_datas) {
                cudaFree(dev_data);
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // compute dims
            int depth = ilog2ceil(n);
            int N = 1 << depth;  // set N the smallest power of 2
            int fullBlocksPerGrid = (N + blockSize - 1) / blockSize;

            // memory allocation
            std::vector<int*> dev_datas;
            int bufferSize = N;
            do {
                int* dev_data;
                cudaMalloc(&dev_data, bufferSize * sizeof(int));
                cudaMemset(dev_data, 0, bufferSize * sizeof(int));
                dev_datas.push_back(dev_data);
                bufferSize /= blockSize;
            } while (bufferSize > 1);
            int *dev_idata, *dev_odata;
            cudaMalloc(&dev_idata, N * sizeof(int));
            cudaMemset(dev_idata, 0, N * sizeof(int));
            cudaMalloc(&dev_odata, N * sizeof(int));
            cudaMemset(dev_odata, 0, N * sizeof(int));

            // copy data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // start kernel
            timer().startGpuTimer();
            // convert to bool
            kernValueToBool<<<fullBlocksPerGrid, blockSize>>>(N, dev_datas[0], dev_idata);
            // do exclusive scan
            recursiveExclusiveScan(N, dev_datas, 0);
            // write values
            kernDoCompaction<<<fullBlocksPerGrid, blockSize>>>(N, dev_odata, dev_idata, dev_datas[0]);
            timer().endGpuTimer();

            // copy back data
            // first copy back index data to find the count
            cudaMemcpy(odata, dev_datas[0], N * sizeof(int), cudaMemcpyDeviceToHost);
            int count = odata[N - 1];
            // then copy real values
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            for (auto dev_data : dev_datas) {
                cudaFree(dev_data);
            }
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            return count;
        }
    }
}
