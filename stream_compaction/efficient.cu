#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "efficient.h"

#define blockSize 512

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernExclusiveScanEachBlock(int N, int* odata, int* blocksumdata) {
            const int globalbase = (blockIdx.x * blockDim.x);
            const int tid = threadIdx.x;
            const int gid = tid + globalbase;
            constexpr int maxdepth = ilog2ceil(blockSize);

            // upsweep
            for (int d = 0; d < maxdepth; ++d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < blockSize / twooffset) {
                    const int gid_t = globalbase + (tid + 1) * twooffset - 1;
                    odata[gid_t] += odata[gid_t - offset];
                }
                __syncthreads();
            }

            // save the last data(total sum of a block) 
            if (threadIdx.x == blockSize - 1) {
                blocksumdata[blockIdx.x] = odata[gid];
                odata[gid] = 0; // and set the last data to 0
            }
            __syncthreads();

            // downsweep
            for (int d = maxdepth - 1; d >= 0; --d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < blockSize / twooffset) {
                    const int gid_t = globalbase + (tid + 1) * twooffset - 1;
                    int temp = odata[gid_t - offset];
                    odata[gid_t - offset] = odata[gid_t];
                    odata[gid_t] += temp;
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
                const int twooffset = offset << 1;
                if (index < blockSize / twooffset) {
                    const int gid_t = (index + 1) * twooffset - 1;
                    odata[gid_t] += odata[gid_t - offset];
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
                const int twooffset = offset << 1;
                if (index < blockSize / twooffset) {
                    const int gid_t = (index + 1) * twooffset - 1;
                    int temp = odata[gid_t - offset];
                    odata[gid_t - offset] = odata[gid_t];
                    odata[gid_t] += temp;
                }
                __syncthreads();
            }
        }

        __global__ void kernAddBlockSum(int N, int* odata, int* scanned_blocksumdata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            odata[index] += scanned_blocksumdata[blockIdx.x];
        }

        void recursiveExclusiveScan(int N, const std::vector<int*>& odatas, int bufferlvl) {
            if (N <= blockSize)
            {
                // I'm sure that N is power of 2
                kernExclusiveScanOneBlock<<<1, N >>>(N, odatas[bufferlvl]);
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
            cudaDeviceSynchronize();

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
            int *dev_idata, *dev_odata, *dev_bools;
            cudaMalloc(&dev_idata, N * sizeof(int));
            cudaMemset(dev_idata, 0, N * sizeof(int));
            cudaMalloc(&dev_odata, N * sizeof(int));
            cudaMemset(dev_odata, 0, N * sizeof(int));
            cudaMalloc(&dev_bools, N * sizeof(int));
            cudaMemset(dev_bools, 0, N * sizeof(int));

            // copy data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            // start kernel
            timer().startGpuTimer();
            // convert to bool
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(N, dev_bools, dev_idata);
            // copy memory to do scan
            cudaMemcpy(dev_datas[0], dev_bools, N * sizeof(int), cudaMemcpyDeviceToDevice);
            // do exclusive scan
            recursiveExclusiveScan(N, dev_datas, 0);
            // write values
            Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(N, dev_odata, dev_idata, dev_bools, dev_datas[0]);
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
            cudaFree(dev_bools);
            return count;
        }

        /**
         * Performs radix sort on idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to sort.
         */
        void radixsort(int n, int* odata, const int* idata) {
            // compute dims
            int depth = ilog2ceil(n);
            int N = 1 << depth;  // set N the smallest power of 2
            int fullBlocksPerGrid = (N + blockSize - 1) / blockSize;

            // memory allocation
            std::vector<int*> dev_fdatas;
            int bufferSize = N;
            do {
                int* dev_data;
                cudaMalloc(&dev_data, bufferSize * sizeof(int));
                cudaMemset(dev_data, 0, bufferSize * sizeof(int));
                dev_fdatas.push_back(dev_data);
                bufferSize /= blockSize;
            } while (bufferSize > 1);
            int* dev_tdata, *dev_ndataA, *dev_ndataB;
            cudaMalloc(&dev_tdata, N * sizeof(int));
            cudaMemset(dev_tdata, 0, N * sizeof(int));
            cudaMalloc(&dev_ndataA, N * sizeof(int));  
            cudaMemset(dev_ndataA, 0, N * sizeof(int)); // put some additional 0 is OK
            cudaMalloc(&dev_ndataB, N * sizeof(int));
            cudaMemset(dev_ndataB, 0, N * sizeof(int));

            // copy data to device
            cudaMemcpy(dev_ndataA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            // start kernel
            timer().startGpuTimer();
            int *in_dev_ndata, *out_dev_ndata;
            for (int digit = 0; digit < 31; ++digit) {
                // radix sort for one digit
                in_dev_ndata = (digit % 2) ? dev_ndataB : dev_ndataA;
                out_dev_ndata = (digit % 2) ? dev_ndataA : dev_ndataB;
                // compute e into fdatas[0]
                Common::kernMapToInverseBooleanDigit<<<fullBlocksPerGrid, blockSize>>>(N, dev_fdatas[0], in_dev_ndata, digit);
                int fLast, eLast;
                cudaMemcpy(&eLast, dev_fdatas[0] + N - 1, sizeof(int), cudaMemcpyDeviceToHost); // this is e[n 每 1]
                // do exclusive scan
                recursiveExclusiveScan(N, dev_fdatas, 0);
                // compute totalFalse
                cudaMemcpy(&fLast, dev_fdatas[0] + N - 1, sizeof(int), cudaMemcpyDeviceToHost); // this is f[n 每 1]
                int totalFalse = fLast + eLast;
                // compute tdata
                Common::kernComputeTData<<<fullBlocksPerGrid, blockSize>>>(N, dev_tdata, dev_fdatas[0], totalFalse);
                // write this sort
                Common::kernWriteRadixSort<<<fullBlocksPerGrid, blockSize>>>(N, out_dev_ndata, in_dev_ndata, dev_fdatas[0], dev_tdata, digit);
            }
            timer().endGpuTimer();

            // copy back data
            cudaMemcpy(odata, out_dev_ndata + (N - n), n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            for (auto dev_data : dev_fdatas) {
                cudaFree(dev_data);
            }
            cudaFree(dev_tdata);
            cudaFree(dev_ndataA);
            cudaFree(dev_ndataB);
        }
    }

    namespace EfficientSharedMem {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernExclusiveScanEachBlock_Eff(int N, int* odata, int* blocksumdata) {
            extern __shared__ int shared_odata[];

            const int tid = threadIdx.x;
            const int index = threadIdx.x + (blockIdx.x * blockDim.x);
            constexpr int maxdepth = ilog2ceil(blockSize);

            // load into sharedmemory
            shared_odata[tid] = odata[index];
            __syncthreads();

            // upsweep
            for (int d = 0; d < maxdepth; ++d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < blockSize / twooffset) {
                    const int tid_t = (tid + 1) * twooffset - 1;
                    shared_odata[tid_t] += shared_odata[tid_t - offset];
                }
                __syncthreads();
            }

            // save the last data(total sum of a block) 
            if (tid == blockSize - 1) {
                blocksumdata[blockIdx.x] = shared_odata[tid];
                shared_odata[tid] = 0; // and set the last data to 0
            }
            __syncthreads();

            // downsweep
            for (int d = maxdepth - 1; d >= 0; --d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < blockSize / twooffset) {
                    const int tid_t = (tid + 1) * twooffset - 1;
                    int temp = shared_odata[tid_t - offset];
                    shared_odata[tid_t - offset] = shared_odata[tid_t];
                    shared_odata[tid_t] += temp;
                }
                __syncthreads();
            }

            // write back to global memory
            odata[index] = shared_odata[tid];
        }

        __global__ void kernExclusiveScanEachBlock_Naive(int N, int* odata, int* blocksumdata) {
            extern __shared__ int shared_odata[];

            const int tid = threadIdx.x;
            const int index = threadIdx.x + (blockIdx.x * blockDim.x);
            constexpr int maxdepth = ilog2ceil(blockSize);

            // load into sharedmemory
            const int x = odata[index];
            shared_odata[tid] = x;
            __syncthreads();

            // upsweep
            for (int d = 0; d < maxdepth; ++d) {
                const int offset = 1 << d;
                int prev = shared_odata[max(tid - offset, 0)];
                __syncthreads();
                if (tid >= offset) {
                    shared_odata[tid] += prev;
                }
                __syncthreads();
            }
            shared_odata[tid] -= x;

            // save the last data(total sum of a block) 
            if (tid == blockSize - 1) {
                blocksumdata[blockIdx.x] = shared_odata[tid] + x;
            }

            // write back to global memory
            odata[index] = shared_odata[tid];
        }

        __global__ void kernExclusiveScanEachBlock(int N, int* odata, int* blocksum) {
            static_assert(blockSize % 32 == 0, "blockSize must be multiple of warpSize");
            constexpr int WARPS = blockSize / 32;

            __shared__ int warpSums[WARPS];
            const int tid = threadIdx.x;
            const int gid = threadIdx.x + (blockIdx.x * blockDim.x);
            const int lid = tid & 31;
            const int wid = tid >> 5;

            int x = odata[gid];

            int inclusiveInWarp = x;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int inclusiveInWarp_by_offset = __shfl_up_sync(0xFFFFFFFF, inclusiveInWarp, offset);
                if (lid >= offset) {
                    inclusiveInWarp += inclusiveInWarp_by_offset;
                }
            }
            int exclusiveInWarp = inclusiveInWarp - x;

            if (lid == 31) {
                warpSums[wid] = inclusiveInWarp;
            }
            __syncthreads();

            if (wid == 0) {
                int inclusiveWarpSum = (lid < WARPS) ? warpSums[lid] : 0;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    int inclusiveWarpSum_by_offset = __shfl_up_sync(0xFFFFFFFF, inclusiveWarpSum, offset);
                    if (lid >= offset) {
                        inclusiveWarpSum += inclusiveWarpSum_by_offset;
                    }
                }
                if (lid < WARPS) {
                    // warpSums should be exlusive warpsum scan
                    warpSums[lid] = inclusiveWarpSum - warpSums[lid];
                }
            }
            __syncthreads();
            const int warpPrefix = warpSums[wid];

            if (gid < N) {
                odata[gid] = warpPrefix + exclusiveInWarp;
            }

            if (tid == blockSize - 1) {
                blocksum[blockIdx.x] = warpPrefix + inclusiveInWarp;
            }
        }

        __global__ void kernExclusiveScanOneBlock(int N, int* odata) {
            extern __shared__ int shared_odata[];

            const int tid = threadIdx.x;
            int maxdepth = ilog2ceil(N);

            // load into sharedmemory
            shared_odata[tid] = odata[tid];
            __syncthreads();

            // upsweep
            for (int d = 0; d < maxdepth; ++d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < blockSize / twooffset) {
                    const int tid_t = (tid + 1) * twooffset - 1;
                    shared_odata[tid_t] += shared_odata[tid_t - offset];
                }
                __syncthreads();
            }

            // save the last data(total sum of a block) 
            if (tid == N - 1) {
                shared_odata[tid] = 0; // and set the last data to 0
            }
            __syncthreads();

            // downsweep
            for (int d = maxdepth - 1; d >= 0; --d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < blockSize / twooffset) {
                    const int tid_t = (tid + 1) * twooffset - 1;
                    int temp = shared_odata[tid_t - offset];
                    shared_odata[tid_t - offset] = shared_odata[tid_t];
                    shared_odata[tid_t] += temp;
                }
                __syncthreads();
            }

            // write back to global memory
            odata[tid] = shared_odata[tid];
        }

        __global__ void kernAddBlockSum(int N, int* odata, int* scanned_blocksumdata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            odata[index] += scanned_blocksumdata[blockIdx.x];
        }

        void recursiveExclusiveScan(int N, const std::vector<int*>& odatas, int bufferlvl) {
            if (N <= blockSize)
            {
                // I'm sure that N is power of 2
                kernExclusiveScanOneBlock <<<1, N, N * sizeof(int) >> > (N, odatas[bufferlvl]);
                return;
            }
            else
            {
                int fullBlocksPerGrid = ((N + blockSize - 1) / blockSize);
                //kernExclusiveScanEachBlock_Eff << <fullBlocksPerGrid, blockSize, blockSize * sizeof(int) >> > (N, odatas[bufferlvl], odatas[bufferlvl + 1]);
                //kernExclusiveScanEachBlock_Naive << <fullBlocksPerGrid, blockSize, blockSize * sizeof(int) >> > (N, odatas[bufferlvl], odatas[bufferlvl + 1]);
                kernExclusiveScanEachBlock << < fullBlocksPerGrid, blockSize >> > (N, odatas[bufferlvl], odatas[bufferlvl + 1]);
                recursiveExclusiveScan(fullBlocksPerGrid, odatas, bufferlvl + 1);
                kernAddBlockSum << <fullBlocksPerGrid, blockSize >> > (N, odatas[bufferlvl], odatas[bufferlvl + 1]);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
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
            cudaDeviceSynchronize();

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
        int compact(int n, int* odata, const int* idata) {
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
            int* dev_idata, * dev_odata, * dev_bools;
            cudaMalloc(&dev_idata, N * sizeof(int));
            cudaMemset(dev_idata, 0, N * sizeof(int));
            cudaMalloc(&dev_odata, N * sizeof(int));
            cudaMemset(dev_odata, 0, N * sizeof(int));
            cudaMalloc(&dev_bools, N * sizeof(int));
            cudaMemset(dev_bools, 0, N * sizeof(int));

            // copy data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            // start kernel
            timer().startGpuTimer();
            // convert to bool
            Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (N, dev_bools, dev_idata);
            // copy memory to do scan
            cudaMemcpy(dev_datas[0], dev_bools, N * sizeof(int), cudaMemcpyDeviceToDevice);
            // do exclusive scan
            recursiveExclusiveScan(N, dev_datas, 0);
            // write values
            Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (N, dev_odata, dev_idata, dev_bools, dev_datas[0]);
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
            cudaFree(dev_bools);
            return count;
        }

        /**
         * Performs radix sort on idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to sort.
         */
        void radixsort(int n, int* odata, const int* idata) {
            // compute dims
            int depth = ilog2ceil(n);
            int N = 1 << depth;  // set N the smallest power of 2
            int fullBlocksPerGrid = (N + blockSize - 1) / blockSize;

            // memory allocation
            std::vector<int*> dev_fdatas;
            int bufferSize = N;
            do {
                int* dev_data;
                cudaMalloc(&dev_data, bufferSize * sizeof(int));
                cudaMemset(dev_data, 0, bufferSize * sizeof(int));
                dev_fdatas.push_back(dev_data);
                bufferSize /= blockSize;
            } while (bufferSize > 1);
            int* dev_tdata, * dev_ndataA, * dev_ndataB;
            cudaMalloc(&dev_tdata, N * sizeof(int));
            cudaMemset(dev_tdata, 0, N * sizeof(int));
            cudaMalloc(&dev_ndataA, N * sizeof(int));
            cudaMemset(dev_ndataA, 0, N * sizeof(int)); // put some additional 0 is OK
            cudaMalloc(&dev_ndataB, N * sizeof(int));
            cudaMemset(dev_ndataB, 0, N * sizeof(int));

            // copy data to device
            cudaMemcpy(dev_ndataA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            // start kernel
            timer().startGpuTimer();
            int* in_dev_ndata, * out_dev_ndata;
            for (int digit = 0; digit < 31; ++digit) {
                // radix sort for one digit
                in_dev_ndata = (digit % 2) ? dev_ndataB : dev_ndataA;
                out_dev_ndata = (digit % 2) ? dev_ndataA : dev_ndataB;
                // compute e into fdatas[0]
                Common::kernMapToInverseBooleanDigit << <fullBlocksPerGrid, blockSize >> > (N, dev_fdatas[0], in_dev_ndata, digit);
                int fLast, eLast;
                cudaMemcpy(&eLast, dev_fdatas[0] + N - 1, sizeof(int), cudaMemcpyDeviceToHost); // this is e[n 每 1]
                // do exclusive scan
                recursiveExclusiveScan(N, dev_fdatas, 0);
                // compute totalFalse
                cudaMemcpy(&fLast, dev_fdatas[0] + N - 1, sizeof(int), cudaMemcpyDeviceToHost); // this is f[n 每 1]
                int totalFalse = fLast + eLast;
                // compute tdata
                Common::kernComputeTData << <fullBlocksPerGrid, blockSize >> > (N, dev_tdata, dev_fdatas[0], totalFalse);
                // write this sort
                Common::kernWriteRadixSort << <fullBlocksPerGrid, blockSize >> > (N, out_dev_ndata, in_dev_ndata, dev_fdatas[0], dev_tdata, digit);
            }
            timer().endGpuTimer();

            // copy back data
            cudaMemcpy(odata, out_dev_ndata + (N - n), n * sizeof(int), cudaMemcpyDeviceToHost);

            // free memory
            for (auto dev_data : dev_fdatas) {
                cudaFree(dev_data);
            }
            cudaFree(dev_tdata);
            cudaFree(dev_ndataA);
            cudaFree(dev_ndataB);
        }
    }
}
