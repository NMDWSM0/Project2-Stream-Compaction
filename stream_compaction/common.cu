#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            bools[index] = (idata[index] > 0);
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (bools[index]) {
                odata[indices[index]] = idata[index];
            }
        }

        __global__ void kernMapToInverseBooleanDigit(int n, int* bools, const int* idata, int digit) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            bools[index] = 1 - GET_BIT(idata[index], digit);
        }

        __global__ void kernComputeTData(int n, int* tdata, const int* fdata, int totalFalses) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            tdata[index] = index - fdata[index] + totalFalses;
        }

        __global__ void kernWriteRadixSort(int n, int* odata, const int* idata, const int* fdata, const int* tdata, int digit) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int in = idata[index];
            int sortedIndex = GET_BIT(in, digit) ? tdata[index] : fdata[index];
            odata[sortedIndex] = in;
        }
    }
}
