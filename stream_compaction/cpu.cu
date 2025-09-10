#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = idata[0];
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int numRemain = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i]) {
                    odata[numRemain++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return numRemain;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* scanResult = new int[n];
            scanResult[0] = 0;
            for (int i = 1; i < n; ++i) {
                scanResult[i] = scanResult[i - 1] + (idata[i - 1] > 0);
            }
            for (int i = 0; i < n; ++i) {
                if (idata[i]) {
                    odata[scanResult[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            int number = scanResult[n - 1];
            delete[] scanResult;
            return number;
        }

        void sort(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            std::copy(idata, idata + n, odata);
            std::sort(odata, odata + n);
            timer().endCpuTimer();
        }
    }
}
