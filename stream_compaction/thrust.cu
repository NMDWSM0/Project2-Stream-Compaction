#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::host_vector<int> host_idata(idata, idata + n);

            thrust::device_vector<int> dev_idata(host_idata);
            thrust::device_vector<int> dev_odata(n);
            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // do exclusive scan
            thrust::exclusive_scan(dev_idata.begin(), dev_idata.end(), dev_odata.begin());
            timer().endGpuTimer();

            thrust::copy(dev_odata.begin() + 1, dev_odata.end(), odata);
            // convert to inclusive
            odata[n - 1] = odata[n - 2] + idata[n - 1];
        }
    }
}
