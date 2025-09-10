#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        void radixsort(int n, int* odata, const int* idata);
    }

    namespace EfficientSharedMem {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int* odata, const int* idata);

        int compact(int n, int* odata, const int* idata);

        void radixsort(int n, int* odata, const int* idata);
    }
}
