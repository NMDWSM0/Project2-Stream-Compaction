# CUDA Stream Compaction

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

-   Yu Jiang
    -   [LinkedIn](https://www.linkedin.com/in/yu-jiang-450815328/), [twitter](https://x.com/lvtha0711)
-   Tested on: Windows 11, Ultra 7 155H @ 3.80 GHz, 32GB RAM, RTX 4060 8192MB (Personal Laptop)

## Features

### 1. CPU Scan & Stream Compaction

Just use for-loop to do all algorithm, and make sure this is correct because we have to use this one as correct result to compare other GPU methods' correctness.

### 2. Naive GPU Scan Algorithm

Dispatch the kernel each pass for one stride, and use ping-pong buffers to save the intermediate results of each pass. A for-loop for stride is used on the host.

### 3. Work-Efficient GPU Scan & Stream Compaction

#### 3.1 Scan

Firstly I implement a kernel using upsweep and downsweep to get the exclusive scan result inside each block, and save the total sum of each block into another buffer. In the kernel, do upsweep in for-loop for the offset length first, in each loop end add a `__syncthreads()` to make sure results are written correctly; similar for downsweep.  
Then I implement a host function which can recursively devide the whole array by blocksize, since we have to do another exclusive scan for sums of each block. When the arraysize is <= than blocksize, that's the terminal condition so we can return to last level.  
For the outermost scan function, I will allocate a series of device memory (the size is decreased by `blockSize`) and save them in a `std::vector`, so I don't need to allocate memory during the recursive function.

#### 3.2 Stream Compaction

By implementing `StreamCompaction::Common::kernMapToBoolean` and `StreamCompaction::Common::kernScatter`, I compute the boolean result and copy it to the first level of device memory series, then perform scan to compute the index, and scatter the array at last.

### 4. Using Thrust's Implementation

Just using thrust::exclusive_scan API is OK. (note that thrust is very slow in Debug Mode)

### 5. Why is My GPU Approach So Slow? (Extra Credit)

In my implementation, I use a complete kernel which can compute the whole process of scan within each block, so there's no such problems about thread numbers and number of blocks, the performance of Efficient method is faster than naive approach and CPU if there's enough data (about 65k, below that all GPU approach even thrust is slower than CPU).

### 6. Extra Credit

#### 6.1 Radix Sort

In my implementation, I use a outer for-loop for digits. In each digit's loop, I first use a kernel to compute the e(or b) value, and save that directly to the first level of device memory series, then perform scan to compute the f index, and use another kernel to compute t index for each thread, at last a scatter kernel is used to compute the real index and put the values into its right place.  
There's also a ping-pong buffer used to save sorted results after each digit's loop.

#### 6.2 Shared Memory GPU Scan

In my implementation, I tried three different types of algorithm about shared memory:

-   The first one is just copying Efficient method, and replacing all global read/write with shared read/write, since this algorithm is only scaning within each block so we can just load all values into the shared memory(one per thread) and then use them. (But this one's performance is not good on my GPU, **See performance analysis later**)
-   The second one is after I analyze the performance of the first one, in this method I'm not using the upsweep-downsweep workflow, instead I just implement a naive method for each block (and use shared memory for sure) and use `__syncthreads()` to sync within blocks rather than one kernel call per pass. (This one's performance is even better than the first one)
-   The third on is implementing naive method at lower levels -- that is for each warp, and do a scan for the sum of each warp and add them later to complete the whole scan for the block. In this implementation, `__shfl_up_sync()` is used to get the value of neighbor threads (within each warp) and sync them much faster than `__syncthreads()`, and suprisingly, this one's performance is the best among all my implementations.

## Performance Analysis

## Output

```
****************
** SCAN TESTS **
****************
    [   4  38  48  42  49  28  16  10   2  11  13  35  33 ...  29   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 113.673ms    (std::chrono Measured)
    [   4  42  90 132 181 209 225 235 237 248 261 296 329 ... 1643619535 1643619535 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 117.37ms    (std::chrono Measured)
    [   4  42  90 132 181 209 225 235 237 248 261 296 329 ... 1643619470 1643619498 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 65.9112ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 65.8544ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 6.77021ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 6.41149ms    (CUDA Measured)
    passed
==== work-efficient shared-memory scan, power-of-two ====
   elapsed time: 4.57613ms    (CUDA Measured)
    passed
==== work-efficient shared-memory scan, non-power-of-two ====
   elapsed time: 4.53162ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 3.26861ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 3.18669ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   1   2   3   2   2   1   3   3   1   1   1   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 152.485ms    (std::chrono Measured)
    [   2   1   2   3   2   2   1   3   3   1   1   1   2 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 149.303ms    (std::chrono Measured)
    [   2   1   2   3   2   2   1   3   3   1   1   1   2 ...   3   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 200.545ms    (std::chrono Measured)
    [   2   1   2   3   2   2   1   3   3   1   1   1   2 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 15.4787ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 15.2743ms    (CUDA Measured)
    passed
==== work-efficient shared-memory compact, power-of-two ====
   elapsed time: 13.3466ms    (CUDA Measured)
    passed
==== work-efficient shared-memory compact, non-power-of-two ====
   elapsed time: 13.3867ms    (CUDA Measured)
    passed

*****************************
** RADIX SORT TESTS **
*****************************
    [  17  14   0   9  39  35  16   1  12  37  21  11  34 ...  11   0 ]
==== cpu sort, power-of-two ====
   elapsed time: 1142.24ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
==== cpu sort, non-power-of-two ====
   elapsed time: 1177.62ms    (std::chrono Measured)
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ...  49  49 ]
==== work-efficient radix sort, power-of-two ====
   elapsed time: 462.555ms    (CUDA Measured)
    passed
==== work-efficient radix sort, non-power-of-two ====
   elapsed time: 461.25ms    (CUDA Measured)
    passed
==== work-efficient shared-memory radix sort, power-of-two ====
   elapsed time: 398.827ms    (CUDA Measured)
    passed
==== work-efficient shared-memory radix sort, non-power-of-two ====
   elapsed time: 398.199ms    (CUDA Measured)
    passed
```
