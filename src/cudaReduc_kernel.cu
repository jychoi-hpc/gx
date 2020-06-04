int scheme=6;
int maxThreads=256;
int maxBlocks=64;

/*
  This version uses sequential addressing -- no divergence or bank conflicts.
*/

template <class T>
__global__ void reduce2 (T *g_idata, T *g_odata, unsigned int n)
{
  extern __shared__ T sdata[];
  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
 
  sdata[tid] = (i < n) ? g_idata[i] : 0;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduce2_partial (T *g_idata, T *g_odata, unsigned int n, unsigned int blocksize)
{
  extern __shared__ T sdata[];
  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blocksize + threadIdx.x;
 
  sdata[tid] = (i < n && tid<blocksize) ? g_idata[i] : 0;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce2_partial_complex (cuComplex *g_idata, cuComplex *g_odata, unsigned int n, unsigned int blocksize)
{
  extern __shared__ float sdata_real[];
  //extern __shared__ float sdata_comp[];
  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blocksize + threadIdx.x;
 
  sdata_real[tid] = (i < n && tid<blocksize) ? g_idata[i].x : 0.;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) sdata_real[tid] = sdata_real[tid] + sdata_real[tid + s];
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x].x = sdata_real[0];
 
  __syncthreads();

  sdata_real[tid] = (i < n && tid<blocksize) ? g_idata[i].y : 0.;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) sdata_real[tid] = sdata_real[tid] + sdata_real[tid + s];
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x].y = sdata_real[0];
}

/*
  This version uses n/2 threads --
  it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void reduce3 (T *g_idata, T *g_odata, unsigned int n)
{
  extern __shared__ T sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;
  if (i + blockDim.x < n) 
    mySum += g_idata[i+blockDim.x];  
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) sdata[tid] = mySum = mySum + sdata[tid + s];
    __syncthreads();
  }

  // write result for this block to global mem 
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
  This version uses nosync in final warp
*/
template <class T, unsigned int blockSize>
__global__ void reduce4 (T *g_idata, T *g_odata, unsigned int n)
{
  extern __shared__ T sdata[];
  unsigned int tid = threadIdx.x;
  //
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;
  if (i + blockDim.x < n) 
    mySum += g_idata[i+blockDim.x];  
  sdata[tid] = mySum;
  /* //
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
 
  sdata[tid] = (i < n) ? g_idata[i] : 0;
  */
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    //if (tid < s) sdata[tid] += sdata[tid + s];
    if (tid < s) sdata[tid] = mySum = mySum + sdata[tid + s];
    __syncthreads();
  }

  if (tid<32) {
    volatile T *smem = sdata;
    /*
    if (blockSize >= 64) { smem[tid] += smem[tid + 32]; }
    if (blockSize >= 32) { smem[tid] += smem[tid + 16]; }
    if (blockSize >= 16) { smem[tid] += smem[tid +  8]; }
    if (blockSize >=  8) { smem[tid] += smem[tid +  4]; }
    if (blockSize >=  4) { smem[tid] += smem[tid +  2]; }
    if (blockSize >=  2) { smem[tid] += smem[tid +  1]; }
    */
    if (blockSize >= 64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
    if (blockSize >= 32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
    if (blockSize >= 16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
    if (blockSize >=  8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
    if (blockSize >=  4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
    if (blockSize >=  2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
  This version is completely unrolled.  It uses a template parameter to achieve 
  optimal code for any (power of 2) number of threads.  This requires a switch 
  statement in the host code to handle all the different thread block sizes at 
  compile time.

  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
  In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
  If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void reduce5 (T *g_idata, T *g_odata, unsigned int n)
{
  extern __shared__ T sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;
  if (i + blockSize < n)
    mySum += g_idata[i+blockSize];  

  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) {
    if (tid < 256) sdata[tid] = mySum = mySum + sdata[tid + 256];
    __syncthreads(); 
  }
  if (blockSize >= 256) {
    if (tid < 128) sdata[tid] = mySum = mySum + sdata[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid <  64) sdata[tid] = mySum = mySum + sdata[tid +  64];
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T *smem = sdata;
    if (blockSize >= 64) smem[tid] = mySum = mySum + smem[tid + 32];
    if (blockSize >= 32) smem[tid] = mySum = mySum + smem[tid + 16];
    if (blockSize >= 16) smem[tid] = mySum = mySum + smem[tid +  8];
    if (blockSize >=  8) smem[tid] = mySum = mySum + smem[tid +  4];
    if (blockSize >=  4) smem[tid] = mySum = mySum + smem[tid +  2];
    if (blockSize >=  2) smem[tid] = mySum = mySum + smem[tid +  1];
  }

  // write result for this block to global mem 
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
  This version adds multiple elements per thread sequentially.  This reduces
  the overall cost of the algorithm while keeping the work complexity O(n) and
  the step complexity O(log n).
  (Brent's Theorem optimization)

  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
  In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
  If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6 (T *g_idata, T *g_odata, unsigned int n)
{
  extern __shared__ T sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the 
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n) mySum += g_idata[i+blockSize];  
    i += gridSize;
  }

  // each thread puts its local sum into shared memory 
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) {
    if (tid < 256) sdata[tid] = mySum = mySum + sdata[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) sdata[tid] = mySum = mySum + sdata[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid <  64) sdata[tid] = mySum = mySum + sdata[tid +  64];
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T* smem = sdata;
    if (blockSize >= 64) smem[tid] = mySum = mySum + smem[tid + 32];
    if (blockSize >= 32) smem[tid] = mySum = mySum + smem[tid + 16];
    if (blockSize >= 16) smem[tid] = mySum = mySum + smem[tid +  8];
    if (blockSize >=  8) smem[tid] = mySum = mySum + smem[tid +  4];
    if (blockSize >=  4) smem[tid] = mySum = mySum + smem[tid +  2];
    if (blockSize >=  2) smem[tid] = mySum = mySum + smem[tid +  1];
  }

  // write result for this block to global mem 
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void MAXreduce6 (T *g_idata, T *g_odata, unsigned int n)
{  
  extern __shared__ T sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;
  

  T myMax = 0;

  // we reduce multiple elements per thread.  The number is determined by the 
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    myMax = myMax > g_idata[i] ? myMax : g_idata[i];
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n) myMax = myMax > g_idata[i+blockSize] ? myMax : g_idata[i+blockSize] ;  
    i += gridSize;
  }

  // each thread puts its local sum into shared memory 
  sdata[tid] = myMax;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) {
    if (tid < 256) sdata[tid] = myMax = myMax > sdata[tid + 256] ? myMax : sdata[tid+256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) sdata[tid] = myMax = myMax > sdata[tid + 128] ? myMax : sdata[tid+128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid <  64) sdata[tid] = myMax = myMax > sdata[tid +  64] ? myMax : sdata[tid+64];
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T* smem = sdata;
    if (blockSize >= 64) smem[tid] = myMax = myMax > smem[tid + 32] ? myMax : smem[tid+32];
    if (blockSize >= 32) smem[tid] = myMax = myMax > smem[tid + 16] ? myMax : smem[tid+16];
    if (blockSize >= 16) smem[tid] = myMax = myMax > smem[tid + 8] ? myMax : smem[tid+8];
    if (blockSize >=  8) smem[tid] = myMax = myMax > smem[tid + 4] ? myMax : smem[tid+4];
    if (blockSize >=  4) smem[tid] = myMax = myMax > smem[tid + 2] ? myMax : smem[tid+2];
    if (blockSize >=  2) smem[tid] = myMax = myMax > smem[tid + 1] ? myMax : smem[tid+1];
  }

  // write result for this block to global mem 
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

unsigned int nextPow2 ( unsigned int x ) {
  // returns the smallest power of 2 which is not smaller than x
  --x;
  x |= x >> 1; // |= is a bit-wise OR substitution
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

bool isPow2 (unsigned int x) {
  return ((x&(x-1))==0);
}

template <class T>
void reduce_wrapper_partial (int size, int threads, int blocks, T *idata, T *odata)
{
  dim3 dimGrid (blocks,1,1);
  dim3 dimBlock (threads,1,1);
  int smem;

  smem = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

  reduce2_partial <<< dimGrid, dimBlock, smem >>> (idata, odata, size, size/blocks);

}
void reduce_wrapper_partial_complex (int size, int threads, int blocks, cuComplex *idata, cuComplex *odata)
{
  dim3 dimGrid (blocks,1,1);
  dim3 dimBlock (threads,1,1);
  int smem;

  smem = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

  reduce2_partial_complex <<< dimGrid, dimBlock, smem >>> (idata, odata, size, size/blocks);

}
template <class T>
void reduce_wrapper (int size, int threads, int blocks, T *idata, T *odata)
{
  dim3 dimGrid (blocks,1,1);
  dim3 dimBlock (threads,1,1);
  int smem;

  smem = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

  switch (scheme) {
  case 2:
    reduce2 <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
    break;
  case 3:
    reduce3 <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
    break;
  case 4:
    switch (threads) {
    case 512:
      reduce4<T,512> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 256:
      reduce4<T,256> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 128:
      reduce4<T,128> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 64:
      reduce4< T,64> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 32:
      reduce4< T,32> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 16:
      reduce4< T,16> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  8:
      reduce4<  T,8> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  4:
      reduce4< T, 4> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  2:
      reduce4< T, 2> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  1:
      reduce4< T, 1> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    }
    break;
  case 5:
    switch (threads) {
    case 512:
      reduce5<T,512> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 256:
      reduce5<T,256> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 128:
      reduce5<T,128> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 64:
      reduce5< T,64> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 32:
      reduce5< T,32> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case 16:
      reduce5< T,16> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  8:
      reduce5<  T,8> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  4:
      reduce5<  T,4> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  2:
      reduce5< T, 2> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    case  1:
      reduce5< T, 1> <<< dimGrid, dimBlock, smem >>> (idata, odata, size); break;
    }
    break;
  case 6:
    if (isPow2(size)) {
      switch (threads) {
      case 512:
	reduce6<T,512,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 256:
	reduce6<T,256,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 128:
	reduce6<T,128,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 64:
	reduce6< T,64,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 32:
	reduce6< T,32,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 16:
	reduce6< T,16,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  8:
	reduce6< T, 8,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  4:
	reduce6< T, 4,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  2:
	reduce6< T, 2,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  1:
	reduce6< T, 1,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      }
    } else {
      switch (threads) {
      case 512:
	reduce6<T,512,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 256:
	reduce6<T,256,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 128:
	reduce6<T,128,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 64:
	reduce6< T,64,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 32:
	reduce6< T,32,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 16:
	reduce6< T,16,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  8:
	reduce6< T, 8,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  4:
	reduce6< T, 4,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  2:
	reduce6< T, 2,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  1:
	reduce6< T, 1,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      }
    }
    break;
  }
}

template <class T>
void MAXreduce_wrapper (int size, int threads, int blocks, T *idata, T *odata)
{
  dim3 dimGrid (blocks,1,1);
  dim3 dimBlock (threads,1,1);
  int smem;

  smem = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

  switch (scheme) {

  case 6:
    if (isPow2(size)) {
      switch (threads) {
      case 512:
	MAXreduce6<T,512,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 256:
	MAXreduce6<T,256,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 128:
	MAXreduce6<T,128,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 64:
	MAXreduce6< T,64,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 32:
	MAXreduce6< T,32,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 16:
	MAXreduce6< T,16,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  8:
	MAXreduce6< T, 8,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  4:
	MAXreduce6< T, 4,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  2:
	MAXreduce6< T, 2,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  1:
	MAXreduce6< T, 1,true> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      }
    } else {
      switch (threads) {
      case 512:
	MAXreduce6<T,512,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 256:
	MAXreduce6<T,256,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 128:
	MAXreduce6<T,128,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 64:
	MAXreduce6< T,64,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 32:
	MAXreduce6< T,32,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case 16:
	MAXreduce6< T,16,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  8:
	MAXreduce6< T, 8,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  4:
	MAXreduce6< T, 4,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  2:
	MAXreduce6< T, 2,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      case  1:
	MAXreduce6< T, 1,false> <<< dimGrid, dimBlock, smem >>> (idata, odata, size);
	break;
      }
    }
    break;
  }
}

void getNumBlocksAndThreads (int n, int &blocks, int &threads)
{
  if (scheme < 3) {
    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  } else {
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  }

  if (scheme == 6) blocks = min (maxBlocks, blocks);
}

void getNumBlocksAndThreads_partial (int n, int &blocks, int &threads)
{
    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = nextPow2((n + threads - 1) / threads);
}

void getThreads (int n, int blocks, int &threads)
{
  if (scheme < 3) {
    threads = (n/blocks < maxThreads) ? nextPow2(n/blocks) : maxThreads;
    
  } else {
    threads = (n/blocks < maxThreads*2) ? nextPow2((n/blocks + 1)/ 2) : maxThreads;
  }
}

void getThreads_partial (int n, int blocks, int &threads)
{
    threads = (n/blocks < maxThreads) ? nextPow2(n/blocks) : maxThreads;
}

template <class X>
void swapargs (X &a, X &b)
{ // This function is from "Teach Yourself C++, Ex11.1.1.
  X temp;
  temp = a;
  a = b;
  b = temp;
}

/*
void swapargs_complex (cuComplex &a, cuComplex &b)
{ // This function is from "Teach Yourself C++, Ex11.1.1.
  cuComplex temp;
  temp = a;
  a = b;
  b = temp;
}
*/
