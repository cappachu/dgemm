

const char* dgemm_desc = "dgemm team 13";

#include <stdio.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>


#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 256
#endif

#define min(a,b) (((a)<(b))?(a):(b))


static void sse_4x4 (int lda, int K, double* A, double* B, double* C) {
    /* Performs Matrix Multiplication on 4x4 block
     * using SSE intrinsics 
     * load, update, store*/
  // A
  __m128d A_0X_A_1X, A_2X_A_3X;
  // B
  __m128d B_X0, B_X1, B_X2, B_X3;
  // C 
  __m128d C_00_C_10, C_20_C_30,
          C_01_C_11, C_21_C_31,
          C_02_C_12, C_22_C_32,
          C_03_C_13, C_23_C_33;

  // LOAD --------
  // load unaligned
  C_00_C_10 = _mm_loadu_pd(C              );
  C_20_C_30 = _mm_loadu_pd(C           + 2);
  C_01_C_11 = _mm_loadu_pd(C + lda        );
  C_21_C_31 = _mm_loadu_pd(C + lda     + 2);
  C_02_C_12 = _mm_loadu_pd(C + (2*lda)    );
  C_22_C_32 = _mm_loadu_pd(C + (2*lda) + 2);
  C_03_C_13 = _mm_loadu_pd(C + (3*lda)    );
  C_23_C_33 = _mm_loadu_pd(C + (3*lda) + 2);

  for (int k = 0; k < K; ++k) {
    // load aligned
    A_0X_A_1X = _mm_load_pd(A);
    A_2X_A_3X = _mm_load_pd(A+2);
    A += 4;
      
    // load unaligned
    B_X0 = _mm_loaddup_pd(B);
    B_X1 = _mm_loaddup_pd(B+1);
    B_X2 = _mm_loaddup_pd(B+2);
    B_X3 = _mm_loaddup_pd(B+3);
    B += 4;
    // UPDATE ---------
    // C := C + A*B
    C_00_C_10 = _mm_add_pd(C_00_C_10, _mm_mul_pd(A_0X_A_1X, B_X0));
    C_20_C_30 = _mm_add_pd(C_20_C_30, _mm_mul_pd(A_2X_A_3X, B_X0));
    C_01_C_11 = _mm_add_pd(C_01_C_11, _mm_mul_pd(A_0X_A_1X, B_X1));
    C_21_C_31 = _mm_add_pd(C_21_C_31, _mm_mul_pd(A_2X_A_3X, B_X1));
    C_02_C_12 = _mm_add_pd(C_02_C_12, _mm_mul_pd(A_0X_A_1X, B_X2));
    C_22_C_32 = _mm_add_pd(C_22_C_32, _mm_mul_pd(A_2X_A_3X, B_X2));
    C_03_C_13 = _mm_add_pd(C_03_C_13, _mm_mul_pd(A_0X_A_1X, B_X3));
    C_23_C_33 = _mm_add_pd(C_23_C_33, _mm_mul_pd(A_2X_A_3X, B_X3));
  }

  // STORE -------
  _mm_storeu_pd(C              , C_00_C_10);
  _mm_storeu_pd(C           + 2, C_20_C_30);
  _mm_storeu_pd(C + lda        , C_01_C_11);
  _mm_storeu_pd(C + lda     + 2, C_21_C_31);
  _mm_storeu_pd(C + (2*lda)    , C_02_C_12);
  _mm_storeu_pd(C + (2*lda) + 2, C_22_C_32);
  _mm_storeu_pd(C + (3*lda)    , C_03_C_13);
  _mm_storeu_pd(C + (3*lda) + 2, C_23_C_33);
}


void naive_helper (int lda, int K, double* A, double* B, double* C, int i, int j) {
    double cij = C[j*lda + i];
    for (int k = 0; k < K; ++k) {
        cij += A[k*lda + i] * B[j*lda + k];
    }
    C[j*lda + i] = cij;
}


void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  // largest multiple of 4 less than M
  int M4_max = (M>>2) << 2;
  // largest multiple of 4 less than N
  int N4_max = (N>>2) << 2;
  //printf("%i, %i\n", M4_max, N4_max);
  
  // pack and align data from A 
  double AA[M4_max*K]; // under allocate
  for(int m=0; m < M4_max; m+=4) {
      double *dst = &AA[m*K];
      double *src = A + m;
      for (int k = 0; k < K; ++k) {
          *dst     = *src;
          *(dst+1) = *(src+1);
          *(dst+2) = *(src+2);
          *(dst+3) = *(src+3);
          dst += 4;
          src += lda;
      }
  }
  // pack and align data from B
  double BB[N4_max*K]; // under allocate
  for(int n=0; n < N4_max; n+=4){
      double *dst = &BB[n*K];
      double *src_0 = B + n*lda;
      double *src_1 = src_0 + lda; 
      double *src_2 = src_1 + lda; 
      double *src_3 = src_2 + lda;
      for (int k = 0; k < K; ++k) {
          *dst++ = *src_0++;
          *dst++ = *src_1++;
          *dst++ = *src_2++;
          *dst++ = *src_3++;
      }
  }

  // compute 4x4's using SSE intrinsics 
  for (int i = 0; i < M4_max; i+=4){
    for (int j = 0; j < N4_max; j+=4){
        //printf("aa,bb: %f, %f\n", AA[i+1], BB[j+1]);
        sse_4x4(lda, K, &AA[i*K], &BB[j*K], &C[j*lda + i]);
    }
  }
  // compute remaining cells using naive dgemm
  // horizontal sliver
  if(M4_max!=M){
      for (int i=M4_max; i < M; ++i)
          for (int tmp=0; tmp < N; ++tmp) 
              naive_helper(lda, K, A, B, C, i, tmp);
  }
  // vertical sliver + bottom right corner 
  if(N4_max!=N){
      for (int j=N4_max; j < N; ++j)
          for (int tmp=0; tmp < M4_max; ++tmp) 
              naive_helper(lda, K, A, B, C, tmp, j);
  }
}



void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, lda-i);
	int N = min (BLOCK_SIZE, lda-j);
	int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
