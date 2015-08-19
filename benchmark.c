#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif

#define MAX_SPEED 8.4  // definning Hopper Max Gflops/s per node

/* reference_dgemm wraps a call to the BLAS-3 routine DGEMM, via the standard FORTRAN interface - hence the reference semantics. */ 
#define DGEMM dgemm_
extern void DGEMM (char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*); 
void reference_dgemm (int N, double ALPHA, double* A, double* B, double* C)
{
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = N;
  int K = N;
  double BETA = 1.;
  int LDA = N;
  int LDB = N;
  int LDC = N;
  DGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}   

/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void square_dgemm (int, double*, double*, double*);

double wall_time ()
{
#ifdef GETTIMEOFDAY
  struct timeval t;
  gettimeofday (&t, NULL);
  return 1.*t.tv_sec + 1.e-6*t.tv_usec;
#else
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
#endif
}

void die (const char* message)
{
  perror (message);
  exit (EXIT_FAILURE);
}

void fill (double* p, int n)
{
  for (int i = 0; i < n; ++i)
    p[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
}

void fill_simple (double* p, int n)
{
  for (int i = 0; i < n; ++i)
    p[i] = i;
    //p[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
}

void absolute_value (double *p, int n)
{
  for (int i = 0; i < n; ++i)
    p[i] = fabs (p[i]);
}

/* The benchmarking program */
int main (int argc, char **argv)
{
  printf ("Description:\t%s\n\n", dgemm_desc);

  /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

  int test_sizes[] = 

  /* Multiples-of-32, +/- 1. Currently commented. */
  /* {31,32,33,63,64,65,95,96,97,127,128,129,159,160,161,191,192,193,223,224,225,255,256,257,287,288,289,319,320,321,351,352,353,383,384,385,415,416,417,447,448,449,479,480,481,511,512,513,543,544,545,575,576,577,607,608,609,639,640,641,671,672,673,703,704,705,735,736,737,767,768,769,799,800,801,831,832,833,863,864,865,895,896,897,927,928,929,959,960,961,991,992,993,1023,1024,1025}; */

  /* A representative subset of the first list. Currently uncommented. */ 
  { 4, 8,9, 16, 32, 48,  31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
    319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769 };

  int nsizes = sizeof(test_sizes)/sizeof(test_sizes[0]);

  /* assume last size is also the largest size */
  int nmax = test_sizes[nsizes-1];

  /* allocate memory for all problems */
  double* buf = NULL;
  buf = (double*) malloc (3 * nmax * nmax * sizeof(double));
  if (buf == NULL) die ("failed to allocate largest problem size");

  double Mflops_s[nsizes],per[nsizes],aveper;

  /* For each test size */
  for (int isize = 0; isize < sizeof(test_sizes)/sizeof(test_sizes[0]); ++isize)
  {
    /* Create and fill 3 random matrices A,B,C*/
    int n = test_sizes[isize];

    double* A = buf + 0;
    double* B = A + nmax*nmax;
    double* C = B + nmax*nmax;

    fill (A, n*n);
    fill (B, n*n);
    fill (C, n*n);

    /* Measure performance (in Gflops/s). */

    /* Time a "sufficiently long" sequence of calls to reduce noise */
    double Gflops_s, seconds = -1.0;
    double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
    for (int n_iterations = 1; seconds < timeout; n_iterations *= 2) 
    {
      /* Warm-up */
      square_dgemm (n, A, B, C);

      /* Benchmark n_iterations runs of square_dgemm */
      seconds = -wall_time();
      for (int it = 0; it < n_iterations; ++it)
	square_dgemm (n, A, B, C);
      seconds += wall_time();

      /*  compute Gflop/s rate */
      Gflops_s = 2.e-9 * n_iterations * n * n * n / seconds;
    }
  
    /* Storing Mflop rate and calculating percentage of peak */
    Mflops_s[isize] = Gflops_s*1000;
    per[isize] = Gflops_s*100/MAX_SPEED;

    printf ("Size: %d\tMflop/s: %8g\tPercentage:%6.2lf\n", n, Mflops_s[isize],per[isize]);

    /* Ensure that error does not exceed the theoretical error bound. */

    /* C := A * B, computed with square_dgemm */
    memset (C, 0, n * n * sizeof(double));
    square_dgemm (n, A, B, C);

    /* Do not explicitly check that A and B were unmodified on square_dgemm exit
     *  - if they were, the following will most likely detect it:   
     * C := C - A * B, computed with reference_dgemm */
    reference_dgemm(n, -1., A, B, C);

    /* A := |A|, B := |B|, C := |C| */
    absolute_value (A, n * n);
    absolute_value (B, n * n);
    absolute_value (C, n * n);

    /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_dgemm */ 
    reference_dgemm (n, -3.*DBL_EPSILON*n, A, B, C);

    /* If any element in C is positive, then something went wrong in square_dgemm */
    for (int i = 0; i < n * n; ++i)
      if (C[i] > 0)
	die("*** FAILURE *** Error in matrix multiply exceeds componentwise error bounds.\n" );
  }

  /* Calculating average percentage of peak reached by algorithm */
  aveper=0;
  for (int i=0; i<nsizes;i++)
    aveper+= per[i];
  aveper/=nsizes*1.0;
  
  /* Printing average percentage and grade to screen */
  printf("Average percentage of Peak = %g\n",aveper);  

  free (buf);

  return 0;
}
