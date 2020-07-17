# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

int main ( void )
{

//Initialise

int a[500][500];
int b[500][500];
int c[500][500];
int i,j,k,jj,kk;
int n = 500;
int size = n;
int block_size = 30;
int thread_num;
double wtime, Seq_time;


printf ( "\n" );
printf ( "CA670  Assignment 2: Matrix multiplication \n" );

thread_num = omp_get_max_threads ( );

printf ( "\n" );
printf ( "  The number of processors available = %d\n", omp_get_num_procs ( ) );
printf ( "  The number of threads available    = %d\n", thread_num );
printf ( "  The matrix order N                 = %d * %d\n", n,n );
printf ( "  The block size is                  = %d", block_size);

//  Loop 1: Initialise Matrix A

    # pragma omp parallel shared ( a, b, c, n ) private ( i, j, k )
    {
    # pragma omp for
    for ( i = 0; i < n; i++ )
    {
        for ( j = 0; j < n; j++ )
        {
        a[i][j] = rand();
        }
    }
    
//    Loop 2: Initialise Matrix B
    
    # pragma omp for
    for ( i = 0; i < n; i++ )
    {
        for ( j = 0; j < n; j++ )
        {
        b[i][j] = rand();
        }
    } 
    }

//  Normal execution - sequential execution
  
  printf ( "\n" );
  wtime = omp_get_wtime ( );
    for ( i = 0; i < n; i++ )
    {
        for ( j = 0; j < n; j++ )
        {
        c[i][j] = 0.0;
        for ( k = 0; k < n; k++ )
        {
            c[i][j] = c[i][j] + a[i][k] * b[k][j];
        }
        }
    } 
  Seq_time = omp_get_wtime ( ) - wtime;
  printf ( "  Normal sequential execution Elapsed seconds = %g\n", Seq_time );  

//  OpenMP parallel for construct

  wtime = omp_get_wtime ( );
  # pragma omp parallel shared ( a, b, c, n ) private ( i, j, k )
    {
        # pragma omp for
        for ( i = 0; i < n; i++ )
        {
            for ( j = 0; j < n; j++ )
            {
            c[i][j] = 0.0;
            for ( k = 0; k < n; k++ )
            {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
            }
        } 
    }
    wtime = omp_get_wtime ( ) - wtime;
    printf ( "  Multithread normal matrix multiplication Elapsed seconds = %g (%g times)\n", wtime, (Seq_time/wtime) ); 

//  Matrix tiling approach
  wtime = omp_get_wtime ( );
	int tmp;
	for (int jj = 0; jj < size; jj += block_size)
	{
		for (int kk = 0; kk < size; kk += block_size)
		{
			for (int i = 0; i < size; i++)
			{
				for (int j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
				{
					tmp = 0;
					for (int k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
					{
						tmp += a[i][k] * b[k][j];
					}
					c[i][j] += tmp;
				}
			}
		}
	}
  wtime = omp_get_wtime ( ) - wtime;
  printf ( "  Blocked Matrix multipilication elapsed seconds = %g (%g times)\n", wtime, (Seq_time/wtime) );
  
//  Matrix tiling with OpenMP parallel for construct 

int chunk = 1;
#pragma omp parallel shared(a, b, c, size, chunk) private(i, j, k, jj, kk, tmp)
	{
    wtime = omp_get_wtime ( );
		#pragma omp for schedule (static, chunk)
		for (jj = 0; jj < size; jj += block_size)
		{
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						tmp = 0;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							tmp += a[i][k] * b[k][j];
						}
						c[i][j] += tmp;
					}
				}
			}
		}
	}

  wtime = omp_get_wtime ( ) - wtime;
  printf ( "  Multiple threads Blocked Matrix multiplication Elapsed seconds = %g (%g times)\n", wtime, (Seq_time/wtime) );




  return 0;
}
