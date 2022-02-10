#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define frand(M) (M*(((double)rand())/RAND_MAX))

#define N 2000000  

double X[N];
double Y[N];

double cost (int nn, double vx[], double vy[], double t0, double t1)
{
	int i;
	double val,sum=0.0;

	#pragma omp parallel for private(i,val) shared(nn,t0,t1,vx,vy) reduction(+:sum) default(none) schedule(static)
	for(i=0;i<nn;i++)
	{
		val = t0 + t1*vx[i] - vy[i];
		sum += val * val;
	}
	sum /= 2*nn;
	return(sum);
}

int gradientDescent (int nn, double vx[], double vy[], double alpha, double *the0, double *the1)
{
	int i;
	double val;
	double z0,z1;
	double c=0,ca;
	double t0=*the0, t1=*the1;
	double a_n = alpha/nn;
	int iter = 0;
	double error = 0.000009; // cinc decimals
  
	do
	{
		z0 = z1 = 0.0;
		#pragma omp parallel for private(i,val) shared(nn,t0,t1,vx,vy) reduction(+:z0,z1) default(none) schedule(static)
		for(i=0;i<nn;i++)
		{
			val = t0 + t1*vx[i] - vy[i];
			z0 += val;
			z1 += val * vx[i];
		}
		t0 -= z0 * a_n;
		t1 -= z1 * a_n;
		iter++;
		ca = c;
		c = cost(nn,vx,vy,t0,t1);
	} while (fabs(c - ca) > error);
	*the0 = t0;
	*the1 = t1;
	return(iter);
}

int main()
{
	int i;
	double ct;
	double theta0=0, theta1=1;

	omp_set_num_threads(4);

	srand(1);
	for (i=0;i<N;i++) 
	{
		X[i] = frand(13);
		Y[i] = frand(9) + ((1.66 + (frand(0.9))) *  X[i]) * X[i] ;
	}

	//for (i=0;i<N;i++) printf("%g %g\n",X[i],Y[i]);
 
	i=gradientDescent (N, X, Y, 0.01, &theta0, &theta1);
	ct=cost(N,X,Y,theta0,theta1);
	printf ("(%d) Theta; %g, %g  cost: %g\n",i,theta0,theta1,ct);

	return(0);
}
