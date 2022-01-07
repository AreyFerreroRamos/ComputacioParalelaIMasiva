#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define frand(M) (M*(((double)rand())/RAND_MAX))

#define N 5000000  

int rang_proces, num_processos;

double X[N];
double Y[N];

double z0p;

double cost (int nn, double vx[], double vy[], double t0, double t1)
{
        int i;
        double val,sum=0.0;

        for(i=0;i<nn;i++)
        {
                val = t0 + t1*vx[i] - vy[i];
                sum += val * val;
        }
        
        sum /= 2*N;
        return(sum);
}

int gradientDescent (int nn, double vx[], double vy[], double alpha, double *the0, double *the1)
{    
        int i; 
        double val; 
        double z0, z1, z1p;
        double c = 0, cp, ca;
        double t0=*the0, t1=*the1;
        double a_n = alpha/N;
        int iter = 0;
        double error = 0.000009; // cinc decimals

        do
        { 
                z0p = z1p = 0.0;
                for(i=0;i<nn;i++)
                {
                        val = t0 + t1*vx[i] - vy[i];
                        z0p += val;
                        z1p += val * vx[i];
                }

                MPI_Allreduce(&z0p,&z0,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                MPI_Allreduce(&z1p,&z1,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

                t0 -= z0 * a_n;
                t1 -= z1 * a_n;
                iter++;

                ca = c;
                cp = cost(nn,vx,vy,t0,t1);

                MPI_Allreduce(&cp,&c,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        } while (fabs(c - ca) > error);
        *the0 = t0;
        *the1 = t1;
        return(iter);
}

int main(int num_args, char * args[])
{
        int i, aux, nn;
        double ct, ctp;
        double theta0=0, theta1=1;

        MPI_Init(&num_args, &args);
        MPI_Comm_rank(MPI_COMM_WORLD, &rang_proces);
        MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

        int fragments[num_processos], despl[num_processos];

        if (rang_proces == 0)
        {
                srand(1);
                for (i=0;i<N;i++)
                {
                        X[i] = frand(13);
                        Y[i] = frand(9) + ((1.66 + (frand(0.9))) *  X[i]) * X[i];
                }
        }

        nn = N / num_processos;

        if (rang_proces == 0) {
                aux = N % num_processos;
                fragments[0] = nn + aux;
                despl[0] = 0;
                for (i=1;i<num_processos;i++) {
                        fragments[i] = nn;
                        despl[i] = fragments[i-1] + despl[i-1];
                }
                nn += aux;
        }

        MPI_Scatterv(X,fragments,despl,MPI_DOUBLE,X,nn,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Scatterv(Y,fragments,despl,MPI_DOUBLE,Y,nn,MPI_DOUBLE,0,MPI_COMM_WORLD);

        //for (i=0;i<N;i++) printf("%g %g\n",X[i],Y[i]);

        i=gradientDescent(nn, X, Y, 0.01, &theta0, &theta1);
        ctp=cost(nn,X,Y,theta0,theta1);

        MPI_Reduce(&ctp,&ct,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

        if (rang_proces == 0)
                printf ("(%d) Theta; %g, %g  cost: %g\n",i,theta0,theta1,ct);

        MPI_Finalize();
        return(0);
}
