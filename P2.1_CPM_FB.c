#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define N 600000
#define G 200  

long V[N];
long R[G];
int A[G];

int rank_proces, num_processos;

void kmean(int fN, int fK, long fV[], long fR[], int fA[])
{
        int i,j,min,iter=0,aux;
        long dif;
        long t;
        long fS[G], fSp[G];
        int fAp[G], fragments[num_processos], despl[num_processos];

        if (rank_proces == 0) {
                aux = fN % num_processos;
                fN = fN / num_processos;
        }
        MPI_Bcast(&fN,1,MPI_INT,0,MPI_COMM_WORLD);
        if (rank_proces == 0) {
                fragments[0] = fN + aux;
                despl[0] = 0;
                for (i = 1; i < num_processos; i++) {
                        fragments[i] = fN;
                        despl[i] = despl[i-1] + fragments[i-1];
                }
                fN += aux;
        }

        int fD[fN];

        MPI_Scatterv(fV,fragments,despl,MPI_LONG,fV,fN,MPI_LONG,0,MPI_COMM_WORLD);

        do
        {
                MPI_Bcast(fR,fK,MPI_LONG,0,MPI_COMM_WORLD);

                for (i=0;i<fN;i++)
                {
			min = 0;
                        dif = abs(fV[i] -fR[0]);
                        for (j=1;j<fK;j++)
                                if (abs(fV[i] -fR[j]) < dif)
                                {
                                        min = j;
                                        dif = abs(fV[i] -fR[j]);
                                }
                        fD[i] = min;
                }

                for(i=0;i<fK;i++)
                        fSp[i] = fAp[i] = 0;

                for(i=0;i<fN;i++)
                {
                        fSp[fD[i]] += fV[i];
                        fAp[fD[i]] ++;
                }

                MPI_Reduce(fSp,fS,fK,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
                MPI_Reduce(fAp,fA,fK,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

                if (rank_proces == 0)
                {
                        dif = 0;
                        for(i=0;i<fK;i++)
                        {
                                t = fR[i];
                                if (fA[i]) fR[i] = fS[i]/fA[i];
                                dif += abs(t - fR[i]);
                        }
                        iter++;
                }

                MPI_Bcast(&dif,1,MPI_LONG,0,MPI_COMM_WORLD);
        } while(dif);
        if (rank_proces == 0)
                printf("iter %d\n",iter);
}

void qs(int ii, int fi, long fV[], int fA[])
{
        int i,f,j;
        long pi,pa,vtmp,vta,vfi,vfa;

        pi = fV[ii];
        pa = fA[ii];
        i = ii +1;
        f = fi;
        vtmp = fV[i];
        vta = fA[i];

        while (i <= f)
        {
                if (vtmp < pi) {
                        fV[i-1] = vtmp;
                        fA[i-1] = vta;
                        i ++;
                        vtmp = fV[i];
                        vta = fA[i];
                }
                else {
                        vfi = fV[f];
                        vfa = fA[f];
                        fV[f] = vtmp;
                        fA[f] = vta;
                        f --;
                        vtmp = vfi;
                        vta = vfa;
                }
        }
        fV[i-1] = pi;
        fA[i-1] = pa;

        if (ii < f) qs(ii,f,fV,fA);
        if (i < fi) qs(i,fi,fV,fA);
}

int main(int num_args, char* args[])
{
        int i;

        MPI_Init(&num_args, &args);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_proces);
        MPI_Comm_size(MPI_COMM_WORLD, &num_processos);

        if (rank_proces == 0) {
                for (i=0;i<N;i++) V[i] = (rand()%rand())/N;

                        // Primers candidats
                for (i=0;i<G;i++) R[i] = V[i];
        }

                // Calcular els G més representatius
        kmean(N,G,V,R,A);

        if (rank_proces == 0) {
                qs(0,G-1,R,A);

                for (i=0;i<G;i++)
                        printf("R[%d] : %ld te %d agrupats\n",i,R[i],A[i]);
        }

        MPI_Finalize();
        return(0);
}
