#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 600000
#define G 200  

long V[N];
long R[G];
int A[G];

void kmean(int fN, int fK, long fV[], long fR[], int fA[])
{
	int i,j,min,iter=0;
	long dif,t;
	long fS[G];
	int fD[N];

	do
	{
		#pragma omp parallel for private(i,min,dif,j) shared(fN,fV,fR,fK,fD) default(none) schedule(static)
		for (i=0;i<fN; i++)
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
			fS[i] = fA[i] = 0;

		#pragma omp parallel for private(i) shared(fN,fD,fV) reduction(+:fS[:G],fA[:G]) default(none) schedule(static)
		for(i=0;i<fN;i++)
		{
			fS[fD[i]] += fV[i];
			fA[fD[i]] ++;
		}

		dif = 0;
		for(i=0;i<fK;i++)
		{
			t = fR[i];
			if (fA[i]) fR[i] = fS[i]/fA[i];
			dif += abs(t - fR[i]);
		}
		iter ++;
	} while(dif);

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

int main()
{
	int i;

	omp_set_num_threads(4);

	for (i=0;i<N;i++) V[i] = (rand()%rand())/N;

	// primers candidats
	for (i=0;i<G;i++) R[i] = V[i];

	// calcular els G mes representatius
	kmean(N,G,V,R,A);

	qs(0,G-1,R,A);

	for (i=0;i<G;i++) 
 		printf("R[%d] : %ld te %d agrupats\n",i,R[i],A[i]);

	return(0);
}
