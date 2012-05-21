#include "mex.h"
#include "matrix.h"
#include "emdL1.h"
//#include "emd_hist.h"

#define ROUND(x)	int((x)+.5)

/*-----------------------------------------------------------------------
[dis] = EMDL1_m( h1, h2, n1, n2, n3 )
	
input:
	h1, h2		: two histograms
output:
	EMD_L1 distance b/w h1 and h2
note:
	For now, only two and three dimensional histograms are supported
 ------------------------------------------------------------------------*/

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    /* Analyze input data and parameters*/
	double *h1	= mxGetPr(prhs[0]);
	double *h2	= mxGetPr(prhs[1]);
	//int		nC	= mxGetN(prhs[0]);
	//int		nR	= mxGetM(prhs[0]);

	int		n1	= ROUND(*mxGetPr(prhs[2]));
	int		n2	= ROUND(*mxGetPr(prhs[3]));
	int		n3	= nrhs>4 ? ROUND(*mxGetPr(prhs[4])) : 1;
    

	/* call emd function */
	EmdL1	em;			// EMD_L1 class
	double	d	= -1.;
	if(n3<=1)
		d	= em.EmdDist(h1,h2,n1,n2);			// 2D EMD_L1
	else
		d	= em.EmdDist(h1,h2,n1,n2,n3);		// 3D EMD_L1

	//printf("dis=%lf, n1=%d, n2=%d, n3=%d \n", d, n1, n2, n3);

	/* return */
	mxArray*	pMxD = mxCreateDoubleMatrix(1,1,mxREAL);
	*(mxGetPr(pMxD))	= d;
	plhs[0]	= pMxD;
}

