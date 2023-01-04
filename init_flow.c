#include<math.h>
#include<stdio.h>
#include<rfftw_threads.h>
#include<time.h>
#include"ns2d.h"

//function to initialize time increment and density of states
void init_den_state(int *den_state)
{
	extern int sys_size;
	extern double dt, vis, mu;
	extern double *exp_dt;
	
	int i, j, k, ij;
	int y_size_f=sys_size/2+1;
	int y_alias = sys_size/3;
	
	for(i=0;i<y_size_f;i++)
		den_state[i]=0;

	for(i=0;i<y_size_f;i++)
		for(j=0;j<y_size_f;j++)
		{
			k=(int)sqrt(i*i+j*j);
			if(k<y_size_f) den_state[k]++;
		}
		
	for(k=0;k<y_size_f;k++)
		den_state[k]*=4;

	for(i=0;i<y_size_f;i++)
		for(j=0;j<y_size_f;j++)
		{
			ij=i*y_size_f+j;
			exp_dt[ij]=exp(-(vis*(i*i+j*j)+mu)*dt/2);
			
			if((i*i+j*j)>y_alias*y_alias)
				exp_dt[ij]=0.0;
		}
	 
	for(i=y_size_f;i<sys_size;i++)//i greater than N/2 corresponds to kx = i-N (where N is sys_size)
		for(j=0;j<y_size_f;j++)
		{
		 	ij=i*y_size_f+j;
		 	exp_dt[ij]=exp(-(vis*((sys_size-i)*(sys_size-i)+j*j)+mu)*dt/2);
		 	
		 	if(((sys_size-i)*(sys_size-i)+j*j)>y_alias*y_alias)
		 		exp_dt[ij]=0.0;
		} 
}

//initializes omega in fourier space
void init_omega(fftw_complex *omega_ft, int *den_state)
{
	extern int sys_size;
	
	int i, j, k, ij, kSqr;
	int y_size_f = sys_size/2+1;
	
	double tempomega;
	
	srand(time(NULL));//seeds random function using current time
	
	for(i=0;i<y_size_f;i++)
		for(j=0;j<y_size_f;j++)
		{
			ij=i*y_size_f+j;
			kSqr=i*i+j*j;
			k=(int)sqrt(kSqr);
		
			if(k<10&&k<y_size_f)
			{
				tempomega= kSqr*kSqr*exp(-kSqr*kSqr/2);
				tempomega/=den_state[k];
				tempomega*=(double)sys_size*sys_size;
					
				omega_ft[ij].re=tempomega*cos(2.0*M_PI*rand()/RAND_MAX);
				omega_ft[ij].im=tempomega*sin(2.0*M_PI*rand()/RAND_MAX);
			}
			else
			{
				omega_ft[ij].re=0;
				omega_ft[ij].im=0;
			}
		}
	for(i=y_size_f;i<sys_size;i++)//i greater than N/2 corresponds to kx = i-N (where N is sys_size)
		for(j=0;j<y_size_f;j++)
	 	{
	 		ij=i*y_size_f+j;
			kSqr=(sys_size-i)*(sys_size-i)+j*j;
			k=(int)sqrt(kSqr);
		
			if(k<10)
			{
				tempomega= kSqr*kSqr*exp(-kSqr*kSqr/2);
				tempomega/=den_state[k];
				tempomega*=(double)sys_size*sys_size;
						
				omega_ft[ij].re=tempomega*cos(2.0*M_PI*rand()/RAND_MAX);
				omega_ft[ij].im=tempomega*sin(2.0*M_PI*rand()/RAND_MAX);
			}
			else
			{
				omega_ft[ij].re=0;
				omega_ft[ij].im=0;
			}
		}
}

//Generates stirring fomega
void gen_force(double famp, int kf)
{
	extern fftw_complex *fomega_ft;
	extern int sys_size, nthreads;
	extern rfftwnd_plan p_for;
	
	int y_size_r=sys_size+2;
	int i, j, ij;

	double *fomega = (double*)fomega_ft;
	
	for(i=0;i<sys_size;i++)
		for(j=0;j<y_size_r;j++)
		{
			ij = i*y_size_r+j;
			fomega[ij]=-kf*famp*cos(kf*2.0*M_PI*i/(sys_size));
		}
	rfftwnd_threads_one_real_to_complex(nthreads, p_for, fomega, 0);

}
