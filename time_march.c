#include<stdio.h>
#include<stdlib.h>
#include<rfftw_threads.h>
#include<math.h>
#include"ns2d.h"

//function to evaluate jacobian
void find_jacobian_ft(double *omega, double* x_vel, double* y_vel)
{
	extern rfftwnd_plan p_for;
	extern int sys_size, nthreads;
	
	int i,j,ij;
	int y_size_f = sys_size/2+1;
	int y_size_r = sys_size+2;
	
	fftw_complex *vx_omega_ft = (fftw_complex*)x_vel;
	fftw_complex *vy_omega_ft = (fftw_complex*)y_vel;
	fftw_complex *jacobian_ft = (fftw_complex*)omega;
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, j, ij)
	for(i=0;i<sys_size;i++)
		for (j=0;j<y_size_r;j++)
		{
			ij = i*y_size_r + j;
			x_vel[ij] = x_vel[ij]*omega[ij]; 
			y_vel[ij] = y_vel[ij]*omega[ij];
		}
	//openmp_e
	
	rfftwnd_threads_one_real_to_complex(nthreads, p_for, x_vel, NULL);
	rfftwnd_threads_one_real_to_complex(nthreads, p_for, y_vel, NULL);
	
	//openmp_s
	# pragma omp parallel
	{
	# pragma omp for schedule(static) private(i, j, ij) 
	for(i=0;i<y_size_f;i++)
		for (j=0; j<y_size_f; j++)
	   	{
			ij = i*y_size_f + j;
			jacobian_ft[ij].re = (-(i*vx_omega_ft[ij].im + j*vy_omega_ft[ij].im));
			jacobian_ft[ij].im = i*vx_omega_ft[ij].re + j*vy_omega_ft[ij].re;
	   	}
	   	
	# pragma omp for schedule(static) private(i, j, ij)	
	for(i=y_size_f;i<sys_size;i++)//i greater than N/2 corresponds to kx = i-N (where N is sys_size)
	   	for(j=0;j<y_size_f;j++)
	   	{
	   		ij=i*y_size_f+j;
	   		jacobian_ft[ij].re = (-((i-sys_size)*vx_omega_ft[ij].im+j*vy_omega_ft[ij].im));
	   		jacobian_ft[ij].im = (i-sys_size)*vx_omega_ft[ij].re+j*vy_omega_ft[ij].re;
	   	}
	}
	//openmp_e
}

//function to solve using Runke Kutta 2
void solve_rk2 (fftw_complex *omega_ft, fftw_complex *omega_t, fftw_complex *x_vel_ft, fftw_complex *y_vel_ft, struct particle *part_main, double *colli_freq, struct pos_grid *grid, struct collision_list *colli_list)
{
	extern double *exp_dt;
	extern fftw_complex *fomega_ft;
	extern double dt;
	extern rfftwnd_plan p_inv;
	extern int sys_size, nthreads;
	
	double *omega = (double*)omega_ft;
	double *x_vel = (double*)x_vel_ft;
	double *y_vel = (double*)y_vel_ft;
	fftw_complex *jacobian_ft = omega_ft;	
	
	int i, j, ij;
	int y_size_f=sys_size/2+1;
	int y_size_r=sys_size+2;
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, j, ij)
	for(i=0;i<sys_size;i++)
	{
		for (j=0; j<y_size_f; j++)
	   	{
	   		ij = i*y_size_f + j;
	   		omega_t[ij].re = omega_ft[ij].re;
	   		omega_t[ij].im = omega_ft[ij].im;
	   	}
	}
	//openmp_e

	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, omega_ft, NULL);
	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, x_vel_ft, NULL);	
	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, y_vel_ft, NULL);      
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, j, ij)
	for(i=0;i<sys_size;i++)
	{
		for (j=0; j<y_size_r; j++)
		{
			ij = i*y_size_r + j;
			omega[ij] = omega[ij]/(sys_size*sys_size);	//normalising for FT inverse
			x_vel[ij] = x_vel[ij]/(sys_size*sys_size);
			y_vel[ij] = y_vel[ij]/(sys_size*sys_size);
		}
	}
	//openmp_e	

	update_part(x_vel, y_vel, part_main, colli_freq, grid, colli_list);
		
	find_jacobian_ft(omega, x_vel, y_vel);          		//update jacobian_ft 
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, j, ij)
	for(i=0;i<sys_size;i++)
	{
		for (j=0; j<y_size_f; j++)
	   	{	
	   		ij = i*y_size_f + j;
	   		omega_ft[ij].re = exp_dt[ij]*omega_t[ij].re + exp_dt[ij]*0.5*dt*(-jacobian_ft[ij].re+fomega_ft[ij].re);
			omega_ft[ij].im = exp_dt[ij]*omega_t[ij].im + exp_dt[ij]*0.5*dt*(-jacobian_ft[ij].im+fomega_ft[ij].im);
		}
	}
	//openmp_e		

	find_vel_ft( omega_ft, x_vel_ft, y_vel_ft);			
	
	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, x_vel_ft, NULL);
	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, y_vel_ft, NULL);
	
	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, omega_ft, NULL);
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, j, ij)
	for(i=0;i<sys_size;i++)
	{
		for (j=0; j<y_size_r; j++)
		{
			ij = i*y_size_r + j;
			omega[ij] = omega[ij]/(sys_size*sys_size);	//normalising for FT inverse
			x_vel[ij] = x_vel[ij]/(sys_size*sys_size);
			y_vel[ij] = y_vel[ij]/(sys_size*sys_size);
		}
	}
	//openmp_e
	
	find_jacobian_ft(omega, x_vel, y_vel);				//update jacobian_ft wrt w1FT 
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, j, ij)	
	for(i=0;i<sys_size;i++)
	{
		for (j=0; j<y_size_f; j++)
		{
			ij = i*y_size_f + j;
			omega_ft[ij].re = exp_dt[ij]*exp_dt[ij]*omega_t[ij].re+exp_dt[ij]*dt*(-jacobian_ft[ij].re+fomega_ft[ij].re);
			omega_ft[ij].im = exp_dt[ij]*exp_dt[ij]*omega_t[ij].im+exp_dt[ij]*dt*(-jacobian_ft[ij].im+fomega_ft[ij].im);
		}
	}
	//openmp_e

	omega_ft[0].re=0.0;
	omega_ft[0].im=0.0;
}
