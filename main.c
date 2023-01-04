/*This program solves NS equations in 2 dimension
* for a periodic velocity using the pseudo-
* spectral algorithm*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include<rfftw_threads.h>
#include"ns2d.h"
	
rfftwnd_plan p_for, p_inv;
double *exp_dt;
fftw_complex *fomega_ft;
int sys_size, part_num, nthreads, nthreads_omp, tau_num, colli_max;
const int grid_max=10000, grid_xmax=100;	//err check set into file
double dt, vis, mu, tau_min, tau_max, rho_ratio;
double tolerence=0.000001;	//err check set into file

//starts function MAIN() 
int main(int argc, char *argv[])
{
	
	int max_iter, i, j, k, ij, jk, kf;
	int ji, k1, flag, pflag, s_flag;
	double famp, epsilon_sum, epsilon, tau_flow, len_flow;
	double energy;
	FILE *fp, *fp_e;
	
	struct particle *part_main;
	struct pos_grid grid[grid_max];
	struct collision_list *colli_list;
	
	//creates directories to store data
	system("mkdir -p spectra");
	system("mkdir -p field");
	system("mkdir -p particle");
	
	fp = fopen(argv[1],"r");
	
	//reads the variables from input file
	fscanf(fp,"%d%d%lE%lE%lE%lE%d%d%d%d%d%d", &sys_size, &max_iter, &dt, &vis, &famp, \
	 &mu, &kf, &nthreads, &nthreads_omp, &flag, &pflag, &s_flag);
	
	fclose(fp);
	
	int y_size_f = sys_size/2+1;
	int y_size_r = sys_size+2;
	
	double scale = 1.0/(sys_size*sys_size);

	//Memory allocation
	double *omega 			= (double*)malloc(sys_size*(sys_size+2)*sizeof(double));
	fftw_complex *omega_ft		= (fftw_complex*)omega;
	fftw_complex *omega_t		= (fftw_complex*)malloc((sys_size*y_size_f)*sizeof(fftw_complex));
	double *x_vel 			= (double*)malloc(sys_size*(sys_size+2)*sizeof(double));
	fftw_complex *x_vel_ft		= (fftw_complex*)x_vel;		
	double *y_vel 			= (double*)malloc(sys_size*(sys_size+2)*sizeof(double));
	fftw_complex *y_vel_ft		= (fftw_complex*)y_vel;	
	double *e_spectra		= (double*)malloc(y_size_f*sizeof(double));
	int *den_state 			= (int*)malloc(y_size_f*sizeof(int));
	
	fomega_ft			= (fftw_complex*)malloc(sys_size*y_size_f*sizeof(fftw_complex));
	exp_dt				= (double*)malloc(sys_size*y_size_f*sizeof(double));

	//fftw variables for forward and inverse transform
	p_for = rfftw2d_create_plan(sys_size, sys_size, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE | FFTW_IN_PLACE);
	p_inv = rfftw2d_create_plan(sys_size, sys_size, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE | FFTW_IN_PLACE);
	
	double time=0.0;
	int t_flag=max_iter/s_flag;		//flags to store data
	char fname[35];				//stores filename
		
	fftw_threads_init();			//initializes threads for fftw
	omp_set_num_threads(nthreads_omp);	//sets number of threads
	
	if(argv[2][0]=='0')				//if no input omega file is given
	{						//initialize omega at t=0 by a periodic function
		init_den_state(den_state);
		init_omega(omega_ft, den_state);
	}
	
	else						//else read from the file
	{
		fp = fopen(argv[2],"r");
		for(i=0;i<sys_size;i++)
			for(j=0;j<y_size_f;j++)
			{
				ij = i*y_size_f+j;
				fscanf(fp,"%lf%lf",&omega_ft[ij].re,&omega_ft[ij].im);
			}
		init_den_state(den_state);
		fclose(fp);
	}

	fp = fopen(argv[3],"r");	//reads particle variables from file
	fscanf(fp,"%d%lf%lf%lf%d", &part_num, &tau_min, &tau_max, &rho_ratio, &tau_num);
	fclose(fp);
	
	colli_max=100*part_num;	//maximum number of collisions within dt
	
	//allocate memory for particle position and velocity variables
	part_main = (struct particle*)malloc(part_num*sizeof(struct particle));
	double *colli_freq = (double*)malloc(tau_num*tau_num*sizeof(double));
	colli_list = (struct collision_list*)malloc(colli_max*sizeof(struct collision_list));
	double tau[tau_num];
	
	for(i=0;i<tau_num;i++)
		tau[i]=tau_min+(double)i*(tau_max-tau_min)/tau_num;
			
	if(argv[4][0]=='0')		//if no input particle data is given initializes by a random function
		init_part(part_main);
		
	else			//else read from file
	{
		fp=fopen(argv[4],"r");
		for(j=0;j<part_num;j++)
		{
			fscanf(fp," %lf%lf%lf%lf%lf%lf%d\n", &part_main[j].x_pos, &part_main[j].y_pos\
			, &part_main[j].x_vel, &part_main[j].y_vel, &part_main[j].radius, &part_main[j].tau, &part_main[j].tau_index);
			part_main[j].part_index=j;
		}
		fclose(fp);
	}
	
	for(i=0;i<tau_num;i++)		//initializes the collision frequency to zero
		for(j=0;j<tau_num;j++)
		{
			ij=i*tau_num+j;
			colli_freq[ij]=0.0;
		}
	
	update_grid(part_main, grid);	//updates position grid and assign each particle to a grid
	gen_force(famp, kf);		//generates stirring force
	
	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, omega_ft, NULL);//inverse fourier transforms and fourier transforms
	for(j=0;j<sys_size;j++)						//the initialized omega
		for(k=0;k<y_size_r;k++)
		{	
			jk=j*y_size_r+k;
			omega[jk]*=scale;
		}
	rfftwnd_threads_one_real_to_complex(nthreads, p_for, omega, NULL);
	find_vel_ft( omega_ft, x_vel_ft, y_vel_ft);
	
	fp_e=fopen("spectra/energy.dat","w");	//file to store energy vs time data
	fprintf(fp_e,"#time-tot_Energy\n");
	
	//time marching
	for(k=0;k<s_flag;k++)
	{
		if(pflag==0)		//if pflag is set zero, store the vorticity profile
		{
			rfftwnd_threads_one_complex_to_real(nthreads, p_inv, omega_ft, NULL);
			for(j=0;j<sys_size;j++)						
				for(k1=0;k1<y_size_r;k1++)
				{	
					jk=j*y_size_r+k1;
					omega[jk]*=scale;
				}
			
			sprintf(fname,"field/velProfile%.3d.dat",k);		
			fp=fopen(fname,"w");
			fprintf(fp,"#Vorticity profile at time %f\n#W\n",time);		
			for(j=0;j<sys_size;j=j+flag)	//writes data to file
			{
				for(k1=0;k1<y_size_r;k1=k1+flag)
				{
					jk=j*y_size_r+k1;
					fprintf(fp,"%f %f %E\n",(double)j*M_PI*2/sys_size, \
						(double)k1*M_PI*2/sys_size, omega[jk]);
				}
				fprintf(fp,"\n");
			}
			fclose(fp);
			rfftwnd_threads_one_real_to_complex(nthreads, p_for, omega, NULL);
		}
		epsilon_sum+=find_epsilon(omega_ft);
		epsilon=epsilon_sum/(k+1);
		tau_flow=sqrt(vis/epsilon);
		len_flow=sqrt(sqrt(vis*vis*vis/epsilon));
		find_e_spectra(x_vel_ft, y_vel_ft, e_spectra);//calculates energy spectra
		sprintf(fname,"spectra/eSpectra%.3d.dat",k);		
		fp=fopen(fname,"w");
		fprintf(fp,"#Energy spectra at time %f\n",time);
		for(j=1;j<y_size_f;j++)
			fprintf(fp, "%d %E\n", j, e_spectra[j]);
		fclose(fp);
	
		energy=find_energy(e_spectra);
		fprintf(fp_e,"%f %f %E %f %E\n", time, energy, epsilon, tau_flow, len_flow);
		fflush(fp_e);

		fp=fopen("init_flow.dat","w");
		for(i=0;i<sys_size;i++)	//writes data to file
			for(j=0;j<y_size_f;j++)
			{
				ij=i*y_size_f+j;
				fprintf(fp, "%E %E ", omega_ft[ij].re, omega_ft[ij].im);
			}
		fclose(fp);
		
		sprintf(fname,"particle/particle%.3d.dat",k);
		fp=fopen(fname,"w");
		fprintf(fp, "#Particle position at time %f\n", time);
		for(j=0;j<part_num;j++)
			fprintf(fp,"%f %f %f %f %f\n", part_main[j].tau/tau_flow, part_main[j].x_pos, \
			part_main[j].y_pos, part_main[j].x_vel, part_main[j].y_vel);
		fclose(fp);
		
		fp=fopen("init_part.dat","w");
		for(j=0;j<part_num;j++)
			fprintf(fp," %f %f %f %f %f %f %d\n",part_main[j].x_pos,part_main[j].y_pos\
			, part_main[j].x_vel, part_main[j].y_vel, part_main[j].radius, part_main[j].tau, part_main[j].tau_index);
		fclose(fp);
		
		sprintf(fname,"particle/colli_freq%.3d.dat",k);
		fp=fopen(fname,"w");
		for(i=0;i<tau_num;i++)
		{
			for(j=0;j<tau_num;j++)
			{
				ij=i*tau_num+j;
				ji=j*tau_num+i;
				fprintf(fp," %f %f %f\n", tau[i]/tau_flow, tau[j]/tau_flow, (colli_freq[ij]+colli_freq[ji])/2);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		
		for(i=0;i<tau_num;i++)
			for(j=0;j<tau_num;j++)
			{
				ij=i*tau_num+j;
				colli_freq[ij]=0.0;
			}
				
		for(i=0;i<t_flag;i++)	
		{	
			/*Solves the differential in fourier space using Runka Kutta 2
			takes curent omega_ft, x_velocity_ft and y_velocity_ft and gives the updated omega_ft
			omega_t is a temporary variable. Note that current x_vel_ft and y_vel_ft gets destroyed*/
			solve_rk2(omega_ft, omega_t, x_vel_ft, y_vel_ft, part_main, colli_freq, grid, colli_list);    
			find_vel_ft( omega_ft, x_vel_ft, y_vel_ft);
		}
		time=time+dt*t_flag;	
	}
	fclose(fp_e);		

	fp=fopen("init_flow.dat","w");
	for(i=0;i<sys_size;i++)	//writes data to file
		for(j=0;j<y_size_f;j++)
		{
			ij=i*y_size_f+j;
			fprintf(fp, "%E %E ", omega_ft[ij].re, omega_ft[ij].im);
		}
	fclose(fp);
	
	fp=fopen("init_part.dat","w");
	for(j=0;j<part_num;j++)
		fprintf(fp," %f %f %f %f %f %f %d\n",part_main[j].x_pos,part_main[j].y_pos\
		, part_main[j].x_vel, part_main[j].y_vel, part_main[j].radius, part_main[j].tau, part_main[j].tau_index);
	fclose(fp);
	
	fp=fopen("particle/colli_freq.dat","w");
	for(i=0;i<tau_num;i++)
	{
		for(j=0;j<tau_num;j++)
		{
			ij=i*tau_num+j;
			ji=j*tau_num+i;
			fprintf(fp," %f %f %f\n", tau[i], tau[j], (colli_freq[ij]+colli_freq[ji])/2);
			
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
		
	rfftwnd_destroy_plan(p_for);	//free heap variables
	rfftwnd_destroy_plan(p_inv);

	free(omega);
	free(omega_t);
	free(x_vel);
	free(y_vel);
	free(exp_dt);
	free(fomega_ft);
	free(den_state);
	free(e_spectra);
	free(part_main);
	free(colli_freq);
	
	return 0;
}//end of MAIN()
