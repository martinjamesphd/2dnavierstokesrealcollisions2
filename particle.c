#include<stdio.h>
#include<time.h>
#include<rfftw_threads.h>
#include<math.h>
#include"ns2d.h"

//function to intialize particle positions
void init_part(struct particle *part_main)
{
	extern int part_num;
	int i, j, part_i, index=part_num/tau_num;
	
	srand(time(NULL));
	
	for(j=0;j<tau_num;j++)
	for(i=0;i<index;i++)
	{
		part_i=j*index+i;
		part_main[part_i].part_index=part_i;
		part_main[part_i].x_pos=2.0*M_PI*rand()/RAND_MAX;
		part_main[part_i].y_pos=2.0*M_PI*rand()/RAND_MAX;
		part_main[part_i].x_vel=0;
		part_main[part_i].y_vel=0;
		part_main[part_i].tau_index=j;
		part_main[part_i].tau=tau_min+(double)part_main[part_i].tau_index*(tau_max-tau_min)/(double)tau_num;
		part_main[part_i].radius=sqrt(vis*tau_min*4.5/rho_ratio);
		part_main[part_i].rho_ratio=rho_ratio*part_main[part_i].tau/tau_min;
	}
}
	
//function to update particle position and velocity
void update_part(double *x_vel_f, double *y_vel_f, struct particle *part_main, double *colli_freq, struct pos_grid *grid, struct collision_list *colli_list)
{
	extern int part_num;
	extern double dt;
	
	double x_vel_inter, y_vel_inter;
	double delta_t, time_left;
		
	int i, part_one, part_two, index1, index2, colli_num=0;

	//openmp_s
	# pragma omp parallel for schedule(static) private(i, x_vel_inter, y_vel_inter)
	for(i=0;i<part_num;i++)
	{
		//interpolates fluid velocity at particle positions
		x_vel_inter = linear_interp(part_main[i].x_pos, part_main[i].y_pos, x_vel_f);
		y_vel_inter = linear_interp(part_main[i].x_pos, part_main[i].y_pos, y_vel_f);
		
		//RK step 1
		part_main[i].x_pos_next=part_main[i].x_pos+dt*part_main[i].x_vel;
		part_main[i].y_pos_next=part_main[i].y_pos+dt*part_main[i].y_vel;
		
		part_main[i].x_vel_next=part_main[i].x_vel-dt*(part_main[i].x_vel-x_vel_inter)/part_main[i].tau;
		part_main[i].y_vel_next=part_main[i].y_vel-dt*(part_main[i].y_vel-y_vel_inter)/part_main[i].tau;
		
		
		//periodic boundary
		if(part_main[i].x_pos_next<0)
			part_main[i].x_pos_next+=2*M_PI;
		if(part_main[i].x_pos_next>=2*M_PI)
			part_main[i].x_pos_next-=2*M_PI;
		
		if(part_main[i].y_pos_next<0)
			part_main[i].y_pos_next+=2*M_PI;
		if(part_main[i].y_pos_next>=2*M_PI)
			part_main[i].y_pos_next-=2*M_PI;
	}
	//openmp_e
	colli_num=detect_collisions(part_main, grid, colli_list);
	
	delta_t=colli_list[0].colli_time;
	part_one=colli_list[0].part_one;
	part_two=colli_list[0].part_two;
	time_left=dt-delta_t;
	
	while((time_left>tolerence)&&(colli_num>0))
	{
		index1 = part_main[part_one].tau_index;
		index2 = part_main[part_two].tau_index;
		colli_freq[index1*tau_num+index2]++;
		update_for_collision(part_one, part_two, time_left, part_main, x_vel_f, y_vel_f);
		colli_num=delete_in_colli_list(part_one, part_two, colli_list, colli_num);
		colli_num=find_other_collisions(part_main, part_one, part_two, \
		 grid, time_left, colli_num, colli_list);
		delta_t=colli_list[0].colli_time;
		part_one=colli_list[0].part_one;
		part_two=colli_list[0].part_two;
		time_left=dt-delta_t;
	}
	
	# pragma omp parallel for schedule(static) private(i)
	for(i=0;i<part_num;i++)
	{
		part_main[i].x_pos=part_main[i].x_pos_next;
		part_main[i].y_pos=part_main[i].y_pos_next;
		part_main[i].x_vel=part_main[i].x_vel_next;
		part_main[i].y_vel=part_main[i].y_vel_next;
	}
	
	update_grid(part_main, grid);
}

int delete_in_colli_list(int part_one, int part_two, struct collision_list *colli_list, int colli_num)
{
	int i;
	
	for(i=0;i<colli_num;i++)
	{
		if((colli_list[i].part_one==part_one)||(colli_list[i].part_two==part_one)\
		||(colli_list[i].part_one==part_two)||(colli_list[i].part_two==part_two))
		{
			colli_list[i]=colli_list[colli_num-1];
			i--;
			colli_num--;
		}
	}
	
	return colli_num;
}

int detect_collisions(struct particle *part_main, struct pos_grid *grid, struct collision_list *colli_list)
{
	extern double dt;
	extern int part_num;
	int i, j=0;
	double x1, y1, radius_sum;
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, x1, y1, radius_sum)
	for(i=0;i<grid_max;i++)
	{
		int x_index = i/grid_xmax;
		int y_index = i%grid_xmax;
		
		double colli_time, vx, vy;
		struct collision_list colli_temp;
				
		int grid_11 = x_index*grid_xmax + y_index;
		int grid_01 = (mod((x_index-1),grid_xmax))*grid_xmax + y_index;
		int grid_10 = (x_index)*grid_xmax + (mod((y_index - 1),grid_xmax));
		int grid_00 = (mod((x_index-1),grid_xmax))*grid_xmax + (mod((y_index - 1),grid_xmax));

		struct particle *part_one;
		struct particle *part_two;
		
		part_one=grid[grid_11].beg;
		while(part_one!=NULL)
		{
			part_two=part_one->next;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				vx=part_two->x_vel-part_one->x_vel;
				vy=part_two->y_vel-part_one->y_vel;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					if(x1*vx+y1*vy<0)
					{
						colli_temp.part_one=part_one->part_index;
						colli_temp.part_two=part_two->part_index;
						colli_temp.colli_time=0;
						# pragma omp critical(TWO)
						colli_list[j++]=colli_temp;
					}
					part_two=part_two->next;
					continue;
				}
			
				colli_time = closest_dist(x1, y1, vx, vy, radius_sum);
				
				if(colli_time<dt)
				{	
					colli_temp.part_one=part_one->part_index;
					colli_temp.part_two=part_two->part_index;
					colli_temp.colli_time=colli_time;
					# pragma omp critical(TWO)
					colli_list[j++]=colli_temp;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_00].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_11].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				vx=part_two->x_vel-part_one->x_vel;
				vy=part_two->y_vel-part_one->y_vel;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					if(x1*vx+y1*vy<0)
					{
						colli_temp.part_one=part_one->part_index;
						colli_temp.part_two=part_two->part_index;
						colli_temp.colli_time=0;
						# pragma omp critical(TWO)
						colli_list[j++]=colli_temp;
					}
					part_two=part_two->next;
					continue;
				}
			
				colli_time = closest_dist(x1, y1, vx, vy, radius_sum);
			
				if(colli_time<dt)
				{
					colli_temp.part_one=part_one->part_index;
					colli_temp.part_two=part_two->part_index;
					colli_temp.colli_time=colli_time;
					# pragma omp critical(TWO)
					colli_list[j++]=colli_temp;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_10].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_01].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				vx=part_two->x_vel-part_one->x_vel;
				vy=part_two->y_vel-part_one->y_vel;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					if(x1*vx+y1*vy<0)
					{
						colli_temp.part_one=part_one->part_index;
						colli_temp.part_two=part_two->part_index;
						colli_temp.colli_time=0;
						# pragma omp critical(TWO)
						colli_list[j++]=colli_temp;
					}
					part_two=part_two->next;
					continue;
				}
			
				colli_time = closest_dist(x1, y1, vx, vy, radius_sum);
			
				if(colli_time<dt)
				{
					colli_temp.part_one=part_one->part_index;
					colli_temp.part_two=part_two->part_index;
					colli_temp.colli_time=colli_time;
					#pragma omp critical(TWO)
					colli_list[j++]=colli_temp;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_10].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_11].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				vx=part_two->x_vel-part_one->x_vel;
				vy=part_two->y_vel-part_one->y_vel;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					if(x1*vx+y1*vy<0)
					{
						colli_temp.part_one=part_one->part_index;
						colli_temp.part_two=part_two->part_index;
						colli_temp.colli_time=0;
						# pragma omp critical(TWO)
						colli_list[j++]=colli_temp;
					}
					part_two=part_two->next;
					continue;
				}
			
				colli_time = closest_dist(x1, y1, vx, vy, radius_sum);
			
				if(colli_time<dt)
				{
					colli_temp.part_one=part_one->part_index;
					colli_temp.part_two=part_two->part_index;
					colli_temp.colli_time=colli_time;
					#pragma omp critical(TWO)
					colli_list[j++]=colli_temp;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_01].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_11].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				vx=part_two->x_vel-part_one->x_vel;
				vy=part_two->y_vel-part_one->y_vel;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					if(x1*vx+y1*vy<0)
					{
						colli_temp.part_one=part_one->part_index;
						colli_temp.part_two=part_two->part_index;
						colli_temp.colli_time=0;
						# pragma omp critical(TWO)
						colli_list[j++]=colli_temp;
					}
					part_two=part_two->next;
					continue;
				}
			
				colli_time = closest_dist(x1, y1, vx, vy, radius_sum);
			
				if(colli_time<dt)
				{
					colli_temp.part_one=part_one->part_index;
					colli_temp.part_two=part_two->part_index;
					colli_temp.colli_time=colli_time;
					#pragma omp critical(TWO)
					colli_list[j++]=colli_temp;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
	}
	//openmp_e
	
	sort(colli_list, j);
	return j;
}

void update_for_collision(int part_one, int part_two, double time_left, struct particle *part_main, double *x_vel_f, double *y_vel_f)
{
	extern int part_num;
	extern double dt;
	
	double x_vel_inter, y_vel_inter;
		
	int i, k=0;

	part_main[part_one].x_pos=part_main[part_one].x_pos_next-time_left*part_main[part_one].x_vel;
	part_main[part_one].y_pos=part_main[part_one].y_pos_next-time_left*part_main[part_one].y_vel;
	
	part_main[part_two].x_pos=part_main[part_two].x_pos_next-time_left*part_main[part_two].x_vel;
	part_main[part_two].y_pos=part_main[part_two].y_pos_next-time_left*part_main[part_two].y_vel;
	
	double rel_pos_x=part_main[part_two].x_pos-part_main[part_one].x_pos;
	double rel_pos_y=part_main[part_two].y_pos-part_main[part_one].y_pos;
	
	swap_vel(part_main[part_one].radius, part_main[part_two].radius, part_main[part_one].rho_ratio, part_main[part_two].rho_ratio, rel_pos_x, rel_pos_y, &part_main[part_one].x_vel, &part_main[part_one].y_vel, &part_main[part_two].x_vel, &part_main[part_two].y_vel);
	
	i = part_one;
	while(k<2)
	{
		//interpolates fluid velocity at particle positions
		x_vel_inter = linear_interp(part_main[i].x_pos, part_main[i].y_pos, x_vel_f);
		y_vel_inter = linear_interp(part_main[i].x_pos, part_main[i].y_pos, y_vel_f);
		
		//RK step 1
		part_main[i].x_pos_next=part_main[i].x_pos+time_left*part_main[i].x_vel;
		part_main[i].y_pos_next=part_main[i].y_pos+time_left*part_main[i].y_vel;
		
		part_main[i].x_vel_next=part_main[i].x_vel-time_left*(part_main[i].x_vel-x_vel_inter)/part_main[i].tau;
		part_main[i].y_vel_next=part_main[i].y_vel-time_left*(part_main[i].y_vel-y_vel_inter)/part_main[i].tau;
		
		
		//periodic boundary
		if(part_main[i].x_pos_next<0)
			part_main[i].x_pos_next+=2*M_PI;
		if(part_main[i].x_pos_next>=2*M_PI)
			part_main[i].x_pos_next-=2*M_PI;
		
		if(part_main[i].y_pos_next<0)
			part_main[i].y_pos_next+=2*M_PI;
		if(part_main[i].y_pos_next>=2*M_PI)
			part_main[i].y_pos_next-=2*M_PI;
			
		i=part_two;
		k++;
	}
}

void swap_vel(double rad1, double rad2, double rho_ratio1, double rho_ratio2, double rel_x, double rel_y, double *x_vel1, double *y_vel1, double *x_vel2, double *y_vel2)
{
	if(rel_x>M_PI) rel_x-=2*M_PI;
	if(rel_x<-M_PI) rel_x+=2*M_PI;
	if(rel_y>M_PI) rel_y-=2*M_PI;
	if(rel_y<-M_PI) rel_y+=2*M_PI;
	
	double r = (rho_ratio1*rad1*rad1*rad1)/(rho_ratio2*rad2*rad2*rad2);
	double rel_vx = *x_vel2-*x_vel1;
	double rel_vy = *y_vel2-*y_vel1;
	double x_vel = *x_vel1;
	double y_vel = *y_vel1;
	
	double rel_vx_n = ((rel_vx*rel_x+rel_vy*rel_y)*rel_x)/(rel_x*rel_x+rel_y*rel_y);
	double rel_vy_n = ((rel_vx*rel_x+rel_vy*rel_y)*rel_y)/(rel_x*rel_x+rel_y*rel_y);
	
	double rel_vx_p = rel_vx-rel_vx_n;
	double rel_vy_p = rel_vy-rel_vy_n;
	
	*x_vel1 = x_vel + (2*rel_vx_n)/(1+r);
	*y_vel1 = y_vel + (2*rel_vy_n)/(1+r);
	
	*x_vel2 = x_vel + ((1-r)*rel_vx_n/(1+r)) + rel_vx_p;
	*y_vel2 = y_vel + ((1-r)*rel_vy_n/(1+r)) + rel_vy_p;
}

int find_other_collisions(struct particle *part_main, int part_num_one, int part_num_two, struct pos_grid *grid, double time_left, int colli_num, struct collision_list *colli_list)
{

	int i;
	int index_array[18];

	int x_index = part_main[part_num_one].grid_index_x;
	int y_index = part_main[part_num_one].grid_index_y;
	
	/*grid indices
		 7 6 5
		 \ | /
		8- 0 -4
		 / | \
		 1 2 3
	*/
	index_array[0] = x_index*grid_xmax + y_index;
	index_array[1] = (mod((x_index-1),grid_xmax))*grid_xmax + (mod((y_index - 1),grid_xmax));
	index_array[2] = (x_index)*grid_xmax + (mod((y_index - 1),grid_xmax));
	index_array[3] = (mod((x_index+1),grid_xmax))*grid_xmax + (mod((y_index - 1),grid_xmax));
	index_array[4] = (mod((x_index+1),grid_xmax))*grid_xmax + y_index;
	index_array[5] = (mod((x_index+1),grid_xmax))*grid_xmax + (mod((y_index +1),grid_xmax));
	index_array[6] = x_index*grid_xmax + (mod((y_index + 1),grid_xmax));
	index_array[7] = (mod((x_index-1),grid_xmax))*grid_xmax+ (mod((y_index + 1),grid_xmax));
	index_array[8] = (mod((x_index-1),grid_xmax))*grid_xmax + y_index;
	
	index_array[9]=index_array[0];
	index_array[10]=index_array[1];
	index_array[11]=index_array[2];
	index_array[12]=index_array[3];
	index_array[13]=index_array[4];
	index_array[14]=index_array[5];
	index_array[15]=index_array[6];
	index_array[16]=index_array[7];
	index_array[17]=index_array[8];
	
	
	# pragma omp parallel for	
	for(i=0;i<18;i++)
	{
		struct particle *part_one;
		struct particle *part_two;
		struct collision_list colli_temp;
		int grid_index;
		double x1, y1, vx, vy, colli_time, radius_sum;
		
		(i<9)?(part_one=&part_main[part_num_one]):(part_one=&part_main[part_num_two]);
		grid_index=index_array[i];
		
		part_two=grid[grid_index].beg;
			
		while(part_two!=NULL)
		{
			x1=(part_two->x_pos_next-time_left*part_two->x_vel)-(part_one->x_pos_next-time_left*part_one->x_vel);
			y1=(part_two->y_pos_next-time_left*part_two->y_vel)-(part_one->y_pos_next-time_left*part_one->y_vel);
			vx=part_two->x_vel-part_one->x_vel;
			vy=part_two->y_vel-part_one->y_vel;
			radius_sum=part_two->radius+part_one->radius;
		
			if((x1*x1+y1*y1)<radius_sum*radius_sum)
			{
				if(x1*vx+y1*vy<0)
				{
					colli_temp.part_one=part_one->part_index;
					colli_temp.part_two=part_two->part_index;
					colli_temp.colli_time=0;
					# pragma omp critical(ONE)
					colli_list[colli_num++]=colli_temp;
				}
				part_two=part_two->next;
				continue;
			}
		
			colli_time = closest_dist(x1, y1, vx, vy, radius_sum);
			
			if(colli_time<time_left)
			{
				colli_temp.part_one=part_one->part_index;
				colli_temp.part_two=part_two->part_index;
				colli_temp.colli_time=colli_time+(dt-time_left);
				#pragma omp critical(ONE)
				colli_list[colli_num++]=colli_temp;
			}
			part_two=part_two->next;
		}
	}
	
	sort(colli_list, colli_num);
	
	return colli_num;
}

void sort(struct collision_list *colli_list, int colli_num)
{
	int i, k=0;
	struct collision_list colli_temp;
	
	for(i=0;i<colli_num;i++)
		if (colli_list[i].colli_time < colli_list[k].colli_time)
			k = i;
	colli_temp=colli_list[0];
	colli_list[0]=colli_list[k];
	colli_list[k]=colli_temp;	
}
				
double closest_dist(double x1, double y1, double vx, double vy, double radius_sum)
{
	double a, b, c, colli_time;
	
	if(x1>M_PI) x1-=2*M_PI;
	if(x1<-M_PI) x1+=2*M_PI;
	if(y1>M_PI) y1-=2*M_PI;
	if(y1<-M_PI) y1+=2*M_PI;
	
	a = vx*vx+vy*vy;
	b = 2*(vx*x1+vy*y1);
	c = x1*x1+y1*y1-radius_sum*radius_sum;
	
	colli_time = b*b-4*a*c;
	
	if(colli_time<0) return 1;
	
	colli_time = (-b - sqrt(colli_time))/(2*a);
	
	if(colli_time<0) return 1;
	
	return colli_time;
}

void update_grid(struct particle *part_main, struct pos_grid *grid)
{
	int i;
	double delta_x=2*M_PI/grid_xmax, delta_y=2*M_PI/grid_xmax;
	
	//killing grid
	# pragma omp parallel for schedule(static) private(i)
	for(i=0;i<grid_max;i++)
	{
		grid[i].beg=NULL;
		grid[i].end=NULL;
	}
	
	//openmp_s	//err check
	//# pragma omp parallel for schedule(static) private(i)
	for(i=0;i<part_num;i++)
	{
		int x_index, y_index, index;
		
		x_index=part_main[i].x_pos/delta_x;
		y_index=part_main[i].y_pos/delta_y;
		part_main[i].grid_index_x=x_index;
		part_main[i].grid_index_y=y_index;
		
		index=x_index*grid_xmax+y_index;
		
		if(grid[index].beg==NULL)
		{
			grid[index].beg=&part_main[i];
			grid[index].end=grid[index].beg;
			grid[index].end->next=NULL;
		}
		else
		{
			grid[index].end->next=&part_main[i];
			grid[index].end=grid[index].end->next;
			grid[index].end->next=NULL;
		}
		
	}
	//openmp_e
}

double linear_interp(double x_pos, double y_pos, double *vel)
{
	extern int sys_size;
	int x_index, y_index, index_00, index_01, index_10, index_11;
	double vel_inter, vel_temp1, vel_temp2, x_index_val, y_index_val;
	
	x_index = x_pos*sys_size/(2*M_PI);
	y_index = y_pos*sys_size/(2*M_PI);
	
	x_index_val = x_index*(2*M_PI)/sys_size;
	y_index_val = y_index*(2*M_PI)/sys_size;
	
	index_00 = x_index*(sys_size+2) + y_index;
	index_01 = x_index*(sys_size+2) + (mod((y_index + 1),sys_size));
	index_10 = (mod((x_index+1),sys_size))*(sys_size+2) + y_index;
	index_11 = (mod((x_index+1),sys_size))*(sys_size+2) + (mod((y_index +1),sys_size));
	
	vel_temp1 = vel[index_00] + (vel[index_10]-vel[index_00])*(x_pos-x_index_val)/((2*M_PI)/sys_size);
	vel_temp2 = vel[index_01] + (vel[index_11]-vel[index_01])*(x_pos-x_index_val)/((2*M_PI)/sys_size);
	
	vel_inter = vel_temp1 + (vel_temp2-vel_temp1)*(y_pos-y_index_val)/((2*M_PI)/sys_size);
	
	return vel_inter;
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}
