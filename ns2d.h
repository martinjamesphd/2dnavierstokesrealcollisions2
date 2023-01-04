#ifndef ns2d_h
#define ns2d_h

extern double tolerence;
extern int sys_size, part_num, nthreads, tau_num, colli_max; 
extern const int grid_max, grid_xmax;
extern double dt, vis, mu, tau_min, tau_max, rho_ratio;

struct particle
{
	int part_index, grid_index_x, grid_index_y;
	double x_pos, x_pos_next;
	double y_pos, y_pos_next;
	double x_vel, x_vel_next;
	double y_vel, y_vel_next;
	double radius;
	double rho_ratio;
	double tau;
	int tau_index;
	struct particle *next;
};

struct pos_grid
{
	struct particle *beg;
	struct particle *end;
};

struct collision_list
{
	int part_one, part_two;
	double colli_time;
};
	
void init_den_state(int *den_state);

void init_omega(fftw_complex *omega_ft, int *den_state);

void find_vel_ft(fftw_complex *omega_ft, fftw_complex *x_vel_ft, fftw_complex *y_vel_ft);

void find_e_spectra(fftw_complex *x_vel, fftw_complex *y_vel, double *e_spectra);

void gen_force(double famp, int kf);

void find_jacobian_ft(double *omega, double *x_vel, double *y_vel);

double find_energy(double *e_spectra);

double find_epsilon(fftw_complex *omega_ft);

void solve_rk2 (fftw_complex *omega_ft, fftw_complex *omega_t, fftw_complex *x_vel_ft, fftw_complex *y_vel_ft, struct particle *part_main, double *colli_freq, struct pos_grid *grid, struct collision_list *colli_list);

void init_part(struct particle *part_main);

void update_part(double *x_vel, double *y_vel, struct particle *part_main, double *colli_freq, struct pos_grid *grid, struct collision_list *colli_list);

int detect_collisions(struct particle *part_main, struct pos_grid *grid, struct collision_list *colli_list);

void update_for_collision(int part_one, int part_two, double delta_t, struct particle *part_main, double *x_vel_f, double *y_vel_f);

int delete_in_colli_list(int part_one, int part_two, struct collision_list *colli_list, int colli_num);

void swap_vel(double rad1, double rad2, double rho_ratio1, double rho_ratio2, double rel_x, double rel_y, double *x_vel1, double *y_vel1, double *x_vel2, double *y_vel2);

int find_other_collisions(struct particle *part_main, int part_num_one, int part_num_two, struct pos_grid *grid, double delta_t, int colli_num, struct collision_list *colli_list);

void update_grid(struct particle *part_main, struct pos_grid *grid);

void sort(struct collision_list *colli_list, int colli_num);

int mod(int a, int b);

double closest_dist(double x1, double y1,double vx, double vy, double radius_sum);

double linear_interp(double x_pos, double y_pos, double *vel);

#endif
