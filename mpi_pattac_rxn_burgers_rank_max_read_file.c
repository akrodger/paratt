/*
 * Lax Wendroff Solver for Burgers-Hopf CDF equation.
 * Written by Bram Rodgers, 2023.
 */

/*
 * Macros and Includes go here: (Some common ones listed)
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdarg.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include "./source/paratt.h"
#include "./source/paratt_ode.h"

#define mcrd_int int
#define mcrd_flt double

#define INIT_DT       1e-5
#define IO_INTERVAL   0.001
#define MODE_DEPTH    64
#define NUM_MODES     11
#define TT_RANK_PARAM 1
#define LTE_TOL       1e-5

#define TT_RANK_MAX   5

//#define AUTOSTEP_ENABLE

/*
 * Function declarations go here:
 */



/*
 * RHS for Euler Forward method.
 * On entry, f_tt is the current time step.
 * One exit, df_tt is the value of the RHS of the PDE
 */
void cdf_eqn_rhs(paratt* f_ptt,
                       paratt* df_ptt,
                       paratt* work1_ptt,
                       paratt* work2_ptt,
                       paratt* work3_ptt,
                       paratt* work4_ptt,
                       paratt** workVec_ptt,
                       double*** source_coefs,
                       double*** int_source_coefs,
                       double dt,
                       double dx,
                       double eps,
                       double*** wave_speed,
                       double*** bdry_wave_speed,
                       double*** gc_wave_speed,
                       double*** wave_int_speed,
                       double*** bdry_int_wave_speed,
                       double*** gc_int_wave_speed,
                       double* C_mat,
                       double* D_mat,
                       double* u_space_grid,
                       double* negative_int_w,
                       int*    max_rank); 
                       
double* generate_centered_d1_mat(int num_grid_points, double dx);
double* generate_centered_d2_mat(int num_grid_points, double dx);
double* generate_centered_d4_mat(int num_grid_points, double dx);


double spaceFlux(double x, int opt){
    static double coef;
    static double offset;
    if(opt == 1){
        coef = x;
        return 0.0;
    }else if(opt == 2){
        offset = x;
        return 0.0;
    }else{
        return (coef*x + offset);
    }
}

/*
 * Given the mpi rank for this segment of the domain space, generate
 * a sampling of the domain.
 *
 */
double* generate_space_grid(int* tensor_dim_local,
                            int* tensor_dim_global,
                            int mode,
                            double domain_left,
                            double domain_right,
                            int mpi_rk,
                            int mpi_sz);

double* generate_bdry_grid(double* space_grid,
                           int     space_grid_sz,
                           double  gc_space_left,
                           double  gc_space_right);

double*** generate_wave_speed(double* space_grid,
                              int     space_grid_sz,
                              double* C_mat,
                              double* D_mat,
                              int     CD_sz);

double*** generate_source_coef(double* space_grid,
                              int     space_grid_sz,
                              double* C_mat,
                              double* D_mat,
                              int     CD_sz);

double*** generate_int_wave_speed(double* space_grid,
                              int     space_grid_sz,
                              double* C_mat,
                              double* D_mat,
                              int     CD_sz);

double*** generate_int_source_coef(double* space_grid,
                              int     space_grid_sz,
                              double* C_mat,
                              double* D_mat,
                              int     CD_sz);
void free_3d(void*** arr, int N1, int N2);
void multiIterate(int* iter,
                  int* indexBound,
                  int* iterSubset,
                  int subsetSize);


void matlabTensorPrint(char *varName,
                       char *fileName,
                       mcrd_int* dim,
                       mcrd_int  ndims,
                       mcrd_flt* x);


mcrd_flt* tensorPtr(mcrd_flt* f,
                     mcrd_int* multiIndex,
                     mcrd_int* dim,
                     mcrd_int  ndims);
/*
 * Main:
 */
int main(int argc, char** argv){
	MPI_Init(&argc, &argv);
    static double pi = acos(-1.0);
    static char name_buffer1[100];
    static char name_buffer2[100];
    char* parse_ptr = NULL;
    int ndims = NUM_MODES, mpi_rk, mpi_sz, rem;
    int k=0,i=0,j=0, k_rhs, nrmlz_period = 500, nrmlz_idx;
    long int totalEntries = 1;
    int rMax = TT_RANK_MAX;
    int num_prints = 0;
    int Nx;
    int df_idx[2] = {0,1};
    int idx_tmp;
    double dt;
    double dt_old, dt_new, err_old, err_new;
    double lte_tol = LTE_TOL;
    double dx;
    double du;
    double normalize_val;
    double trunc_eps_old = 1*pow(10.0,-1.0)*dt*dt*dt, trunc_eps_new;
    double lte_o1;
    double gc_pos[2];
    double gamma = 0.25;
    double wall_time;
    double*** wave_speed;
    double*** bdry_wave_speed;
    double*** gc_wave_speed;
    double*** source_coef;
    double*** int_wave_speed;
    double*** bdry_int_wave_speed;
    double*** gc_int_wave_speed;
    double*** int_source_coef;
    paratt* f_ptt;
    paratt* f_damp_ptt;
    paratt* df_ptt[2];
    paratt* f_euler_ptt;
    paratt* work1_ptt;
    paratt* work2_ptt;
    paratt* work3_ptt;
    paratt* work4_ptt;
    paratt** workVec_ptt;
    int* dim;
    int* ttR;
    int* mIdx;
    int* iSub;
    int* dim_local;
    double* u_grid_sample;
    double* bdry_grid_sample;
    double* D1_mat;
    double* D2_mat;
    double* D4_kse_mat;
    double* neg_int_w = NULL;
    double avg_sten[3];
    double t_start, t_end;
    double TIME_LENGTH;
    int d1_slen = 3;
    FILE* timing_log;
    char log_name[128]; 
    char write_mode[2]  = "w";
    char append_mode[2] = "a";
    avg_sten[0] = 0.5;
    avg_sten[1] = 0.0;
    avg_sten[2] = 0.5;
    MPI_Comm_rank(MPI_COMM_WORLD, &(mpi_rk));
    MPI_Comm_size(MPI_COMM_WORLD, &(mpi_sz));
    if(argc <= 6){
        fprintf(stderr,
"\nUsage: %s [tt_folder] [dim] [max_rank] [dt] [t_start] [t_end] [print_idx]\n"
                       "Reads the initial condition from [tt_folder]\n",
                       argv[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
    }
    ndims      = atoi(argv[2]);
    rMax       = atoi(argv[3]);
    dt         = strtod(argv[4], &parse_ptr);
    t_start    = strtod(argv[5], &parse_ptr);
    t_end      = strtod(argv[6], &parse_ptr);
    num_prints = atoi(argv[7]);
    TIME_LENGTH = t_end - t_start;
    dt     *= (t_end >= t_start) ? 1.0 : -1.0; 
    sprintf(&(name_buffer1[0]),argv[1],ndims);
    workVec_ptt = (paratt**) malloc(sizeof(paratt*)*ndims);
    dim = (int*) malloc(sizeof(int)*ndims);
    ttR = (int*) malloc(sizeof(int)*(ndims+1));
    mIdx = (int*) malloc(sizeof(int)*ndims);
    iSub = (int*) malloc(sizeof(int)*ndims);
    dim_local = (int*) malloc(sizeof(int)*ndims);
    sprintf(&(log_name[0]),"kse_cdf_timing_nprocs=%d.log",mpi_sz);
    f_ptt = pattac_load_tt_folder(&(name_buffer1[0]), ndims);
    Nx = f_ptt->dim[0];
    for(k=0;k<ndims;k++){
        dim[k] = f_ptt->dim[k];
        ttR[k] = f_ptt->x->ttRank[k];
        totalEntries = totalEntries*dim[k];
        mIdx[k] = 0;
        dim_local[k] = f_ptt->x->dim[k];
        iSub[k] = k;
    }
    ttR[ndims] = 1;
    u_grid_sample    = generate_space_grid(dim_local,dim,0,-1.3,1.3,mpi_rk,mpi_sz);
    du               = u_grid_sample[1]-u_grid_sample[0];
    gc_pos[0]        = u_grid_sample[0] - du;
    gc_pos[1]        = u_grid_sample[dim_local[0]-1] + du;
    bdry_grid_sample = generate_bdry_grid(u_grid_sample,dim_local[0],
                                            gc_pos[0],gc_pos[1]);
    ttR[0]=1;
    for(k=1; k < ndims; k++){
        ttR[k] = rMax;
    }
    ttR[ndims]=1;
    D1_mat   = generate_centered_d1_mat(ndims, 2.0*pi/ndims);
    D2_mat = generate_centered_d2_mat(ndims, (2.0*pi/ndims)/(1.0));
    for(k=0; k < ndims; k++){
        for(i=0; i < ndims; i++){
            D2_mat[i+ndims*k] = gamma*D2_mat[i+ndims*k];
        }
    }
    wave_speed      = generate_wave_speed(u_grid_sample,dim_local[0],
                                        D1_mat,D2_mat,ndims);
    bdry_wave_speed = generate_wave_speed(bdry_grid_sample,dim_local[0]+1,
                                        D1_mat,D2_mat,ndims);
    gc_wave_speed   = generate_wave_speed(&(gc_pos[0]),2,
                                        D1_mat,D2_mat,ndims);
    source_coef     = generate_source_coef(u_grid_sample,dim_local[0],
                                        D1_mat,D2_mat,ndims);
    int_wave_speed      = generate_int_wave_speed(u_grid_sample,dim_local[0],
                                                D1_mat,D2_mat,ndims);
    bdry_int_wave_speed = generate_int_wave_speed(bdry_grid_sample,
                                                  dim_local[0]+1,
                                                  D1_mat,D2_mat,ndims);
    gc_int_wave_speed   = generate_int_wave_speed(&(gc_pos[0]),2,
                                        D1_mat,D2_mat,ndims);
    int_source_coef     = generate_int_source_coef(u_grid_sample,dim_local[0],
                                                    D1_mat,D2_mat,ndims);
    work1_ptt = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    work2_ptt = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    work3_ptt = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    work4_ptt = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    df_ptt[0] = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    df_ptt[1] = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    f_euler_ptt = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    f_damp_ptt = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    for(k=0;k<ndims;k++){
        workVec_ptt[k] = alloc_ptt(ndims, dim, f_ptt->x->ttRank);
    }
    neg_int_w = (double*) malloc(sizeof(double)*f_ptt->x->dim[0]);

    for(k = 0; k < f_ptt->x->dim[0]; k++){
        neg_int_w[k] = du;
    }
    for(k=0;k<ndims;k++){
        pattac_marginal_trap(f_ptt, k, neg_int_w);
    }
    for(k = 0; k < f_ptt->x->dim[0]; k++){
        neg_int_w[k] = -du;
    }
    err_old = 1.0;
    num_prints = 0;
    k = 0; 
    dt_old = dt;
    double time_now = 0.0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    wall_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    
    while(time_now < TIME_LENGTH){
        if(time_now == 0.0){
            cdf_eqn_rhs(f_ptt,df_ptt[df_idx[0]],
                             work1_ptt, work2_ptt, work3_ptt, work4_ptt,
                             workVec_ptt,
                             source_coef, int_source_coef,
                             dt, du, trunc_eps_old/(dt),
                             wave_speed, bdry_wave_speed, gc_wave_speed,
                             int_wave_speed, bdry_int_wave_speed, gc_int_wave_speed,
                             D1_mat,D2_mat,
                             u_grid_sample, neg_int_w, ttR);
            err_new = err_old;
        }
        err_old = err_new;
        dt_new  = dt_old;
        do{
#ifdef AUTOSTEP_ENABLE
            dt = dt_new;
#else           
            dt = INIT_DT;
#endif
            trunc_eps_new = pow(10.0,-4.0)*pow(dt,3.0);
            trunc_eps_old = trunc_eps_new;
            attac_tt_axpby( 1.0,f_ptt->x ,dt,df_ptt[df_idx[0]]->x, f_euler_ptt->x);
            paratt_truncate(f_euler_ptt, trunc_eps_new/(dt), ttR);
            cdf_eqn_rhs(f_euler_ptt,df_ptt[df_idx[1]],
                             work1_ptt, work2_ptt, work3_ptt, work4_ptt,
                             workVec_ptt,
                             source_coef, int_source_coef,
                             dt, du, trunc_eps_new/(dt*dt),
                             wave_speed, bdry_wave_speed, gc_wave_speed,
                             int_wave_speed, bdry_int_wave_speed, gc_int_wave_speed,
                             D1_mat,D2_mat,
                             u_grid_sample, neg_int_w, ttR);
#ifdef AUTOSTEP_ENABLE
            attac_tt_axpby( 0.5, df_ptt[df_idx[0]]->x,
                                      -0.5, df_ptt[df_idx[1]]->x, work2_ptt->x);
            lte_o1 = dt*paratt_norm(work2_ptt);
            err_old = err_new;
            err_new = lte_o1;
            dt_old = dt;
            dt_new = pow(lte_tol/err_new,1.0/(3.0)*pow(err_old/lte_tol,0.08));
            if(err_new > lte_tol){
                dt_new = 0.5;
            }
            if(df_ptt[0]->mpi_rk == 0 && dt  <1e-6){
                printf("\n%le",dt);
            }
            //dt_new = (0.01 > dt_new) ? 0.01 : dt_new;
            dt_new = (2.0 < dt_new) ? 2.0 : dt_new;
            dt_new *= dt;
        }while(err_new > lte_tol);
#else
        }while(0);
#endif 
        
        if((nrmlz_idx == nrmlz_period)){
            for(i=0; i < work2_ptt->x->numDim; i++){
                mIdx[i] = work2_ptt->dim[0]-1;
            }
            nrmlz_idx = 0;
            normalize_val = paratt_entry(f_euler_ptt, mIdx);
            for(i = 0; i < f_euler_ptt->x->ttRank[1]; i++){
                for(j=0; j < f_euler_ptt->x->dim[0]; j++){
                    f_euler_ptt->x->coreList[0][j + f_euler_ptt->x->dim[0]*i]
                                     /= normalize_val;
                }
            }
            for(i = 0; i < df_ptt[df_idx[1]]->x->ttRank[1]; i++){
                for(j=0; j < df_ptt[df_idx[1]]->x->dim[0]; j++){
                    df_ptt[df_idx[1]]->x->coreList[0][j +
                                        df_ptt[df_idx[1]]->x->dim[0]*i]
                                              /= normalize_val;
                }
            }
            normalize_val = paratt_entry(f_ptt, mIdx);
            for(i = 0; i < f_ptt->x->ttRank[1]; i++){
                for(j=0; j < f_ptt->x->dim[0]; j++){
                    f_ptt->x->coreList[0][j + f_ptt->x->dim[0]*i]
                                     /= normalize_val;
                }
            }
            for(i = 0; i < df_ptt[df_idx[0]]->x->ttRank[1]; i++){
                for(j=0; j < df_ptt[df_idx[0]]->x->dim[0]; j++){
                    df_ptt[df_idx[0]]->x->coreList[0][j +
                                        df_ptt[df_idx[0]]->x->dim[0]*i]
                                              /= normalize_val;
                }
            }
        }
        nrmlz_idx++;
        idx_tmp   = df_idx[0];
        df_idx[0] = df_idx[1];
        df_idx[1] = idx_tmp;
        while(time_now <= k*IO_INTERVAL &&
                k*IO_INTERVAL < time_now + dt_old){
            /*
            fill in print out of marginal CDFs
            */
            double* marginal_cdf = 
                            (double*) malloc(sizeof(double)*f_ptt->dim[0]);
            double* marginal_cdf_01 = 
                            (double*) malloc(sizeof(double)*
                                            f_ptt->dim[0]*f_ptt->dim[1]);
            double time_frac = (k*IO_INTERVAL-time_now)/dt;
            attac_tt_axpby((1.0-time_frac) , f_ptt->x,
                            time_frac, f_euler_ptt->x, work3_ptt->x);
            paratt_truncate(work3_ptt, trunc_eps_old, ttR);
            for(i=0; i < work3_ptt->x->numDim; i++){
                mIdx[i] = work3_ptt->dim[0]-1;
            }
            normalize_val = paratt_entry(work3_ptt, mIdx);
            for(i = 0; i < work3_ptt->x->ttRank[1]; i++){
                for(j=0; j < work3_ptt->x->dim[0]; j++){
                    work3_ptt->x->coreList[0][j + work3_ptt->x->dim[0]*i]
                                     /= normalize_val;
                }
            }
            for(i = 0; i < work3_ptt->dim[0]; i++){
                mIdx[0] = i;
                marginal_cdf[i] = paratt_entry(work3_ptt, mIdx);
            }
            for(i = 0; i < work3_ptt->dim[0]; i++){
                mIdx[0] = i;
                for(j = 0; j < work3_ptt->dim[work3_ptt->x->numDim/2]; j++){
                    mIdx[work3_ptt->x->numDim/2] = j;
                    marginal_cdf_01[i+work3_ptt->dim[0]*j] =
                                        paratt_entry(work3_ptt, mIdx);
                }
            }
            if(work3_ptt->mpi_rk == 0){
                num_prints++;
                printf("\nTime = %lf\n",t_start + k*IO_INTERVAL);
                printf("dt   = %le\n",dt_old);
                printf("tt rank:\n");
                for(i=1;i<work3_ptt->x->numDim;i++){
                    mIdx[i] = 0;
                    printf("\t%d",f_ptt->x->ttRank[i]);
                }
                printf("\n");
                struct stat st = {0};
                sprintf(name_buffer1, "data",num_prints);
                if (stat(name_buffer1, &st) == -1) {
                    mkdir(name_buffer1, 0777);
                }
                sprintf(name_buffer1, "data/bg_rxn_d%d_rk%d_mrg_0",
                                        f_ptt->x->numDim,rMax);
                if (stat(name_buffer1, &st) == -1) {
                    mkdir(name_buffer1, 0777);
                }
                sprintf(name_buffer1, "data/bg_rxn_d%d_rk%d_mrg_0_mid",
                                        f_ptt->x->numDim,rMax);
                if (stat(name_buffer1, &st) == -1) {
                    mkdir(name_buffer1, 0777);
                }
                sprintf(name_buffer1, "mrg_rk%d_0{%d}",rMax,num_prints);
                sprintf(name_buffer2,
                    "data/bg_rxn_d%d_rk%d_mrg_0/marg_0_%d.m",
                                    f_ptt->x->numDim,rMax, k+1);
                matlabTensorPrint(name_buffer1,
                       name_buffer2,
                       &(work3_ptt->dim[0]),
                       1,
                       marginal_cdf);
                sprintf(name_buffer1, "mrg_0_rk%d_mid{%d}",rMax,num_prints);
                sprintf(name_buffer2,
                    "data/bg_rxn_d%d_rk%d_mrg_0_mid/mrg_0_mid_%d.m",
                            f_ptt->x->numDim,rMax, k+1);
                mIdx[0] = work3_ptt->dim[0];
                mIdx[1] = work3_ptt->dim[1];
                matlabTensorPrint(name_buffer1,
                       name_buffer2,
                       mIdx,
                       2,
                       marginal_cdf_01);
                sprintf(name_buffer1, "data/bg_rxn_d%d_rk%d_tt",
                                        f_ptt->x->numDim,rMax);
                if (stat(name_buffer1, &st) == -1) {
                    mkdir(name_buffer1, 0777);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            free(marginal_cdf);
            free(marginal_cdf_01);
	        sprintf(&(name_buffer1[0]),"data/bg_rxn_d%d_rk%d_tt/cdf_t_%1.3lf",f_ptt->x->numDim,rMax,t_start + k*IO_INTERVAL);
            pattac_write_tt_folder(&(name_buffer1[0]), work3_ptt);
            //write a checkpoint solution to disk.
            k++;
            if(k*IO_INTERVAL > TIME_LENGTH){
                break;
            }
        }
        time_now = time_now + dt_old;
        dt_old = dt_new;
        copy_rank_change_tt(f_ptt->x,f_euler_ptt->x);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    wall_time = MPI_Wtime() - wall_time;
    printf("\nsuccess proc %d\n",f_ptt->mpi_rk);
    if(mpi_rk == 0){
        timing_log = fopen(&(log_name[0]), write_mode);
        fprintf(timing_log,"\nSeconds to compute "
                           "Burger's equation to t = %le:\n\n%le",
                           (double)TIME_LENGTH,wall_time);
        fclose(timing_log);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    free(D1_mat);
    free(D2_mat);
    free_ptt(f_ptt);
    free_ptt(f_euler_ptt);
    free_ptt(df_ptt[0]);
    free_ptt(df_ptt[1]);
    free_ptt(work1_ptt);
    free_ptt(work2_ptt);
    free_ptt(work3_ptt);
    free_ptt(work4_ptt);
    for(k=0;k<ndims;k++){
        free_ptt(workVec_ptt[k]);
    }
    free(workVec_ptt);
    free_3d((void***) wave_speed,ndims,ndims);
    free_3d((void***) bdry_wave_speed,ndims,ndims);
    free_3d((void***) gc_wave_speed,ndims,ndims);
    free_3d((void***) source_coef,ndims,ndims);
    free_3d((void***) int_wave_speed,ndims,ndims);
    free_3d((void***) bdry_int_wave_speed,ndims,ndims);
    free_3d((void***) gc_int_wave_speed,ndims,ndims);
    free_3d((void***) int_source_coef,ndims,ndims);
    free(u_grid_sample);
    free(neg_int_w);
    free(ttR);
    free(mIdx);
    free(dim_local);
    free(iSub);
	MPI_Finalize();
    return 0;
}


double* generate_centered_d1_mat(int num_grid_points, double dx){
    double* D1;
    double stencil[3];
    int slen = 3, i, j, slen_m1_o2, rel_i;
    slen_m1_o2 = (slen-1)/2;
    stencil[0] = -1.0/(2.0*dx);
    stencil[1] = 0.0;
    stencil[2] = 1.0/(2.0*dx);
    D1 = (double*) malloc(sizeof(double)*num_grid_points*num_grid_points);
    for(j = 0; j < num_grid_points; j++){
        for(i = 0; i < num_grid_points; i++){
            rel_i = i+j;
            rel_i -= slen_m1_o2;
            if(rel_i < 0){
                rel_i += num_grid_points;
            }else if(rel_i >= num_grid_points){
                rel_i -= num_grid_points;
            }
            if(i < slen){
                D1[j+num_grid_points*rel_i] = stencil[i];
            }else{
                D1[j+num_grid_points*rel_i] = 0.0;
            }
        }
    }
    return D1;
}

double* generate_centered_d2_mat(int num_grid_points, double dx){
    double* D2;
    double stencil[3];
    int slen = 3, i, j, slen_m1_o2, rel_i;
    slen_m1_o2 = (slen-1)/2;
    stencil[0] = 1.0/(dx*dx);
    stencil[1] = -2.0/(dx*dx);
    stencil[2] = 1.0/(dx*dx);
    D2 = (double*) malloc(sizeof(double)*num_grid_points*num_grid_points);
    for(j = 0; j < num_grid_points; j++){
        for(i = 0; i < num_grid_points; i++){
            rel_i = i+j;
            rel_i -= slen_m1_o2;
            if(rel_i < 0){
                rel_i += num_grid_points;
            }else if(rel_i >= num_grid_points){
                rel_i -= num_grid_points;
            }
            if(i < slen){
                D2[rel_i+num_grid_points*j] = stencil[i];
            }else{
                D2[rel_i+num_grid_points*j] = 0.0;
            }
        }
    }
    return D2;
}

double* generate_centered_d4_mat(int num_grid_points, double dx){
    double* D4;
    double dx4 = (dx*dx)*(dx*dx);
    double stencil[5];
    int slen = 5, i, j, slen_m1_o2, rel_i;
    slen_m1_o2 = (slen-1)/2;
    stencil[0] =  1.0/(dx4);
    stencil[1] = -4.0/(dx4);
    stencil[2] =  6.0/(dx4);
    stencil[3] = -4.0/(dx4);
    stencil[4] =  1.0/(dx4);
    D4 = (double*) malloc(sizeof(double)*num_grid_points*num_grid_points);
    for(j = 0; j < num_grid_points; j++){
        for(i = 0; i < num_grid_points; i++){
            rel_i = i+j;
            rel_i -= slen_m1_o2;
            if(rel_i < 0){
                rel_i += num_grid_points;
            }else if(rel_i >= num_grid_points){
                rel_i -= num_grid_points;
            }
            if(i < slen){
                D4[rel_i+num_grid_points*j] = stencil[i];
            }else{
                D4[rel_i+num_grid_points*j] = 0.0;
            }
        }
    }
    return D4;
}

/*
 * Given the mpi rank for this segment of the domain space, generate
 * a sampling of the domain.
 *
 */
double* generate_space_grid(int* tensor_dim_local,
                            int* tensor_dim_global,
                            int mode,
                            double domain_left,
                            double domain_right,
                            int mpi_rk,
                            int mpi_sz){
    int rem, k, k_rhs;
    double* u_grid_sample = (double*) malloc(sizeof(double)*
                                            tensor_dim_local[mode]);
    for(k=0;k<tensor_dim_local[mode];k++){
        rem  = tensor_dim_global[mode] % mpi_sz;
        if(mpi_rk < rem || rem == 0){
            k_rhs = k + ((tensor_dim_local[mode])*mpi_rk);
        }else{
            k_rhs = ((tensor_dim_local[mode]+1)*rem + 
                    tensor_dim_local[mode]*(mpi_rk-rem))+k;
        }
        u_grid_sample[k] = (((double)k_rhs)/(tensor_dim_global[mode]-1));
        u_grid_sample[k] = ((1.0 - u_grid_sample[k])*domain_left) +
                                  (u_grid_sample[k]*domain_right);
    }
    return u_grid_sample;
}


/*
 * Given 1d cell centers and 2 ghost cells, generate a vector of cell boundary
 * locations.
 */
double* generate_bdry_grid(double* space_grid,
                           int     space_grid_sz,
                           double  gc_space_left,
                           double  gc_space_right){
    int k;
    double* bdry_grid_sample = (double*) malloc(sizeof(double)*
                                                    (space_grid_sz+1));
    for(k=0;k<space_grid_sz;k++){

        if(k == 0){
            bdry_grid_sample[k]   = (space_grid[k] + gc_space_left)*0.5;
        }else if(k == space_grid_sz-1){
            bdry_grid_sample[k]   = (space_grid[k] + space_grid[k-1])*0.5;
            bdry_grid_sample[k+1] = (space_grid[k] + gc_space_right)*0.5;
        }else if(k >= 1){
            bdry_grid_sample[k]   = (space_grid[k] + space_grid[k-1])*0.5;
        }
    }
    return bdry_grid_sample;
}


/*
* Generate a 2d array of wave speed vectors. This array is accessed as
* 
* W[i][j][:] = (i,j) term in CDF Eqn sum sampled along all points in space_grid
* Every term when i!=j takes the form
*   -(C(i,j) + D(i,j)*space_grid[k])
* If i==j, we instead have
*   -(C(i,j)*space_grid[k] + D(i,j))*space_grid[k]
*
* If C(i,j) and D(i,j) are numerical zero, the sum's term is skiped
* and  W[i][j] is set to the null pointer.
*
* The returned 3d array is CD_sz*CD_sz*space_grid_sz
*/
double*** generate_wave_speed(double* space_grid,
                              int     space_grid_sz,
                              double* C_mat,
                              double* D_mat,
                              int     CD_sz){
    double*** W;
    int i, j, k;
    W = (double***) malloc(sizeof(double**)*CD_sz);
    for(i = 0; i < CD_sz; i++){
        W[i] = (double**) malloc(sizeof(double*)*CD_sz);
        for(j = 0; j < CD_sz; j++){
            if(fabs(C_mat[i+CD_sz*j]) > 1e-14 || 
               fabs(D_mat[i+CD_sz*j]) > 1e-14 ||
               i == j){
                W[i][j] = (double*) malloc(sizeof(double)*space_grid_sz);
                for(k = 0; k < space_grid_sz; k++){
                    W[i][j][k] = 0.0;
                }
                for(k = 0; k < space_grid_sz; k++){
                    W[i][j][k] += -(C_mat[i+CD_sz*j]*space_grid[k]);
                }
                for(k = 0; k < space_grid_sz; k++){
                    W[i][j][k] += D_mat[i+CD_sz*j];
                }
                if(i == j){
                    for(k = 0; k < space_grid_sz; k++){
                        W[i][j][k] *= space_grid[k];
                        W[i][j][k] += sin(acos(-1.0)*space_grid[k])*10.0;
                    }
                }
            }else{
                W[i][j] = NULL;
            }
        }
    }
    return W;
}


double*** generate_int_wave_speed(double* space_grid,
                                     int     space_grid_sz,
                                     double* C_mat,
                                     double* D_mat,
                                     int     CD_sz){
    double*** W;
    int i, j, k;
    W = (double***) malloc(sizeof(double**)*CD_sz);
    for(i = 0; i < CD_sz; i++){
        W[i] = (double**) malloc(sizeof(double*)*CD_sz);
        for(j = 0; j < CD_sz; j++){
            if(fabs(C_mat[i+CD_sz*j]) > 1e-14 || 
               fabs(D_mat[i+CD_sz*j]) > 1e-14){
                W[i][j] = (double*) malloc(sizeof(double)*space_grid_sz);
                for(k = 0; k < space_grid_sz; k++){
                    W[i][j][k] = 0.0;
                }
                if(fabs(C_mat[i+CD_sz*j]) > 1e-14){
                    for(k = 0; k < space_grid_sz; k++){
                        W[i][j][k] += -(C_mat[i+CD_sz*j]*space_grid[k]);
                    }
                }
                if(fabs(D_mat[i+CD_sz*j]) > 1e-14){
                    for(k = 0; k < space_grid_sz; k++){
                        W[i][j][k] += D_mat[i+CD_sz*j];
                    }
                }
                if(i == j){
                    for(k = 0; k < space_grid_sz; k++){
                        W[i][j][k] *= space_grid[k];
                    }
                }
            }else{
                W[i][j] = NULL;
            }
        }
    }
    return W;
}

double*** generate_source_coef(double* space_grid,
                              int     space_grid_sz,
                              double* C_mat,
                              double* D_mat,
                              int     CD_sz){
    double*** W;
    int i, j, k;
    W = (double***) malloc(sizeof(double**)*CD_sz);
    for(i = 0; i < CD_sz; i++){
        W[i] = (double**) malloc(sizeof(double*)*CD_sz);
        for(j = 0; j < CD_sz; j++){
            W[i][j] = (double*) malloc(sizeof(double)*space_grid_sz);
            for(k = 0; k < space_grid_sz; k++){
                W[i][j][k] = 0.0;
            }
            if(i == j){
                if(fabs(C_mat[i+CD_sz*j]) > 1e-14){
                    for(k = 0; k < space_grid_sz; k++){
                        W[i][j][k] += -2.0*(C_mat[i+CD_sz*j]*space_grid[k]);
                    }
                }
                if(fabs(D_mat[i+CD_sz*j]) > 1e-14){
                    for(k = 0; k < space_grid_sz; k++){
                        W[i][j][k] +=  D_mat[i+CD_sz*j];
                        W[i][j][k] += acos(-1.0)*cos(acos(-1.0)*space_grid[k])*10.0;
                    }
                }
            }else{
                if(fabs(C_mat[i+CD_sz*j]) > 1e-14){
                    for(k = 0; k < space_grid_sz; k++){
                        W[i][j][k] += -C_mat[i+CD_sz*j];
                    }
                }
            }
        }
    }
    return W;
}

double*** generate_int_source_coef(double* space_grid,
                              int     space_grid_sz,
                              double* C_mat,
                              double* D_mat,
                              int     CD_sz){
    double*** W;
    int i, j, k;
    W = (double***) malloc(sizeof(double**)*CD_sz);
    for(i = 0; i < CD_sz; i++){
        W[i] = (double**) malloc(sizeof(double*)*CD_sz);
        for(j = 0; j < CD_sz; j++){
            if(fabs(C_mat[i+CD_sz*j]) > 1e-14 || 
               fabs(D_mat[i+CD_sz*j]) > 1e-14){
                W[i][j] = (double*) malloc(sizeof(double)*space_grid_sz);
                for(k = 0; k < space_grid_sz; k++){
                    W[i][j][k] = 0.0;
                }
                if(i == j){
                    if(fabs(C_mat[i+CD_sz*j]) > 1e-14){
                        for(k = 0; k < space_grid_sz; k++){
                            W[i][j][k] += -2.0*(C_mat[i+CD_sz*j]*space_grid[k]);
                        }
                    }
                    if(fabs(D_mat[i+CD_sz*j]) > 1e-14){
                        for(k = 0; k < space_grid_sz; k++){
                            W[i][j][k] +=  D_mat[i+CD_sz*j];
                        }
                    }
                }else{
                    if(fabs(C_mat[i+CD_sz*j]) > 1e-14){
                        for(k = 0; k < space_grid_sz; k++){
                            W[i][j][k] += -C_mat[i+CD_sz*j];
                        }
                    }
                }
            }else{
                W[i][j] = NULL;
            }
        }
    }
    return W;
}



void cdf_rhs_no_int_term(paratt* f_ptt,
                         paratt* df_ptt,
                         double     dt,
                         double     dx,
                         double***  wave_speed,
                         double***  bdry_wave_speed,
                         double***  gc_wave_speed,
                         double* u_space_grid,
                         double* source_coef,
                         int i,
                         int j){
    double sten[3];
    sten[0] = 0.5/dx;
    sten[1] = 0.0;
    sten[2] = -0.5/dx;
    copy_rank_change_tt(df_ptt->x,f_ptt->x);
    pattac_apply_cdf_lw_rhs(df_ptt, i, wave_speed[i][j], gc_wave_speed[i][j],
                                    bdry_wave_speed[i][j],
                                   source_coef, dt, dx);
    if(i != j){
        attac_mult_diag_mat(df_ptt->x, j, u_space_grid);
    }
}

void cdf_rhs_int_term(paratt* f_ptt,
                         paratt* df_ptt,
                         double     dt,
                         double     dx,
                         double***  wave_speed,
                         double***  bdry_wave_speed,
                         double***  gc_wave_speed,
                         double* u_space_grid,
                         double* negative_int_w,
                         double* source_coef,
                         int i,
                         int j){
    double sten[3];
    sten[0] = 0.5/dx;
    sten[1] = 0.0;
    sten[2] = -0.5/dx;
    copy_rank_change_tt(df_ptt->x,f_ptt->x);
    pattac_apply_cdf_lw_rhs(df_ptt, i, wave_speed[i][j], gc_wave_speed[i][j],
                                    bdry_wave_speed[i][j],
                                    source_coef, dt, dx);
    pattac_marginal_trap(df_ptt, j, negative_int_w);
}



void cdf_eqn_rhs(paratt* f_ptt,
                       paratt* df_ptt,
                       paratt* work1_ptt,
                       paratt* work2_ptt,
                       paratt* work3_ptt,
                       paratt* work4_ptt,
                       paratt** workVec_ptt,
                       double*** source_coefs,
                       double*** int_source_coefs,
                       double dt,
                       double dx,
                       double eps,
                       double*** wave_speed,
                       double*** bdry_wave_speed,
                       double*** gc_wave_speed,
                       double*** int_wave_speed,
                       double*** bdry_int_wave_speed,
                       double*** gc_int_wave_speed,
                       double* C_mat,
                       double* D_mat,
                       double* u_space_grid,
                       double* negative_int_w,
                       int*    max_rank){
    int i = 0, j = 0,CD_sz;
    int first = 1;
    CD_sz = f_ptt->x->numDim;
    for(i = 0; i < f_ptt->x->numDim; i++){
        first = 1;
        for(j = 0; j < f_ptt->x->numDim; j++){
            if(int_wave_speed[i][j] != NULL && i!=j){
                cdf_rhs_no_int_term(f_ptt,work4_ptt,
                                    dt, dx,
                                    wave_speed,
                                    bdry_wave_speed,
                                    gc_wave_speed,
                                    u_space_grid,
                                    source_coefs[i][j],
                                    i, j);
                cdf_rhs_int_term(f_ptt,work2_ptt,
                                    dt, dx,
                                    int_wave_speed,
                                    bdry_int_wave_speed,
                                    gc_int_wave_speed,
                                    u_space_grid,
                                    negative_int_w,
                                    int_source_coefs[i][j],
                                    i, j);
                attac_tt_axpby( 1.0, work4_ptt->x,
                            1.0, work2_ptt->x, work3_ptt->x);
            }else if(wave_speed[i][j] != NULL){
                cdf_rhs_no_int_term(f_ptt,work3_ptt,
                                    dt, dx,
                                    wave_speed,
                                    bdry_wave_speed,
                                    gc_wave_speed,
                                    u_space_grid,
                                    source_coefs[i][j],
                                    i, j);
                paratt_truncate(work3_ptt, 0.0, f_ptt->x->ttRank);
            }
            if(int_wave_speed[i][j] != NULL ||
                   wave_speed[i][j] != NULL){
                if(first == 1){
                    first = 0;
                    copy_rank_change_tt(work1_ptt->x, work3_ptt->x);
                }else{
                    attac_tt_axpby(1.0, work1_ptt->x,
                                   1.0, work3_ptt->x, work2_ptt->x);
                    copy_rank_change_tt(work1_ptt->x, work2_ptt->x);
                }
            }
        }
        paratt_truncate(work1_ptt, eps,max_rank);
        copy_rank_change_tt(workVec_ptt[i]->x, work1_ptt->x);
        first = 1;
    }
    attac_tt_axpby(1.0, f_ptt->x, dt, workVec_ptt[0]->x, work1_ptt->x);
    attac_tt_axpby(1.0, workVec_ptt[0]->x, 1.0, workVec_ptt[1]->x, work1_ptt->x);
    paratt_truncate(work1_ptt, eps/(f_ptt->x->numDim),
                    work1_ptt->x->ttRank);
    for(i = 2; i < f_ptt->x->numDim; i++){
        if(i % 2 == 0){
            attac_tt_axpby(1.0, work1_ptt->x, 1.0, workVec_ptt[i]->x,
                            work2_ptt->x);
            paratt_truncate(work2_ptt, eps/(f_ptt->x->numDim),max_rank);
        }else{
            attac_tt_axpby(1.0, work2_ptt->x, 1.0, workVec_ptt[i]->x,
                            work1_ptt->x);
            paratt_truncate(work1_ptt, eps/(f_ptt->x->numDim),max_rank);
        }
    }
    if(i % 2 == 1){
        copy_rank_change_tt(df_ptt->x, work2_ptt->x);
    }else{
        copy_rank_change_tt(df_ptt->x, work1_ptt->x);
    }
}


void free_3d(void*** arr, int N1, int N2){
    int i,j;
    for(i=0;i<N1;i++){
        for(j=0;j<N2;j++){
            if(arr[i][j] != NULL){
                free(arr[i][j]);
            }
        }
        free(arr[i]);
    }
    free(arr);
}


void multiIterate(int* multiIndex,
                  int* indexBound,
                  int* iterSubset,
                  int subsetSize){
    int i;
    int multiIndex_idx = 0;
    int carryFlag = 1;
    while(multiIndex_idx < subsetSize && carryFlag == 1){
        carryFlag = 0;
        i=multiIndex_idx;
        if(iterSubset != NULL){
            multiIndex[iterSubset[i]] += 1;
            //this case handles carrying over.
            if(multiIndex[iterSubset[i]] >= indexBound[iterSubset[i]]){
                multiIndex[iterSubset[i]] = 0;
                multiIndex_idx++;
                carryFlag = 1;
            }
        }else{
            multiIndex[i] += 1;
            if(multiIndex[i] >= indexBound[i]){
			    multiIndex[i] = 0;
			    multiIndex_idx++;
			    carryFlag = 1;
            }
        }
    }
}


mcrd_flt* tensorPtr(mcrd_flt* f,
                     mcrd_int* multiIndex,
                     mcrd_int* dim,
                     mcrd_int  ndims){
	mcrd_int k, l, memOffset, product, d;
	//Iterate through the multi-index in column major order
	memOffset = multiIndex[0];
	k = 1;
	d = ndims;
	product = 1;
	while(k < d){
		l = k-1;
		product *= dim[l];
		memOffset += multiIndex[k] * product;
		k++;
	}
	return &(f[memOffset]);
}


void matlabTensorPrint(char *varName,
                       char *fileName,
                       mcrd_int* dim,
                       mcrd_int  ndims,
                       mcrd_flt* x){
    FILE *f = fopen(fileName, "w");
    mcrd_int i, j, k, numSlices, dim0, dim1;
    mcrd_int* mIdx;
    mcrd_int* iSub;
    dim0 = dim[0];
    numSlices = 1;
    mIdx = (mcrd_int*) malloc(sizeof(mcrd_int)*ndims);
    iSub = (mcrd_int*) malloc(sizeof(mcrd_int)*ndims);
    if(ndims > 1){
        dim1 = dim[1];
        for(k=2;k<ndims;k++){
            numSlices *= dim[k];
            iSub[k-2] = k;
        }
    }else{
        dim1 = 1;
    }
    for(k=0;k<ndims;k++){
        mIdx[k] = 0;
    }
    if (f == NULL){
        fprintf(stderr,"Error opening file!\n");
        exit(1);
    }
    fprintf(f,"%s = zeros([", varName);
    for(k=0;k<ndims-1;k++){
        fprintf(f,"%ld,",(long)dim[k]);
    }
    if(ndims > 1){
        fprintf(f,"%ld]);\n",(long)dim[ndims-1]);
    }else{
        fprintf(f,"%ld,1]);\n",(long)dim[ndims-1]);
    }
    //start printing slices
    for(k=0;k<numSlices;k++){
        if(ndims > 1){
            fprintf(f,"%s(:,:",varName);
        }else{
            fprintf(f,"%s(:",varName);
        }
        for(i=2;i<ndims;i++){
            fprintf(f,",%ld",(long)mIdx[i]+1);
        }
        fprintf(f,")=[...\n");
        for(i = 0; i < dim0; i++){
            mIdx[0] = i;
            for(j = 0; j < dim1; j++){
                if(ndims > 1){
                    mIdx[1] = j;
                }
                if(j > 0){
                    fprintf(f,";%+1.16le", *(tensorPtr(x,mIdx,dim,ndims)));
                }else{
                    fprintf(f,"[%+1.16le", *(tensorPtr(x,mIdx,dim,ndims)));
                }
            }
            if(i == dim0 - 1){
                fprintf(f,"]];\n");
            }else{
                fprintf(f,"],...\n");
            }
        }
        multiIterate(mIdx,dim,iSub,ndims-2);
    }
    fclose(f);
    free(mIdx);
    free(iSub);
}
