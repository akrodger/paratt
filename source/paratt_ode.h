#ifndef PARATT_ODE_SOLVE_H
#define PARATT_ODE_SOLVE_H

/*
 * Parallel Tensor Train ODE solvers by Bram Rodgers.
 * These are all implemented using an extended version of mpi-attac.
 * Original Draft Dated: 08, July 2022
 */

/*
 * Header File Body:
 */

/*
 * Macros and Includes go here.
 */
#include<stdlib.h>
#include<stdarg.h>
#include"paratt.h"
/*
 * Struct Definitions:
 */

/*
 * Function Declarations:
 */
 
/*
* Step truncation euler forward estimate for an intitial value problem
* 
*  df
* ----  = vecField(f)
*  dt
* 
* Computes the order 1 estimate
*
* f_new = Trunc_2( f_old + dt*Trunc_1( vecField(f_old) ) )
*
* Where:
*         norm(Trunc_1(f_temp) - f_temp) <= eps_1
*         norm(Trunc_2(f_temp) - f_temp) <= eps_2
*
* For convergence, select eps_1 = A*dt, eps_2 = B*dt*dt
* for any error scaling parameters A,B.
*
* f_old_tt:   tt format tensor of the current step.
* f_new_tt:   tt format tensor of an order 1 estimate of the next step.
* work_tt:    a tt tensor which is used in computation of the vector field.
*             work_tt may be resized larger, but never smaller during runtime.
* vecField:   A function pointer where first argument is input and second
*             argument is the return value for an ODE's velocity field.
*             Third variable is the variable argument count, always assumed
*             zero at runtime. It is assumed that static variables
*             are used within the implementation of vecField. These are
*             set using the variable argument count, then not modified
*             during calls by this routine.
* dt:         Time step size for ODE solver.
* eps_1:      Low rank error for the first  truncation evaluated.
* eps_2:      Low rank error for the second truncation evaluated.
*/
void attac_exp_euler_st(attac_tt* f_old_tt,
                        attac_tt* f_new_tt,
                        attac_tt* work_tt,
                        void (*vecField)(attac_tt*,attac_tt*,int,...),
                        double dt,
                        double eps_1,
                        double eps_2);

void pattac_exp_euler_st(paratt* f_old_ptt,
                        paratt* f_new_ptt,
                        paratt* work_ptt,
                        void (*vecField)(paratt*,paratt*,int,...),
                        double dt,
                        double eps_1,
                        double eps_2);

/*
* Step truncation explicit midpoint estimate for an intitial value problem
* 
*  df
* ----  = vecField(f)
*  dt
* 
* Computes the order 2 estimate
* 
* f_stage = f_old + 0.5*dt*Trunc_1(vecField(f_old))
* f_new = Trunc_3( f_old + dt*Trunc_2( vecField(f_stage) ) )
*
* Where:
*         norm(Trunc_1(f_temp) - f_temp) <= eps_1
*         norm(Trunc_2(f_temp) - f_temp) <= eps_2
*         norm(Trunc_2(f_temp) - f_temp) <= eps_4
*
* For convergence, select eps_1 = A*dt, eps_2 = B*dt*dt, eps_3 = C*dt*dt*dt
* for any error scaling parameters A,B,C.
*
* f_old_tt:   tt format tensor of the current step.
* f_new_tt:   tt format tensor of an order 2 estimate of the next step.
* work_tt:    a tt tensor which is used in computation of the vector field.
*             work_tt may be resized larger, but never smaller during runtime.
* vecField:   A function pointer where first argument is input and second
*             argument is the return value for an ODE's velocity field.
*             Third variable is the variable argument count, always assumed
*             zero at runtime. It is assumed that static variables
*             are used within the implementation of vecField. These are
*             set using the variable argument count, then not modified
*             during calls by this routine.
* dt:         Time step size for ODE solver.
* eps_1:      Low rank error for the first  truncation evaluated.
* eps_2:      Low rank error for the second truncation evaluated.
* eps_3:      Low rank error for the third  truncation evaluated.
*/
void attac_exp_midpoint_st(attac_tt* f_old_tt,
                           attac_tt* f_new_tt,
                           attac_tt* work_tt,
                           void (*vecField)(attac_tt*,attac_tt*,int,...),
                           double dt,
                           double eps_1,
                           double eps_2,
                           double eps_3);

#endif
