/*
 * C Source file Template by Bram Rodgers.
 * Original Draft Dated: 25, Feb 2018
 */

/*
 * Macros and Includes go here: (Some common ones included)
 */
#include "paratt_ode.h"
/*
 * Locally used helper functions:
 */
typedef void (*vecField_ptr)(paratt*,paratt*,int,...);

typedef struct{
    vecField_ptr vF;
    paratt* work_ptt;
    double dt;
} iEuler_t;
/*
 * Static Local Variables:
 */
//A function for evaluating the implicit euler method's linear operator.
void ieuler_eval(paratt* arg_ptt,
                 paratt* res_ptt, void* iEuler_data){
    iEuler_t *dat = (iEuler_t*) iEuler_data;
    dat->vF(arg_ptt, dat->work_ptt, 0);
    attac_tt_axpby(1.0, arg_ptt->x, -dat->dt, dat->work_ptt->x, res_ptt->x);
}
/*
 * Function Implementations:
 */

void attac_exp_euler_st(attac_tt* f_old_tt,
                        attac_tt* f_new_tt,
                        attac_tt* work_tt,
                        void (*vecField)(attac_tt*,attac_tt*,int,...),
                        double dt,
                        double eps_1,
                        double eps_2){
    vecField(f_old_tt, work_tt,0);
    attac_tt_truncate(work_tt, eps_1, work_tt->ttRank);
    attac_tt_axpby(1.0, f_old_tt, dt, work_tt, f_new_tt);
    attac_tt_truncate(f_new_tt, eps_2, f_new_tt->ttRank);
}

void pattac_exp_euler_st(paratt* f_old_ptt,
                        paratt* f_new_ptt,
                        paratt* work_ptt,
                        void (*vecField)(paratt*,paratt*,int,...),
                        double dt,
                        double eps_1,
                        double eps_2){
    vecField(f_old_ptt, work_ptt,0);
    paratt_truncate(work_ptt, eps_1, work_ptt->x->ttRank);
    attac_tt_axpby(1.0, f_old_ptt->x, dt, work_ptt->x, f_new_ptt->x);
    paratt_truncate(f_new_ptt, eps_2, f_new_ptt->x->ttRank);
}
