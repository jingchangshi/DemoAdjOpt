import jax.numpy as jnp
import jax.scipy as jsp
import jax
from tqdm import tqdm
import time

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

bnd_x_l = 0.0
bnd_x_h = 1.0
bnd_y_l = 0.0
bnd_y_h = 1.0
npts_int = 10
bnd_T_ = 200.0
init_K_ = 50.0
beta1_ref = 13221.0
beta2_ref = 27.0
fwd_iter_max = 100
adj_iter_max = 10
# FP32 cannot reach such tol. FP32 reaches only the order of magnitude of 1.
fwd_abs_tol = 1E-6
fwd_rel_tol = 1E-6
adj_abs_tol = 1E-4
adj_rel_tol = 1E-4
omega_fwd_T = 0.8
omega_adj_K = 0.01

npts_ = npts_int + 1

@jax.jit
def compute_Q(x, y):
    gamma = 0.5E6
    # Negative RHS leads to the solution distribution being convex upwards
    q = -gamma*(1-(x-0.5)*(x-0.5))*(1-(y-0.5)*(y-0.5))
    return q

@jax.jit
def compute_K(T, beta1, beta2):
    K = beta1 / T + beta2
    return K
#  compute_dKdT = jax.grad(compute_K, argnums=0)

@jax.jit
def compute_R(T, Tl, Tr, Tb, Tt, K, Kl, Kr, Kb, Kt, q, dx):
    dTdx = (Tr - Tl)/(2.0*dx)
    dTdy = (Tt - Tb)/(2.0*dx)
    dKdx = (Kr - Kl)/(2.0*dx)
    dKdy = (Kt - Kb)/(2.0*dx)
    T_grad2 = (Tr - 2.0*T + Tl)/(dx*dx) \
            + (Tt - 2.0*T + Tb)/(dx*dx)
    R = K * T_grad2
    R += dKdx * dTdx + dKdy * dTdy
    R -= q
    return R
compute_dRdT = jax.grad(compute_R, argnums=[0,1,2,3,4])
compute_dRdK = jax.grad(compute_R, argnums=[5,6,7,8,9])

@jax.jit
def assembleSystemNoGrad(grid_T, grid_K, grid_q, dx):
    npts22 = npts_ * npts_
    R_vec = jnp.empty((npts22,))
    for i in range(npts_):
        for j in range(npts_):
            T = grid_T[i, j]
            K = grid_K[i, j]
            # 
            Tl = grid_T[i, j-1]
            Tr = grid_T[i, j+1]
            Tb = grid_T[i-1, j]
            Tt = grid_T[i+1, j]
            Kl = grid_K[i, j-1]
            Kr = grid_K[i, j+1]
            Kb = grid_K[i-1, j]
            Kt = grid_K[i+1, j]
            # 
            # Left
            if (j==0):
               Tl = bnd_T_
               Kl = 2.0*grid_K[i, j] - grid_K[i, j+1]
            # Right
            if (j==npts_-1):
               Tr = bnd_T_
               Kr = 2.0*grid_K[i, j] - grid_K[i, j-1]
            # Bottom
            if (i==0):
               Tb = bnd_T_
               Kb = 2.0*grid_K[i, j] - grid_K[i+1, j]
            # Top
            if (i==npts_-1):
               Tt = bnd_T_
               Kt = 2.0*grid_K[i, j] - grid_K[i-1, j]
            # Compute R
            ij = i*npts_+j
            R_vec = R_vec.at[ij].set(compute_R(T, Tl, Tr, Tb, Tt, K, Kl, Kr, Kb, Kt, grid_q[i, j], dx))
    return R_vec

@jax.jit
def assembleSystemGradT(grid_T, grid_K, grid_q, dx):
    npts22 = npts_ * npts_
    R_vec = jnp.empty((npts22,))
    dRdT_dns = jnp.zeros((npts22, npts22))
    for i in range(npts_):
        for j in range(npts_):
            T = grid_T[i, j]
            K = grid_K[i, j]
            # 
            Tl = grid_T[i, j-1]
            Tr = grid_T[i, j+1]
            Tb = grid_T[i-1, j]
            Tt = grid_T[i+1, j]
            Kl = grid_K[i, j-1]
            Kr = grid_K[i, j+1]
            Kb = grid_K[i-1, j]
            Kt = grid_K[i+1, j]
            # 
            # Left
            if (j==0):
               Tl = bnd_T_
               Kl = 2.0*grid_K[i, j] - grid_K[i, j+1]
            # Right
            if (j==npts_-1):
               Tr = bnd_T_
               Kr = 2.0*grid_K[i, j] - grid_K[i, j-1]
            # Bottom
            if (i==0):
               Tb = bnd_T_
               Kb = 2.0*grid_K[i, j] - grid_K[i+1, j]
            # Top
            if (i==npts_-1):
               Tt = bnd_T_
               Kt = 2.0*grid_K[i, j] - grid_K[i-1, j]
            # Compute R
            ij = i*npts_+j
            R_vec = R_vec.at[ij].set(compute_R(T, Tl, Tr, Tb, Tt, K, Kl, Kr, Kb, Kt, grid_q[i, j], dx))
            dRdT_row = compute_dRdT(T, Tl, Tr, Tb, Tt, K, Kl, Kr, Kb, Kt, grid_q[i, j], dx)
            dRdT_dns = dRdT_dns.at[ij, ij].set(dRdT_row[0])
            if (j>0):
                dRdT_dns = dRdT_dns.at[ij, ij-1].set(dRdT_row[1])
            if (j<npts_-1):
                dRdT_dns = dRdT_dns.at[ij, ij+1].set(dRdT_row[2])
            if (i>0):
                dRdT_dns = dRdT_dns.at[ij, ij-npts_].set(dRdT_row[3])
            if (i<npts_-1):
                dRdT_dns = dRdT_dns.at[ij, ij+npts_].set(dRdT_row[4])
    return dRdT_dns, R_vec

@jax.jit
def assembleSystemGradK(grid_T, grid_K, grid_q, dx):
    npts22 = npts_ * npts_
    R_vec = jnp.empty((npts22,))
    dRdK_dns = jnp.zeros((npts22, npts22))
    for i in range(npts_):
        for j in range(npts_):
            T = grid_T[i, j]
            K = grid_K[i, j]
            # 
            Tl = grid_T[i, j-1]
            Tr = grid_T[i, j+1]
            Tb = grid_T[i-1, j]
            Tt = grid_T[i+1, j]
            Kl = grid_K[i, j-1]
            Kr = grid_K[i, j+1]
            Kb = grid_K[i-1, j]
            Kt = grid_K[i+1, j]
            # 
            # Left
            if (j==0):
               Tl = bnd_T_
               Kl = 2.0*grid_K[i, j] - grid_K[i, j+1]
            # Right
            if (j==npts_-1):
               Tr = bnd_T_
               Kr = 2.0*grid_K[i, j] - grid_K[i, j-1]
            # Bottom
            if (i==0):
               Tb = bnd_T_
               Kb = 2.0*grid_K[i, j] - grid_K[i+1, j]
            # Top
            if (i==npts_-1):
               Tt = bnd_T_
               Kt = 2.0*grid_K[i, j] - grid_K[i-1, j]
            # Compute R
            ij = i*npts_+j
            R_vec = R_vec.at[ij].set(compute_R(T, Tl, Tr, Tb, Tt, K, Kl, Kr, Kb, Kt, grid_q[i, j], dx))
            dRdK_row = compute_dRdK(T, Tl, Tr, Tb, Tt, K, Kl, Kr, Kb, Kt, grid_q[i, j], dx)
            dRdK_dns = dRdK_dns.at[ij, ij].set(dRdK_row[0])
            if (j>0):
                dRdK_dns = dRdK_dns.at[ij, ij-1].set(dRdK_row[1])
            if (j<npts_-1):
                dRdK_dns = dRdK_dns.at[ij, ij+1].set(dRdK_row[2])
            if (i>0):
                dRdK_dns = dRdK_dns.at[ij, ij-npts_].set(dRdK_row[3])
            if (i<npts_-1):
                dRdK_dns = dRdK_dns.at[ij, ij+npts_].set(dRdK_row[4])
    return dRdK_dns, R_vec



grid_1d = jnp.linspace(bnd_x_l, bnd_x_h, npts_)
dx = grid_1d[1] - grid_1d[0]
dy = grid_1d[1] - grid_1d[0]
grid_x, grid_y = jnp.meshgrid(grid_1d, grid_1d)
grid_q = compute_Q(grid_x, grid_y)
# Init solution
grid_T = jnp.ones((npts_, npts_)) * bnd_T_
grid_K = jnp.ones((npts_, npts_)) * init_K_
case_name = "T_ref-N%d"%(npts_int)
dat_fname = "%s.npy"%(case_name)
grid_T_ref = jnp.load(dat_fname)
dJdT_vec = jnp.empty((npts_*npts_,))

adj_abs_err_0 = 0.0
for adj_it in range(adj_iter_max):
    #  print("==> No. %d adjoint iteration"%(adj_it))
    t_beg = time.perf_counter()
    fwd_it = 0
    fwd_abs_res = 0.0
    R_vec = assembleSystemNoGrad(grid_T, grid_K, grid_q, dx)
    fwd_abs_res_0 = jnp.sqrt(jnp.sum(R_vec*R_vec))
    for fwd_it in range(fwd_iter_max):
        # Assemble the linear system
        dRdT_dns, R_vec = assembleSystemGradT(grid_T, grid_K, grid_q, dx)
        fwd_abs_res = jnp.sqrt(jnp.sum(R_vec*R_vec))
        fwd_rel_res = fwd_abs_res/fwd_abs_res_0
        if (jnp.isnan(fwd_abs_res)):
            print("   FWD==> No. %d iteration encounters NaN"%(fwd_it))
            break
        #  else:
        #      print("No. %d iteration, fwd_abs_res = %.4e, fwd_rel_res = %.2e"%(fwd_it, fwd_abs_res, fwd_rel_res))
        if (fwd_abs_res<fwd_abs_tol or fwd_rel_res<fwd_rel_tol):
            print("   FWD==> Converged with fwd_abs_res = %.4e fwd_rel_res = %.2e after %d iterations"%(fwd_abs_res, fwd_rel_res, fwd_it))
            break
        # Solve the linear system
        b_vec = -R_vec
        #  dT = jsp.linalg.solve(dRdT_dns, b_vec, assume_a='pos')
        dT = jsp.linalg.solve(dRdT_dns, b_vec)
        # Update the solution
        for i in range(npts_):
            for j in range(npts_):
                grid_T = grid_T.at[i, j].set( grid_T[i, j] + omega_fwd_T*dT[i*npts_+j] )
    if (fwd_it == fwd_iter_max-1):
        fwd_it += 1
        R_vec = assembleSystemNoGrad(grid_T, grid_K, grid_q, dx)
        fwd_abs_res = jnp.sqrt(jnp.sum(R_vec*R_vec))
        fwd_rel_res = fwd_abs_res/fwd_abs_res_0
        if (jnp.isnan(fwd_abs_res)):
            print("   FWD==> No. %d iteration encounters NaN"%(fwd_it))
        #  else:
        #      print("No. %d iteration, fwd_abs_res = %.4e, fwd_rel_res = %.2e"%(fwd_it, fwd_abs_res, fwd_rel_res))
        if (fwd_abs_res<fwd_abs_tol or fwd_rel_res<fwd_rel_tol):
            print("   FWD==> Converged with fwd_abs_res = %.4e fwd_rel_res = %.2e after %d iterations"%(fwd_abs_res, fwd_rel_res, fwd_it))
        else:
            print("   FWD==> Not converged with fwd_abs_res = %.4e fwd_rel_res = %.2e after %d iterations"%(fwd_abs_res, fwd_rel_res, fwd_it))
    t_end = time.perf_counter()
    #  print("Time cost is %.2f seconds"%(t_end-t_beg))
    #
    # Compute dJdT
    for i in range(npts_):
        for j in range(npts_):
            ij = i*npts_+j
            dJdT_vec = dJdT_vec.at[ij].set(2.0*(grid_T[i,j]-grid_T_ref[i,j]))
    # Solve the adjoint system
    dJdR_T_vec = jsp.linalg.solve(dRdT_dns.T, dJdT_vec)
    # Compute dRdK based on the latest accurate T
    dRdK_dns, R_vec = assembleSystemGradK(grid_T, grid_K, grid_q, dx)
    # Compute the sensitivity
    dJdK_vec = jnp.matmul(-dRdK_dns.T, dJdR_T_vec)
    # Update the design variables
    for i in range(npts_):
        for j in range(npts_):
            grid_K = grid_K.at[i, j].set( grid_K[i, j] - omega_adj_K*dJdK_vec[i*npts_+j] )
    # Check the adjoint residual
    adj_abs_err = jnp.sqrt(jnp.sum(dJdK_vec*dJdK_vec))
    if (adj_it==0):
        adj_abs_err_0 = adj_abs_err
    adj_rel_err = adj_abs_err/adj_abs_err_0
    if (jnp.isnan(adj_abs_err)):
        print("ADJ==> No. %d iteration encounters NaN"%(adj_it))
        break
    else:
        print("ADJ==> No. %d iteration, adj_abs_err = %.4e, adj_rel_err = %.2e"%(adj_it, adj_abs_err, adj_rel_err))
    if (adj_abs_err<adj_abs_tol or adj_rel_err<adj_rel_tol):
        print("ADJ==> Converged with adj_abs_err = %.4e, adj_rel_err = %.2e after %d iterations"%(adj_abs_err, adj_rel_err, adj_it))
        break

case_name = "K_adj-N%d"%(npts_int)
dat_fname = "%s.npy"%(case_name)
jnp.save(dat_fname, grid_K)
print("%s is saved"%(dat_fname))

