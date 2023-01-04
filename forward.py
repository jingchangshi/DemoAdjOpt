import jax.numpy as jnp
import jax.scipy as jsp
import jax
#  from jax import grad, jit, vmap, pmap
from tqdm import tqdm
import time

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

# For the analytic solution case, 
# check https://math.stackexchange.com/questions/3628949/solution-to-a-poisson-equation

#  bnd_x_l =-1.0
#  bnd_x_h = 1.0
#  bnd_y_l =-1.0
#  bnd_y_h = 1.0
#  bnd_T_ = 0.0
bnd_x_l = 0.0
bnd_x_h = 1.0
bnd_y_l = 0.0
bnd_y_h = 1.0
npts_int = 10
bnd_T_ = 200.0
beta1_ref = 13221.0
beta2_ref = 27.0
n_iter = 100
# FP32 cannot reach such tol. FP32 reaches only the order of magnitude of 1.
abs_tol = 1E-6
rel_tol = 1E-6
omega_ = 0.8
dat_fname = "T_ref-N%d.npy"%(npts_int)


#  npts_ = npts_int + 2
npts_ = npts_int + 1

@jax.jit
def compute_Q(x, y):
    gamma = 0.5E6
    # Negative RHS leads to the solution distribution being convex upwards
    q = -gamma*(1-(x-0.5)*(x-0.5))*(1-(y-0.5)*(y-0.5))
    #  q = -jnp.sin(jnp.pi*x) * jnp.sin(jnp.pi*y)
    return q

@jax.jit
def compute_K(T, beta1, beta2):
    K = beta1 / T + beta2
    #  K = 1.0
    return K
compute_dKdT = jax.grad(compute_K, argnums=0)

@jax.jit
def compute_R(T, Tl, Tr, Tb, Tt, q, beta1, beta2, dx):
    K = compute_K(T, beta1, beta2)
    dKdT = compute_dKdT(T, beta1, beta2)
    dTdx = (Tr - Tl)/(2.0*dx)
    dTdy = (Tt - Tb)/(2.0*dx)
    dKdx = dKdT * dTdx
    dKdy = dKdT * dTdy
    T_grad2 = (Tr - 2.0*T + Tl)/(dx*dx) \
            + (Tt - 2.0*T + Tb)/(dx*dx)
    R = K * T_grad2
    R += dKdx * dTdx + dKdy * dTdy
    R -= q
    return R
compute_dRdT = jax.grad(compute_R, argnums=[0,1,2,3,4])

#  @jax.jit
#  def assemblePoint(i, j, npts, grid_T, beta1, beta2, dx, R_vec, dRdT_dns):
#      T = grid_T[i, j]
#      #
#      Tl = grid_T[i, j-1]
#      Tr = grid_T[i, j+1]
#      Tb = grid_T[i-1, j]
#      Tt = grid_T[i+1, j]
#      #
#      # Left
#      if (j==0):
#         Tl = bnd_T_
#      # Right
#      if (j==npts_-1):
#         Tr = bnd_T_
#      # Bottom
#      if (i==0):
#         Tb = bnd_T_
#      # Top
#      if (i==npts_-1):
#         Tt = bnd_T_
#      # Compute R
#      #
#      #  i1 = i-1
#      #  j1 = j-1
#      #  i1j1 = i1*npts2+j1
#      #  R_vec = R_vec.at[i1j1].set(compute_R(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx))
#      #  dRdT_row = compute_dRdT(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx)
#      #  dRdT_dns = dRdT_dns.at[i1j1, i1j1].set(dRdT_row[0])
#      #  dRdT_dns = dRdT_dns.at[i1j1, i1j1-1].set(dRdT_row[1])
#      #  dRdT_dns = dRdT_dns.at[i1j1, i1j1+1].set(dRdT_row[2])
#      #  dRdT_dns = dRdT_dns.at[i1j1, i1j1-npts2].set(dRdT_row[3])
#      #  dRdT_dns = dRdT_dns.at[i1j1, i1j1+npts2].set(dRdT_row[4])
#      #
#      ij = i*npts+j
#      R_vec = R_vec.at[ij].set(compute_R(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx))
#      dRdT_row = compute_dRdT(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx)
#      dRdT_dns = dRdT_dns.at[ij, ij].set(dRdT_row[0])
#      if (j>0):
#          dRdT_dns = dRdT_dns.at[ij, ij-1].set(dRdT_row[1])
#      if (j<npts-1):
#          dRdT_dns = dRdT_dns.at[ij, ij+1].set(dRdT_row[2])
#      if (i>0):
#          dRdT_dns = dRdT_dns.at[ij, ij-npts].set(dRdT_row[3])
#      if (i<npts-1):
#          dRdT_dns = dRdT_dns.at[ij, ij+npts].set(dRdT_row[4])
#      return dRdT_dns, R_vec

@jax.jit
def assembleSystemNoGrad(grid_T, grid_q, beta1, beta2, dx):
    npts22 = npts_ * npts_
    R_vec = jnp.empty((npts22,))
    for i in range(npts_):
        for j in range(npts_):
            T = grid_T[i, j]
            # 
            Tl = grid_T[i, j-1]
            Tr = grid_T[i, j+1]
            Tb = grid_T[i-1, j]
            Tt = grid_T[i+1, j]
            # 
            # Left
            if (j==0):
               Tl = bnd_T_
            # Right
            if (j==npts_-1):
               Tr = bnd_T_
            # Bottom
            if (i==0):
               Tb = bnd_T_
            # Top
            if (i==npts_-1):
               Tt = bnd_T_
            # Compute R
            ij = i*npts_+j
            R_vec = R_vec.at[ij].set(compute_R(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx))
    return R_vec

@jax.jit
def assembleSystem(grid_T, grid_q, beta1, beta2, dx, require_assemble_system=True):
    #  npts2 = npts_-2
    #  npts22 = npts2 * npts2
    npts22 = npts_ * npts_
    R_vec = jnp.empty((npts22,))
    dRdT_dns = jnp.zeros((npts22, npts22))
    #  for i in range(1,npts_-1):
    #      for j in range(1,npts_-1):
    for i in range(npts_):
        for j in range(npts_):
            #  dRdT_dns, R_vec = assemblePoint(i, j, npts_, grid_T, beta1, beta2, dx, R_vec, dRdT_dns)
            T = grid_T[i, j]
            # 
            Tl = grid_T[i, j-1]
            Tr = grid_T[i, j+1]
            Tb = grid_T[i-1, j]
            Tt = grid_T[i+1, j]
            # 
            # Left
            if (j==0):
               Tl = bnd_T_
            # Right
            if (j==npts_-1):
               Tr = bnd_T_
            # Bottom
            if (i==0):
               Tb = bnd_T_
            # Top
            if (i==npts_-1):
               Tt = bnd_T_
            # Compute R
            # 
            #  i1 = i-1
            #  j1 = j-1
            #  i1j1 = i1*npts2+j1
            #  R_vec = R_vec.at[i1j1].set(compute_R(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx))
            #  dRdT_row = compute_dRdT(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx)
            #  dRdT_dns = dRdT_dns.at[i1j1, i1j1].set(dRdT_row[0])
            #  dRdT_dns = dRdT_dns.at[i1j1, i1j1-1].set(dRdT_row[1])
            #  dRdT_dns = dRdT_dns.at[i1j1, i1j1+1].set(dRdT_row[2])
            #  dRdT_dns = dRdT_dns.at[i1j1, i1j1-npts2].set(dRdT_row[3])
            #  dRdT_dns = dRdT_dns.at[i1j1, i1j1+npts2].set(dRdT_row[4])
            #
            ij = i*npts_+j
            R_vec = R_vec.at[ij].set(compute_R(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx))
            if (require_assemble_system):
                dRdT_row = compute_dRdT(T, Tl, Tr, Tb, Tt, grid_q[i, j], beta1, beta2, dx)
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

grid_1d = jnp.linspace(bnd_x_l, bnd_x_h, npts_)
dx = grid_1d[1] - grid_1d[0]
dy = grid_1d[1] - grid_1d[0]
grid_x, grid_y = jnp.meshgrid(grid_1d, grid_1d)
grid_q = compute_Q(grid_x, grid_y)
# Init solution
grid_T = jnp.ones((npts_, npts_)) * bnd_T_

t_beg = time.perf_counter()
it = 0
abs_res = 0.0
R_vec = assembleSystemNoGrad(grid_T, grid_q, beta1_ref, beta2_ref, dx)
abs_res_0 = jnp.sqrt(jnp.sum(R_vec*R_vec))
#  for it in tqdm(range(n_iter)):
for it in range(n_iter):
    # Assemble the linear system
    dRdT_dns, R_vec = assembleSystem(grid_T, grid_q, beta1_ref, beta2_ref, dx)
    abs_res = jnp.sqrt(jnp.sum(R_vec*R_vec))
    rel_res = abs_res/abs_res_0
    if (jnp.isnan(abs_res)):
        print("No. %d iteration encounters NaN"%(it))
        break
    else:
        print("No. %d iteration, abs_res = %.4e, rel_res = %.2e"%(it, abs_res, rel_res))
    if (abs_res<abs_tol or rel_res<rel_tol):
        print("Converged with abs_res = %.4e rel_res = %.2e after %d iterations"%(abs_res, rel_res, it))
        break
    # Solve the linear system
    b_vec = -R_vec
    #  dT = jsp.linalg.solve(dRdT_dns, b_vec, assume_a='pos')
    dT = jsp.linalg.solve(dRdT_dns, b_vec)
    # Update the solution
    #  for i in range(1, npts_-1):
    #      for j in range(1, npts_-1):
            #  grid_T = grid_T.at[i, j].set( grid_T[i, j] + omega_*dT[(i-1)*(npts_-2)+(j-1)])
    for i in range(npts_):
        for j in range(npts_):
            grid_T = grid_T.at[i, j].set( grid_T[i, j] + omega_*dT[i*npts_+j])
if (it == n_iter-1):
    it += 1
    dRdT_dns, R_vec = assembleSystem(grid_T, grid_q, beta1_ref, beta2_ref, dx)
    abs_res = jnp.sqrt(jnp.sum(R_vec*R_vec))
    rel_res = abs_res/abs_res_0
    if (jnp.isnan(abs_res)):
        print("No. %d iteration encounters NaN"%(it))
    else:
        print("No. %d iteration, abs_res = %.4e, rel_res = %.2e"%(it, abs_res, rel_res))
    if (abs_res<abs_tol or rel_res<rel_tol):
        print("Converged with abs_res = %.4e rel_res = %.2e after %d iterations"%(abs_res, rel_res, it))
    else:
        print("Not converged with abs_res = %.4e rel_res = %.2e after %d iterations"%(abs_res, rel_res, it))
t_end = time.perf_counter()
print("Time cost is %.2f seconds"%(t_end-t_beg))
jnp.save(dat_fname, grid_T)
print("%s is saved"%(dat_fname))
