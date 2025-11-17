import numpy as np
from numpy import linalg as LA

np.set_printoptions(precision=7, suppress=True, linewidth=150)

filename = "enuc.dat"

f = open(filename, "r")
enuc = float(f.readline())

dimension = 7
N_occ = 5
s_matrix, t_matrix, v_matrix = np.zeros((dimension, dimension), dtype=float), np.zeros((dimension, dimension), dtype=float), np.zeros((dimension, dimension), dtype=float)
eri_tensor = np.zeros((dimension, dimension, dimension, dimension))
delta_threshold = 1.0e-8
rms_threshold = 1.0e-5


for line in open("s.dat", 'r'):
    parts = line.strip().split()

    row_index = int(parts[0])-1
    col_index = int(parts[1])-1
    value = float(parts[2])

    s_matrix[row_index][col_index] = value
    s_matrix[col_index][row_index] = value


for line in open("t.dat", 'r'):
    parts = line.strip().split()

    row_index = int(parts[0])-1
    col_index = int(parts[1])-1
    value = float(parts[2])

    t_matrix[row_index][col_index] = value
    t_matrix[col_index][row_index] = value

for line in open("v.dat", 'r'):
    parts = line.strip().split()

    row_index = int(parts[0])-1
    col_index = int(parts[1])-1
    value = float(parts[2])

    v_matrix[row_index][col_index] = value
    v_matrix[col_index][row_index] = value

H_core = t_matrix + v_matrix


for line in open("eri.dat", 'r'):
    parts = line.strip().split()

    i, j, k, l = map(int, parts[:4])
    i, j, k, l = i-1, j-1, k-1, l-1
    value = float(parts[4])
    
    indices_tuples = [
        (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
        (k, l ,i, j), (l, k, i, j), (k, l, j, i), (l, k ,j, i)
    ]

    unique_indices = set(indices_tuples)

    p_idx, q_idx, r_idx, s_idx = [], [], [], []
    for p,q,r,s in unique_indices:
        p_idx.append(p)
        q_idx.append(q)
        r_idx.append(r)
        s_idx.append(s)
    
    eri_tensor[p_idx, q_idx, r_idx, s_idx] = value

def diagonalization_S(S):
    eigvals,eigvecs = LA.eigh(S)
    Lambda_S = np.diag((eigvals) ** (-0.5))
    X = np.dot(np.dot(eigvecs, Lambda_S), eigvecs.T)
    return X

def density(X,F):
    F_prime = np.dot(np.dot(X.T, F), X)
    eigvals,eigvecs = LA.eigh(F_prime)
    C = np.dot(X, eigvecs)
    C_occ = C[:, :N_occ]
    D = np.dot(C_occ, C_occ.T)
    return D, C, eigvals

def scf(D, H, F):
    E_elec = 0
    for i in range(0, dimension):
        for j in range(0, dimension):
            E_elec += D[i][j] * (H[i][j] + F[i][j])
    return E_elec 

def fock(D, H, eri):
    F = H.copy()
    J = np.einsum('pqrs,rs->pq', eri, D)
    K = np.einsum('prqs,rs->pq', eri, D)
    F = H + 2*J - K
    return F

def error_matrix(F, D):
    term1 = np.dot(np.dot(F,D),s_matrix)
    term2 = np.dot(np.dot(s_matrix,D),F)
    return term1-term2



X = diagonalization_S(s_matrix)
D = density(X, H_core)[0]
F = H_core
E_elec_old = None
D_old = D.copy()
C = None
orbital_energies = None

for i in range(100):
    F_old = F
    E_elec = scf(D, H_core, F)
    print(f"Circle {i+1}, E_elec = {E_elec} Hartree, E_tol = {E_elec + enuc} Hartree.")

    if E_elec_old is not None:
        delta_E = np.abs(E_elec - E_elec_old)
        rms_D = np.sqrt(np.sum((D - D_old)**2))
        print(f"Convergence Check: Delta_E = {delta_E:.2e}, RMS_D = {rms_D:.2e}")
        if delta_E < delta_threshold and rms_D < rms_threshold:
            print("\n--- SCF Converged ---")
            print("Final Total Energy (E_tol):", E_elec + enuc)

            print("\nOrbital Energies (eigvals from diagonalization):")
            print(orbital_energies)

            print("\nPerforming AO-to-MO transformation: F_mo = C.T @ F @ C")

            F_mo = C.T @ F @ C

            print("\nMO-Basis Fock Matrix (F_mo):")
            print(F_mo)

            print("\nMO coefficients matrix:")
            print(C)
            break
    D_old = D
    F = fock(D_old, H_core, eri_tensor)
    D,C,orbital_energies = density(X, F)
    E_elec_old = E_elec

print("\n--- Starting MP2 Calculation ---")
print("Preparing the MO basis ERI...")

step_1 = np.einsum('uvls, sS -> uvlS', eri_tensor, C, optimize=True)
step_2 = np.einsum('uvlS, lR -> uvRS', step_1, C, optimize=True)
step_3 = np.einsum('uvRS, vQ -> uQRS', step_2, C, optimize=True)
eri_tensor_MO = np.einsum('uQRS, uP -> PQRS', step_3, C, optimize=True)
print(f"MO ERI tensor (pq|rs) created with shape: {eri_tensor_MO.shape}")

print("\nPreparing the MP2 energy...")
eps_occ = orbital_energies[:N_occ]
eps_vir = orbital_energies[N_occ:]

eps_i = eps_occ[:,None,None,None]
eps_j = eps_occ[None,None,:,None]
eps_a = eps_vir[None,:,None,None]
eps_b = eps_vir[None,None,None,:]

denominator = eps_i + eps_j - eps_a - eps_b

O = slice(None,N_occ)
V = slice(N_occ,None)

I_iajb = eri_tensor_MO[O,V,O,V]
I_ibja = eri_tensor_MO[O,V,O,V].transpose(0, 3, 2, 1)

numerator = I_iajb * (2*I_iajb-I_ibja)

E_MP2_tensor = numerator / denominator
E_MP2 = np.sum(E_MP2_tensor)
print(f"\n--- MP2 Calculation Complete ---")
print(f"MP2 Correlation Energy: {E_MP2:.8f} Hartree")
print(f"Total HF Energy (E_tol):  {E_elec + enuc:.8f} Hartree")
print(f"Total MP2 Energy:         {E_elec + enuc + E_MP2:.8f} Hartree")

print("\n--- Starting CCSD Calculation ---")
print("Preparing the spin-orbital basis Fock Matrix...")
N_spatial = dimension
N_spin = N_spatial * 2
f_pq = np.diag(np.repeat(orbital_energies,2))
spin_orbital_energies = np.diag(f_pq)

print("Preparing spin-orbital eri tensor...")
eri_tensor_spin = np.zeros((N_spin, N_spin, N_spin, N_spin))
for p in range(N_spin):
    for q in range(N_spin):
        for r in range(N_spin):
            for s in range (N_spin):
                P = p // 2
                Q = q // 2
                R = r // 2
                S = s // 2

                sigma_p = p % 2
                sigma_q = q % 2
                sigma_r = r % 2
                sigma_s = s % 2
                
                if (sigma_p == sigma_r) and (sigma_q == sigma_s):
                    eri_tensor_spin[p,q,r,s] = eri_tensor_MO[P,R,Q,S]

print("Preparing the initial guess of T1 and T2...")
N_occ_spin = N_occ * 2
N_vir_spin = N_spin - N_occ_spin

T1 = np.zeros((N_occ_spin, N_vir_spin))

O_spin = slice(None, N_occ_spin)
V_spin = slice(N_occ_spin, None)

eps_occ_spin = spin_orbital_energies[O_spin]
eps_vir_spin = spin_orbital_energies[V_spin]

eps_i_spin = eps_occ_spin[:,None,None,None]
eps_j_spin = eps_occ_spin[None,:,None,None]
eps_a_spin = eps_vir_spin[None,None,:,None]
eps_b_spin = eps_vir_spin[None,None,None,:]

D_ijab = eps_i_spin + eps_j_spin - eps_a_spin - eps_b_spin

I_ijab = eri_tensor_spin[O_spin,O_spin,V_spin,V_spin]
I_ijba = eri_tensor_spin[O_spin,O_spin,V_spin,V_spin].transpose(0,1,3,2)
as_ijab = I_ijab - I_ijba

T2 = as_ijab / D_ijab

def tau_intermediates(T1, T2):
    t1_t1 = np.einsum('ia, jb -> ijab', T1, T1, optimize=True)
    T1_product = t1_t1 - t1_t1.transpose(0,1,3,2)
    tau = T2 + T1_product
    tau_tilde = T2 + 0.5 * T1_product
    return tau, tau_tilde

eri_VOVO = eri_tensor_spin[V_spin,O_spin,V_spin,O_spin]
eri_VOVV = eri_tensor_spin[V_spin,O_spin,V_spin,V_spin]
eri_OVVV = eri_tensor_spin[O_spin,V_spin,V_spin,V_spin]
eri_OOVV = eri_tensor_spin[O_spin,O_spin,V_spin,V_spin]
eri_OOOV = eri_tensor_spin[O_spin,O_spin,O_spin,V_spin]
eri_OOOO = eri_tensor_spin[O_spin,O_spin,O_spin,O_spin]
eri_VVVV = eri_tensor_spin[V_spin,V_spin,V_spin,V_spin]
eri_OVVO = eri_tensor_spin[O_spin,V_spin,V_spin,O_spin]
eri_OOVO = eri_tensor_spin[O_spin,O_spin,V_spin,O_spin]
eri_OVOV = eri_tensor_spin[O_spin,V_spin,O_spin,V_spin] 
eri_VVVO = eri_tensor_spin[V_spin,V_spin,V_spin,O_spin] 
eri_VVOV = eri_tensor_spin[V_spin,V_spin,O_spin,V_spin] 
eri_OVOO = eri_tensor_spin[O_spin,V_spin,O_spin,O_spin]

f_oo = f_pq[O_spin,O_spin]
f_vv = f_pq[V_spin,V_spin]
f_ov = f_pq[O_spin,V_spin]
D_ia = np.diag(f_oo)[:,None]-np.diag(f_vv)[None,:]

def F_intermediates(T1, tau_tilde):
    term_ae_1 = np.einsum('mf, mafe -> ae', T1, eri_OVVV, optimize=True)
    term_ae_2 = np.einsum('mf, maef -> ae', T1, eri_OVVV, optimize=True)
    term_ae_a = term_ae_1 - term_ae_2

    term_ae_3 = np.einsum('mnaf, mnef -> ae', tau_tilde, eri_OOVV, optimize=True)
    term_ae_4 = np.einsum('mnaf, nmef -> ae', tau_tilde, eri_OOVV, optimize=True)
    term_ae_b = -0.5 * (term_ae_3 - term_ae_4)

    F_ae = term_ae_a + term_ae_b

    term_mi_1 = np.einsum('ne, mnie -> mi', T1, eri_OOOV, optimize=True)
    term_mi_2 = np.einsum('ne, nmie -> mi', T1, eri_OOOV, optimize=True)
    term_mi_a = term_mi_1 - term_mi_2

    term_mi_3 = np.einsum('inef, mnef -> mi', tau_tilde, eri_OOVV, optimize=True)
    term_mi_4 = np.einsum('inef, nmef -> mi', tau_tilde, eri_OOVV, optimize=True)
    term_mi_b = 0.5 * (term_mi_3 -term_mi_4)

    F_mi = term_mi_a + term_mi_b

    term_me_a = np.einsum('nf, mnef -> me', T1, eri_OOVV, optimize=True)
    term_me_b = np.einsum('nf, nmef -> me', T1, eri_OOVV, optimize=True)

    F_me = term_me_a - term_me_b

    return F_ae, F_mi, F_me

def W_intermediates(T1, T2, tau):
    term_mnij_a = eri_OOOO - eri_OOOO.transpose(1,0,2,3)

    term_mnij_1 = np.einsum('je, mnie -> mnij', T1, eri_OOOV, optimize=True)
    term_mnij_2 = np.einsum('je, nmie -> mnij', T1, eri_OOOV, optimize=True)
    term_mnij_3 = np.einsum('ie, mnje -> mnij', T1, eri_OOOV, optimize=True)
    term_mnij_4 = np.einsum('ie, nmje -> mnij', T1, eri_OOOV, optimize=True)
    term_mnij_b = term_mnij_1 - term_mnij_2 - (term_mnij_3 -term_mnij_4)

    term_mnij_5 = np.einsum('ijef, mnef -> mnij', tau, eri_OOVV, optimize=True)
    term_mnij_6 = np.einsum('ijef, nmef -> mnij', tau, eri_OOVV, optimize=True)
    term_mnij_c = (term_mnij_5 - term_mnij_6) / 4

    W_mnji = term_mnij_a + term_mnij_b + term_mnij_c

    term_abef_a = eri_VVVV - eri_VVVV.transpose(1,0,2,3)

    term_abef_1 = np.einsum('mb, amef -> abef', T1, eri_VOVV, optimize=True)
    term_abef_2 = np.einsum('mb, amfe -> abef', T1, eri_VOVV, optimize=True)
    term_abef_3 = np.einsum('ma, bmef -> abef', T1, eri_VOVV, optimize=True)
    term_abef_4 = np.einsum('ma, bmfe -> abef', T1, eri_VOVV, optimize=True)
    term_abef_b = term_abef_1 - term_abef_2 - (term_abef_3 -term_abef_4)

    term_abef_5 = np.einsum('mnab, mnef -> abef', tau, eri_OOVV, optimize=True)
    term_abef_6 = np.einsum('mnab, nmef -> abef', tau, eri_OOVV, optimize=True)
    term_abef_c = (term_abef_5 - term_abef_6) / 4

    W_abef = term_abef_a - term_abef_b + term_abef_c

    term_mbej_a = eri_OVVO - eri_OVOV.transpose(0,1,3,2)

    term_mbej_1 = np.einsum('jf, mbef -> mbej', T1, eri_OVVV, optimize=True)
    term_mbej_2 = np.einsum('jf, mbfe -> mbej', T1, eri_OVVV, optimize=True)
    term_mbej_b = term_mbej_1 -term_mbej_2

    term_mbej_3 = np.einsum('nb, mnej -> mbej', T1, eri_OOVO, optimize=True)
    term_mbej_4 = np.einsum('nb, nmej -> mbej', T1, eri_OOVO, optimize=True)
    term_mbej_c = term_mbej_3 - term_mbej_4

    as_OOVV = eri_OOVV - eri_OOVV.transpose(0,1,3,2)
    term_mbej_5 = 0.5 * np.einsum('jnfb, mnef -> mbej', T2, as_OOVV, optimize=True)
    term_mbej_6 = np.einsum('jf, nb, mnef -> mbej', T1, T1, as_OOVV, optimize=True)
    term_mbej_d = term_mbej_5 + term_mbej_6

    W_mbej = term_mbej_a + term_mbej_b - term_mbej_c - term_mbej_d

    return W_mnji, W_abef, W_mbej

def update_T1(T1, T2, F_ae, F_mi, F_me):
    RHS_T1 = f_ov.copy()
    RHS_T1 +=  np.einsum('ie, ae -> ia', T1, F_ae, optimize=True)
    RHS_T1 -=  np.einsum('ma, mi -> ia', T1, F_mi, optimize=True)
    RHS_T1 +=  np.einsum('imae, me -> ia', T2, F_me, optimize=True)

    RHS_T1 -= np.einsum('mf, maif -> ia', T1, eri_OVOV, optimize=True)

    RHS_T1 += np.einsum('mf, mafi -> ia', T1, eri_OVVO, optimize=True)
    
    as_OVVV = eri_OVVV - eri_OVVV.transpose(0,1,3,2) 
    RHS_T1 -= 0.5 * np.einsum('imef, maef -> ia', T2, as_OVVV, optimize=True)
    
    I_nmei = eri_OOVO 
    I_nmie = eri_OOOV.transpose(0,1,3,2) 
    as_OOOV_nmei = I_nmei - I_nmie
    
    RHS_T1 -= 0.5 * np.einsum('mnae, nmei -> ia', T2, as_OOOV_nmei, optimize=True)
    
    return RHS_T1

def update_T2(T1, T2, tau, F_ae, F_mi, F_me, W_mnji, W_abef, W_mbej):

    RHS_T2 = np.zeros_like(T2) 

    as_OOVV = eri_OOVV - eri_OOVV.transpose(0, 1, 3, 2)
    RHS_T2 += as_OOVV

    X_e = F_ae - 0.5 * np.einsum('mb, me -> be', T1, F_me, optimize=True)
    X_ijab = np.einsum('ijae, be -> ijab', T2, X_e, optimize=True) 
    RHS_T2 += X_ijab - X_ijab.transpose(0, 1, 3, 2) 

    Y_m = F_mi + 0.5 * np.einsum('je, me -> mj', T1, F_me, optimize=True)
    Y_ijab = np.einsum('imab, mj -> ijab', T2, Y_m, optimize=True)
    RHS_T2 -= (Y_ijab - Y_ijab.transpose(1, 0, 2, 3)) 
 
    RHS_T2 += 0.5 * np.einsum('mnab, mnij -> ijab', tau, W_mnji, optimize=True)

    RHS_T2 += 0.5 * np.einsum('ijef, abef -> ijab', tau, W_abef, optimize=True)

    Z_ijab = np.einsum('imae, mbej -> ijab', T2, W_mbej, optimize=True) 
    RHS_T2 += Z_ijab \
            - Z_ijab.transpose(1,0,2,3) \
            - Z_ijab.transpose(0,1,3,2) \
            + Z_ijab.transpose(1,0,3,2)

    as_OVVO = eri_OVVO - eri_OVOV.transpose(0,1,3,2)
    K_ijab = np.einsum('ie, ma, mbej -> ijab', T1, T1, as_OVVO, optimize=True)
    RHS_T2 -= (K_ijab - K_ijab.transpose(1,0,2,3) - K_ijab.transpose(0,1,3,2) + K_ijab.transpose(1,0,3,2)) 

    as_VVVO = eri_VVVO - eri_VVOV.transpose(0,1,3,2)
    L_ijab = np.einsum('ie, abej -> ijab', T1, as_VVVO, optimize=True)
    RHS_T2 += L_ijab - L_ijab.transpose(1,0,2,3) 

    as_OVOO = eri_OVOO - eri_OVOO.transpose(0, 1, 3, 2)
    M_ijab = np.einsum('ma, mbij -> ijab', T1, as_OVOO, optimize=True)
    RHS_T2 -= (M_ijab - M_ijab.transpose(0,1,3,2)) 

    return RHS_T2

def E_CCSD(T1, T2):
    as_OOVV = eri_OOVV - eri_OOVV.transpose(0,1,3,2)
    term_1 = np.einsum('ijab, ijab ->', T2, as_OOVV, optimize=True)
    term_2 = np.einsum('ia, jb, ijab ->', T1, T1, as_OOVV, optimize=True)
    return (term_1 / 4 + term_2 / 2)

print("\n--- Starting CCSD Iteration Loop ---")

max_iter = 50
E_threshold = 1.0e-7
E_ccsd_old = 0.0

for i in range(max_iter):

    T1_old = T1.copy()
    T2_old = T2.copy()

    tau, tau_tilde = tau_intermediates(T1,T2)
    F_ae, F_mi, F_me = F_intermediates(T1,tau_tilde)
    W_mnji, W_abef, W_mbej = W_intermediates(T1,T2,tau)

    RHS_T1 = update_T1(T1, T2, F_ae, F_mi, F_me)
    T1_new = RHS_T1 / D_ia

    RHS_T2 = update_T2(T1, T2, tau, F_ae, F_mi, F_me, W_mnji, W_abef, W_mbej)
    T2_new = RHS_T2 / D_ijab

    E_ccsd_new = E_CCSD(T1_new,T2_new)

    delta_E = np.abs(E_ccsd_new - E_ccsd_old)
    rms_T1 = np.sqrt(np.mean((T1_new - T1_old)**2))
    rms_T2 = np.sqrt(np.mean((T2_new - T2_old)**2))
    
    print(f"Iter: {i+1:2d}   E_corr: {E_ccsd_new:.12f}   Delta_E: {delta_E:.2e}   RMS_T1: {rms_T1:.2e}   RMS_T2: {rms_T2:.2e}")
    
    if delta_E < E_threshold:
        print("\n--- CCSD Converged! ---")
        break
    
    T1 = T1_new
    T2 = T2_new
    E_ccsd_old = E_ccsd_new


E_CCSD_total = E_elec + enuc + E_ccsd_new 
print(f"\nFinal CCSD Correlation Energy: {E_ccsd_new:.8f} Hartree")
print(f"Final Total CCSD Energy:     {E_CCSD_total:.8f} Hartree")

print("\n--- Starting (T) Calculation ---")
e_i = np.diag(f_oo)[:,None,None,None,None,None]
e_j = np.diag(f_oo)[None,:,None,None,None,None]
e_k = np.diag(f_oo)[None,None,:,None,None,None]
e_a = np.diag(f_vv)[None,None,None,:,None,None]
e_b = np.diag(f_vv)[None,None,None,None,:,None]
e_c = np.diag(f_vv)[None,None,None,None,None,:]

D_ijkabc = e_i + e_j + e_k - e_a - e_b - e_c

print("Building 'Disconnected' base tensor V_d...")
as_OOVV = eri_OOVV - eri_OOVV.transpose(0,1,3,2) 
V_d = np.einsum('ia, jkbc -> ijkabc', T1, as_OOVV, optimize=True)

print("Building 'Connected' base tensor V_c...")

as_VOVV = eri_VOVV - eri_VOVV.transpose(0,1,3,2) 
TermA = np.einsum('jkae, eibc -> ijkabc', T2, as_VOVV, optimize=True)


as_OVOO = eri_OVOO - eri_OVOO.transpose(0,1,3,2) 
TermB = -np.einsum('imbc, majk -> ijkabc', T2, as_OVOO, optimize=True)

V_c = TermA + TermB 

# P(p/qr)f(pqr) = f(pqr) - f(qpr) - f(rqp)
# D = P(i/jk) [ P(a/bc)[V] ]
# D = P(i/jk) [ V(abc) - V(bac) - V(cba) ]
# D = [V(i,jk,a,bc) - V(i,jk,b,ac) - V(i,jk,c,ba)]  (f(pqr))
#   - [V(j,ik,a,bc) - V(j,ik,b,ac) - V(j,ik,c,ba)]  (-f(qpr))
#   - [V(k,ji,a,bc) - V(k,ji,b,ac) - V(k,ji,c,ba)]  (-f(rqp))
print("Applying 9-term permutation P(i/jk)P(a/bc)...")

def apply_permutation(V):
    # V(i,j,k, a,b,c)
    V_abc = V
    # V(i,j,k, b,a,c)
    V_bac = V.transpose(0,1,2, 4,3,5)
    # V(i,j,k, c,b,a)
    V_cba = V.transpose(0,1,2, 5,4,3)
    
    # f(pqr) -> [V(abc) - V(bac) - V(cba)]
    term_i = V_abc - V_bac - V_cba

    # V(j,i,k, a,b,c)
    V_jik = V.transpose(1,0,2, 3,4,5)
    # V(j,i,k, b,a,c)
    V_jbk = V.transpose(1,0,2, 4,3,5)
    # V(j,i,k, c,b,a)
    V_jck = V.transpose(1,0,2, 5,4,3)

    # -f(qpr) -> -[V(jik,abc) - V(jik,bac) - V(jik,cba)]
    term_j = V_jik - V_jbk - V_jck

    # V(k,j,i, a,b,c)
    V_kji = V.transpose(2,1,0, 3,4,5)
    # V(k,j,i, b,a,c)
    V_kbi = V.transpose(2,1,0, 4,3,5)
    # V(k,j,i, c,b,a)
    V_kci = V.transpose(2,1,0, 5,4,3)

    # -f(rqp) -> -[V(kji,abc) - V(kji,bac) - V(kji,cba)]
    term_k = V_kji - V_kbi - V_kci
    
    # D = f(pqr) - f(qpr) - f(rqp)
    return term_i - term_j - term_k

D_d = apply_permutation(V_d)
D_c = apply_permutation(V_c)

print("Permutations complete.")

Numerator = D_c * (D_c + D_d)
E_T_tensor = Numerator / D_ijkabc

E_T = (1.0 / 36.0) * np.sum(E_T_tensor)

E_CCSD_T_total = E_CCSD_total + E_T

print(f"\n--- CCSD(T) Calculation Complete ---")
print(f"E(T) Correction:    {E_T:.12f} Hartree")
print(f"Final Total CCSD(T) Energy: {E_CCSD_T_total:.12f} Hartree")

