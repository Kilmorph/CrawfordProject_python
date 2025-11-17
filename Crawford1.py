import numpy as np
from numpy import linalg as LA

filename = "acetaldehyde.dat"

h = 6.62607015e-34
c = 299792458
atomic_number = []
coordinate = []
atom_mass = [0.000, 1.007825, 4.002603, 6.015123, 9.021, 10.8135, 12.0106, 14.006855, 15.9994]
Bohr_to_Angstrom = 0.529177249
percm_to_mHz = 29979.2458

f = open(filename, "r")
number_of_atoms = int(f.readline())

def bond(i, j):
    vec_ij = coordinate[i] - coordinate[j]
    return np.linalg.norm(vec_ij)

def angle(i,j,k):
    vec_ij = coordinate[i] - coordinate[j]
    vec_kj = coordinate[k] - coordinate[j]
    cos_phi = np.dot(vec_ij, vec_kj) / (bond(i,j) * bond(k,j)) 
    return np.degrees(np.arccos(cos_phi))

def out_of_plane_angle(i,j,k,l):
    vec_jk = coordinate[j] - coordinate[k]
    vec_lk = coordinate[l] - coordinate[k]
    vec_ik = coordinate[i] - coordinate[k]
    cross_product = np.cross(vec_jk, vec_lk)
    norm = np.linalg.norm(cross_product)
    sin_theta = np.dot((cross_product/norm), (vec_ik/bond(i,k)))
    return np.degrees(np.arcsin(sin_theta))

def dihedral_angle(i,j,k,l):
    vec_ji = coordinate[j] - coordinate[i]
    vec_kj = coordinate[k] - coordinate[j]
    vec_lk = coordinate[l] - coordinate[k]
    cross_product_1 = np.cross(vec_ji, vec_kj)
    norm_1 = np.linalg.norm(cross_product_1)
    cross_product_2 = np.cross(vec_kj, vec_lk)
    norm_2 = np.linalg.norm(cross_product_2)
    cos_tau = np.dot((cross_product_1/norm_1), (cross_product_2/norm_2))
    return np.degrees(np.arccos(cos_tau))

def rotation_constant(I):
    amu_to_kg = 1.66053907e-27
    Bohr_to_m = 5.29177249e-11
    B = h / (8 * (np.pi**2) * c * I)
    return B / (amu_to_kg * ((Bohr_to_m)**2) * 100)

    
for line in f:
    parts = line.strip().split()
    atomic_number.append(int(parts[0]))
    coords = [float(parts[1]), float(parts[2]), float(parts[3])]
    coordinate.append(coords)
f.close()

coordinate = np.array(coordinate)

print(f"Number of atoms: ", number_of_atoms)
print("Input Cartesian coordinates:")
for i in range(0, number_of_atoms):
    print(f"{atomic_number[i]:<5d} {coordinate[i][0]:>18.12f} {coordinate[i][1]:>18.12f} {coordinate[i][2]:>18.12f}")
    
print("Intermolecular distances:")
for i in range(0, number_of_atoms):
    for j in range(0, i):
        print(f"{i} {j} {bond(i,j):>8.5f}")

print("Bond Angles:")
for i in range(0, number_of_atoms):
    for j in range(0, i):
        for k in range (0, j):
         if bond(i, j) < 4.0 and bond(j, k) < 4.0:
          print(f"{i} - {j} - {k} {angle(i,j,k):>10.6f}")

print("Out-of-plane angles:")
for i in range(0, number_of_atoms):
   for k in range(0, number_of_atoms):
      for j in range (0, number_of_atoms):
         for l in range(0, j):
            if i!=j and i!=k and i!=l and j!=k and k!=l and bond(i,k) < 4.0 and bond(k,j) < 4.0 and bond(k,l) < 4.0:
               print(f"{i} - {j} - {k} - {l} {out_of_plane_angle(i,j,k,l):>10.6f}")

print("Dihedral angles:")
for i in range(0, number_of_atoms):
   for j in range(0, i):
      for k in range(0, j):
        for l in range(0, k):
           if bond(i,j) < 4.0 and bond(j,k) < 4.0 and bond(k,l) < 4.0: 
               print(f"{i} - {j} - {k} - {l} {dihedral_angle(i,j,k,l):>10.6f}")

X_cm, Y_cm, Z_cm = 0, 0, 0
sum_mass = 0
mass = []
for i in range(0, number_of_atoms):
   X_cm += atom_mass[atomic_number[i]] * coordinate[i][0]
   Y_cm += atom_mass[atomic_number[i]] * coordinate[i][1]
   Z_cm += atom_mass[atomic_number[i]] * coordinate[i][2]
   sum_mass += atom_mass[atomic_number[i]]
   mass.append(atom_mass[atomic_number[i]])
   m = np.array(mass)
print(f"Centre-of-Mass: {X_cm/sum_mass:>18.12f} {Y_cm/sum_mass:>18.12f} {Z_cm/sum_mass:>18.12f}")

X_cm_f = X_cm / sum_mass
Y_cm_f = Y_cm / sum_mass
Z_cm_f = Z_cm / sum_mass
R_cm = np.array([X_cm_f, Y_cm_f, Z_cm_f])
coordinate_cm = coordinate - R_cm
x, y, z = coordinate_cm[:, 0], coordinate_cm[:, 1], coordinate_cm[:, 2]

I_xx = np.sum(m * (y**2 + z**2))
I_xy = np.sum(-m * x * y)
I_xz = np.sum(-m * x * z)
I_yy = np.sum(m * (x**2 + z**2))
I_yz = np.sum(-m * y * z)
I_zz = np.sum(m * (x**2 + y**2))

print("The Moment of inertia tensor (amu bohr^2):")
print(f"{I_xx:>18.12f} {I_xy:>18.12f} {I_xz:>18.12f}")
print(f"{I_xy:>18.12f} {I_yy:>18.12f} {I_yz:>18.12f}")
print(f"{I_xz:>18.12f} {I_yz:>18.12f} {I_zz:>18.12f}")

I = np.array([
    [I_xx, -I_xy, -I_xz],
    [-I_xy, I_yy, -I_yz],
    [-I_xz, -I_yz, I_zz]
])

eigvals, eigvecs = LA.eig(I)
principal_moments = np.sort(eigvals)

print("Principal moments of inertia (amu * bohr^2):")
print(f"{principal_moments[0]:>18.12f} {principal_moments[1]:>18.12f} {principal_moments[2]:>18.12f}")

print("Principal moments of inertia (amu * AA^2):")
print(f"{principal_moments[0]*(Bohr_to_Angstrom**2):>18.12f} {principal_moments[1]*(Bohr_to_Angstrom**2):>18.12f} {principal_moments[2]*(Bohr_to_Angstrom**2):>18.12f}")

if number_of_atoms == 2:
   print("Molecule is diatomic.")
else:
   if principal_moments[0] < 1e-4:
      print("Molecule is linear.")
   else:
      if np.abs(principal_moments[0]-principal_moments[1]) < 1e-4 and np.abs(principal_moments[1]-principal_moments[2]) < 1e-4:
         print("Molecule is a spherical top.")
      else:
         if np.abs(principal_moments[0]-principal_moments[1]) < 1e-4 and np.abs(principal_moments[1]-principal_moments[2]) > 1e-4:
            print("Molecule is an oblate symmetric top.")
         else:
            if np.abs(principal_moments[0]-principal_moments[1]) > 1e-4 and np.abs(principal_moments[1]-principal_moments[2]) < 1e-4:
               print("Molecule is a prolate symmetric top.")
            else:
               print("Molecule is an asymmetric top.")

print("Rotational constants (cm-1):")
print(f"A = {rotation_constant(principal_moments[0])} B = {rotation_constant(principal_moments[1])} C = {rotation_constant(principal_moments[2])}")

print("Rotational constants (mHz):")
print(f"A = {rotation_constant(principal_moments[0])*percm_to_mHz} B = {rotation_constant(principal_moments[1])*percm_to_mHz} C = {rotation_constant(principal_moments[2])*percm_to_mHz}")



