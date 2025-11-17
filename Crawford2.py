import numpy as np
from numpy import linalg as LA

filename = "h2o_hessian.txt"

f = open(filename, "r")
number_of_atoms = int(f.readline())
dimension = 3 * number_of_atoms

mass_O = 15.999
mass_H = 1.008

hessian_matrix = np.zeros((dimension,dimension))

matrix_row_build = 0
matrix_row_read = 0

for line in f:
    parts = line.strip().split()
    
    values = [float(p) for p in parts]

    start_col = matrix_row_read*3
    end_col = start_col + 3

    hessian_matrix[matrix_row_build, start_col:end_col]=values

    matrix_row_read += 1
    if matrix_row_read == 3:
        matrix_row_build += 1
        matrix_row_read = 0
        

    if matrix_row_build >= dimension:
        break

print("Read the Cartesian Hessian Data:")
np.set_printoptions(linewidth=150, precision=8, suppress=True)
print(hessian_matrix)

atomic_masses = np.array([mass_O] * 3 + [mass_H] * 6)

mass_weight = np.sqrt(atomic_masses)
weigt_matrix = np.outer(mass_weight, mass_weight)

weighted_hessian = hessian_matrix / weigt_matrix

print("Mass-Weight the Hessian Matrix:")
np.set_printoptions(linewidth=150, precision=8, suppress=True)
print(weighted_hessian)

eigvals, eigvecs = LA.eig(weighted_hessian)
eigenvalues = np.sort(eigvals)
print(eigenvalues)




