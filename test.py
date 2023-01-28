import numpy as np

xx = [1, 2, 3]
yy = [4, 5, 6]

# Return two 2D lists combining the coordinates in xx and yy
YY, XX = np.meshgrid(yy, xx)
print('XX: \n' , XX, '\nYY\n', YY)
print('XX ravel: \n' , XX.ravel(), '\nYY ravel\n', YY.ravel())

# Ravel flattens each array into 1D. Vstack joins the two 1D arrays into a 2D array, and is transposed [[x1, y1], [x2, y2], ... [xn, yn]]
xy = np.vstack([XX.ravel(), YY.ravel()])

print('xy: \n', xy)
print('xy transposed: \n', xy.T)