Author: Tu Yuanyang

The program is an implementation of the Harris corner detection algorithm

Features being implemented:
1. Conversion from RGB image to gray scale by using the Y formula
2. Calculation of the gradients of the gray scale image in both X and Y direction and their square and product,
i.e. Ix, Iy, Ix ^ 2, Iy ^ 2 , Ix * Iy
3. Smoothing of all the above 2D images by convolving with a 1D Gaussian filters along x and y direction respectively
3a. The size of the filter is valued based on its bound related to the sigma value
3b. The filter is constructed based on the Gaussian formula without normalizing, and partial filtering can be achieved
with the help of a dummy 2D image with all values being 1
4. Constructing the cornerness function based on Ix ^ 2, Iy ^ 2 , Ix * Iy
5. Finding corner candidates using non-maximal suppression by locating the local maxima in a range of 8 neighbors
6. Computing the location of the corner candidates up to sub-pixel accuracy by quadratic approximation
7. Thresholding all corner candidates to identify strong corners for output.
8. Outputing corners detected to the output.txt file
