################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    img_gray = img_color @ [0.299,0.587,0.114]
    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    n = int(sigma * (2 * np.log(1000)) ** 0.5)
    x = np.arange(-n, n + 1)
    filter = np.exp((x ** 2) / -2 / (sigma ** 2))

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    w, h = img.shape
    dummy = np.ones([w,h], dtype = np.float64)

    img_smoothed = convolve1d(img, filter, 1, np.float64, 'constant', 0, 0)/convolve1d(dummy, filter, 1, np.float64, 'constant', 0, 0)
    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img_smoothed = smooth1D(img, sigma)

    # TODO: smooth the image along the horizontal direction
    img_smoothed = smooth1D(img_smoothed.T, sigma).T

    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner
    # TODO: compute Ix & Iy
    Ix = np.gradient(img,axis=1)
    Iy = np.gradient(img,axis=0)
    # TODO: compute Ix2, Iy2 and IxIy
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # TODO: smooth the squared derivatives
    Ix2 = smooth2D(Ix2,sigma)
    Iy2 = smooth2D(Iy2,sigma)
    IxIy = smooth2D(IxIy,sigma)


    # TODO: compute cornesness functoin R
    det = Ix2 * Iy2 - IxIy ** 2
    trace = Ix2 + Iy2
    R = det - 0.04 * trace ** 2

    # plt.imshow(np.float32(R), cmap = 'gray')
    # plt.show()

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy
    corner_candidates = []
    w, h = R.shape

    for row in range(1,w-1):
        for column in range(1,h-1):
            r = R[row][column]
            l_max = True
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    y = row-1+i
                    x = column-1+j
                    if R[y][x] >= r:
                        l_max = False
            if l_max:
                a = (R[row][column-1] + R[row,column+1] - 2 * R[row][column])/2
                b = (R[row-1][column] + R[row+1][column] - 2 * R[row][column])/2
                c = (R[row][column+1] - R[row][column-1])/2
                d = (R[row+1][column] - R[row-1][column])/2
                e = R[row][column]
                x1 = -c/(2*a)
                y1 = -d/(2*b)
                corner_candidates.append([column + x1, row + y1, a*x1**2 + b*y1**2 + c*x1 + d*y1 + e])

    # TODO: perform thresholding and discard weak corners
    corners = []
    for i in corner_candidates:
        if i[2] > threshold:
            corners.append(i)

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#  show corner detection result
################################################################################
def show_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]

    plt.ion()
    fig = plt.figure('Harris corner detection')
    plt.imshow(img_color)
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  load image from a file
################################################################################
def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  save corners to a file
################################################################################
def save_corners(outputfile, corners) :
    # input:
    #    outputfile - path of the output file
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(outputfile, 'w')
        file.write('{}\n'.format(len(corners)))
        for corner in corners :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(corner[0], corner[1], corner[2]))
        file.close()
    except :
        print('Error occurs in writting output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load corners from a file
################################################################################
def load_corners(inputfile) :
    # input:
    #    inputfile - path of the file containing corner detection output
    # return:
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading {} corners'.format(nc))
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in loading corners from \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--image', type = str, default = 'grid1.jpg',
                        help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1,
                        help = 'sigma value for Gaussain filter (default = 1.0)')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6,
                        help = 'threshold value for corner detection (default = 1e6)')
    parser.add_argument('-o', '--output', type = str, default = 'output.txt',
                        help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : {}'.format(args.image))
    print('sigma      : {:.2f}'.format(args.sigma))
    print('threshold  : {:.2e}'.format(args.threshold))
    print('output file: {}'.format(args.output))
    print('------------------------------')

    # load the image
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()
    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)

    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()
    # plt.imshow(np.float32(smooth2D(img_gray,2)), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('{} corners detected...'.format(len(corners)))
    show_corners(img_color, corners)

    # save corners to a file
    if args.output :
        save_corners(args.output, corners)
        print('corners saved to \'{}\'...'.format(args.output))

if __name__ == '__main__' :
    main()
