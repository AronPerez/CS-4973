""" Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import scipy.signal 
import cv2


def getImageCorners(image):
    """Return the x, y coordinates for the four corners bounding the input
    image and return them in the shape expected by the cv2.perspectiveTransform
    function. (The boundaries should completely encompass the image.)

    Parameters
    ----------
    image : numpy.ndarray
        Input can be a grayscale or color image

    Returns
    -------
    numpy.ndarray(dtype=np.float32)
        Array of shape (4, 1, 2).  The precision of the output is required
        for compatibility with the cv2.warpPerspective function.

    Notes
    -----
        (1) Review the documentation for cv2.perspectiveTransform (which will
        be used on the output of this function) to see the reason for the
        unintuitive shape of the output array.

        (2) When storing your corners, they must be in (X, Y) order -- keep
        this in mind and make SURE you get it right.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    width, length = image.shape[:2]
    corners = np.array([[[0,0]],[[length,0]],[[0,width]],[[length,width]]], dtype = np.float32)
    #print corners
    return corners


def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    Notes
    -----
        (1) You will not be graded for this function.
    """
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE.
    for i, match in enumerate(matches):
        image_1_points[i] = image_1_kp[match.queryIdx].pt
        image_2_points[i] = image_2_kp[match.trainIdx].pt
    M_hom, inliers = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, 5.0)
    #print M_hom
    return M_hom


def getBoundingCorners(corners_1, corners_2, homography):
    """Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Follow these steps:

        1. Use the homography to transform the perspective of the corners from
           image 1 (but NOT image 2) to get the location of the warped
           image corners.

        2. Get the boundaries in each dimension of the enclosing rectangle by
           finding the minimum x, maximum x, minimum y, and maximum y.

    Parameters
    ----------
    corners_1 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 1

    corners_2 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 2

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_min, y_min) -- the coordinates of the
        top left corner of the bounding rectangle of a canvas large enough to
        fit both images (leave them as floats)

    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_max, y_max) -- the coordinates of the
        bottom right corner of the bounding rectangle of a canvas large enough
        to fit both images (leave them as floats)

    Notes
    -----
        (1) The inputs may be either color or grayscale, but they will never
        be mixed; both images will either be color, or both will be grayscale.

        (2) Python functions can return multiple values by listing them
        separated by commas.

        Ex.
            def foo():
                return [], [], []
    """
    # WRITE YOUR CODE HERE
    corners_1 = cv2.perspectiveTransform(corners_1, homography)
    Cornersjoined = np.concatenate((corners_1, corners_2))
    left = Cornersjoined[:, :, :1]
    right = Cornersjoined[:, :, 1:]
    x_min = np.min(left)
    y_min = np.min(right)
    x_max = np.max(left)
    y_max = np.max(right)
    min_xy = np.array([x_min, y_min], dtype=np.float64)
    max_xy = np.array([x_max, y_max], dtype=np.float64)
    #print min_xy
    return min_xy, max_xy


def warpCanvas(image, homography, min_xy, max_xy):
    """Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Follow these steps:

        1. Create a translation matrix (numpy.ndarray) that will shift
           the image by x_min and y_min. This looks like this:

            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]]

        2. Compute the dot product of your translation matrix and the
           homography in order to obtain the homography matrix with a
           translation.

        NOTE: Matrix multiplication (dot product) is not the same thing
              as the * operator (which performs element-wise multiplication).
              See Numpy documentation for details.

        3. Call cv2.warpPerspective() and pass in image 1, the combined
           translation/homography transform matrix, and a vector describing
           the dimensions of a canvas that will fit both images.

        NOTE: cv2.warpPerspective() is touchy about the type of the output
              shape argument, which should be an integer.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between two sequential
        images in a panorama sequence

    min_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the top left corner of a
        canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    max_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the bottom right corner of
        a canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    Returns
    -------
    numpy.ndarray(dtype=image.dtype)
        An array containing the warped input image embedded in a canvas
        large enough to join with the next image in the panorama; the output
        type should match the input type (following the convention of
        cv2.warpPerspective)

    Notes
    -----
        (1) You must explain the reason for multiplying x_min and y_min
        by negative 1 in your writeup.
    """
    # canvas_size properly encodes the size parameter for cv2.warpPerspective,
    # which requires a tuple of ints to specify size, or else it may throw
    # a warning/error, or fail silently
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
    # WRITE YOUR CODE HERE
    x_min=min_xy[0]
    y_min=min_xy[1]
    Translation = np.array([[1, 0, -x_min], \
                            [0, 1, -y_min], \
                            [0, 0, 1]])
    TranslationHomography = np.dot(Translation, homography)
    output_image = cv2.warpPerspective(image, TranslationHomography, canvas_size)
    
    return output_image

def generatingKernel(a):
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)

def reduce_layer(image, kernel=generatingKernel(0.4)):
    image = np.float64(image)
    image = cv2.copyMakeBorder(image,5,5,5,5,cv2.BORDER_REFLECT)
    image = scipy.signal.convolve2d(image, np.float64(kernel), mode='same')
    image = image[5:-5, 5:-5]
    image = image[::2, ::2]
    return np.float64(image)

def expand_layer(image, kernel=generatingKernel(0.4)):
    (width, height) = image.shape
    newImage = np.zeros([width*2,height*2])
    newImage[::2, ::2] = image
    newImage = cv2.copyMakeBorder(newImage,5,5,5,5,cv2.BORDER_REFLECT)
    newImage = scipy.signal.convolve2d(newImage, generatingKernel(0.4), 'same') * 4
    newImage = newImage[5:-5, 5:-5]
    return np.float64(newImage)

def gaussPyramid(image, levels):
    image = np.float64(image)
    pyramidLevel = [image]
    for i in range(levels):
        pyramidLevel.append(np.float64(reduce_layer(pyramidLevel[i])))
    return pyramidLevel

def laplPyramid(gaussPyr):
    output = []
    for i in range(len(gaussPyr) - 1):
        (width, height) = gaussPyr[i].shape
        output.append(gaussPyr[i] - expand_layer(gaussPyr[i+1])[:width,:height])
    output.append(gaussPyr[len(gaussPyr) - 1])
    return output

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    blendLevel=[]
    for i in range(len(laplPyrWhite)):
        blendLevel.append(laplPyrBlack[i]*(1-gaussPyrMask[i])+laplPyrWhite[i]*gaussPyrMask[i])
    return blendLevel

def collapse(pyramid):
    newImage = pyramid[len(pyramid)-1]
    for i in range(len(pyramid)-2,-1,-1):
        (width, height) = pyramid[i].shape
        newImage = expand_layer(newImage)[:width,:height]+pyramid[i]
    return newImage

def viz_pyramid(stack, shape, norm=False):
    """Create a single image by vertically stacking the levels of a pyramid."""
    layers = [normalize(np.dstack(imgs)) if norm else np.clip(np.dstack(imgs), 0, 255) for imgs in zip(*stack)]
    stack = [cv2.resize(layer, shape, interpolation=3) for layer in layers]
    img = np.vstack(stack).astype(np.uint8)
    return img

def pyramidblending(black_image, white_image, mask, min_depth):
    black_image = np.atleast_3d(black_image).astype(np.float)
    white_image = np.atleast_3d(white_image).astype(np.float)
    mask_img = np.atleast_3d(mask).astype(np.float)
    
    shape = mask_img.shape[1::-1]
    min_size = min(black_image.shape[:2])
    depth = int(np.log2(min_size)) - min_depth
    
    gauss_pyr_mask = [gaussPyramid(ch, depth) for ch in np.rollaxis(mask_img, -1)]
    gauss_pyr_black = [gaussPyramid(ch, depth) for ch in np.rollaxis(black_image, -1)]
    gauss_pyr_white = [gaussPyramid(ch, depth) for ch in np.rollaxis(white_image, -1)]
    
    lapl_pyr_black = [laplPyramid(ch) for ch in gauss_pyr_black]
    lapl_pyr_white = [laplPyramid(ch) for ch in gauss_pyr_white]
    
    outpyr = [blend(*x) for x in zip(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)]
    outimg = [[collapse(x)] for x in outpyr]
    ans=viz_pyramid(outimg, shape)
    
    return ans

def blendImagePair(image_1, image_2, num_matches):
    """This function takes two images as input and fits them onto a single
    canvas by performing a homography warp on image_1 so that the keypoints
    in image_1 aligns with the matched keypoints in image_2.

    **************************************************************************

        You MUST replace the basic insertion blend provided here to earn
                         credit for this function.

       The most common implementation is to use alpha blending to take the
       average between the images for the pixels that overlap, but you are
                    encouraged to use other approaches.

           Be creative -- good blending is the primary way to earn
                  Above & Beyond credit on this assignment.

    **************************************************************************

    Parameters
    ----------
    image_1 : numpy.ndarray
        A grayscale or color image

    image_2 : numpy.ndarray
        A grayscale or color image

    num_matches : int
        The number of keypoint matches to find between the input images

    Returns:
    ----------
    numpy.ndarray
        An array containing both input images on a single canvas

    Notes
    -----
        (1) This function is not graded by the autograder. It will be scored
        manually by the TAs.

        (2) The inputs may be either color or grayscale, but they will never be
        mixed; both images will either be color, or both will be grayscale.

        (3) You can modify this function however you see fit -- e.g., change
        input parameters, return values, etc. -- to develop your blending
        process.
    """
    kp1, kp2, matches = findMatchesBetweenImages(
        image_1, image_2, num_matches)
    homography = findHomography(kp1, kp2, matches)
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
    output_image = warpCanvas(image_1, homography, min_xy, max_xy)
    # WRITE YOUR CODE HERE - REPLACE THIS WITH YOUR BLENDING CODE
    # Please note that the NUM_MATCHES in main function was chosen as 18 to produce my results.
    min_xy = min_xy.astype(np.int)
    #output_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],-min_xy[0]:-min_xy[0] + image_2.shape[1]]=cv2.addWeighted(output_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],-min_xy[0]:-min_xy[0] + image_2.shape[1]], 0.05, image_2, 0.95, 0)
    #return output_image
    output_image=output_image.astype(np.float)
    mask=np.zeros_like(output_image, dtype='float')
    edge=150
    #for i in range(1,edge):
        #mask[-min_xy[1]+i:-min_xy[1] + image_2.shape[0]-i,-min_xy[0]+i:-min_xy[0] + image_2.shape[1]-i]=np.tanh(i*1.0/edge/2)
    mask[-min_xy[1]+edge:-min_xy[1] + image_2.shape[0]-edge,-min_xy[0]+edge:-min_xy[0] + image_2.shape[1]-edge]=1
    #cv2.imwrite("mask.jpg", mask*255)
    canvasimage_2=np.zeros_like(output_image, dtype='float')
    canvasimage_2[-min_xy[1]:-min_xy[1] + image_2.shape[0],-min_xy[0]:-min_xy[0] + image_2.shape[1]]=image_2.astype(np.float)
    output=pyramidblending(output_image,canvasimage_2,mask, 4)
    return output
    # END OF FUNCTION
