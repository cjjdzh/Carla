import cv2
import matplotlib as plt
import numpy as np
def roi(image):
    bottom_padding = 0 # Front bumper compensation
    height = image.shape[0]
    width = image.shape[1]
    bottom_left = [width*0.25, height-bottom_padding]
    bottom_right = [width*0.75, height-bottom_padding]
    # top_left = [width*0.20, height*0.65]
    # top_right = [width*0.85, height*0.65]
    top_left = [width * 0.35, height * 0.65]
    top_right = [width*0.65, height*0.65]
    vertices = [np.array([bottom_left, bottom_right, top_left,top_right], dtype=np.int32)]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, vertices[0]

def perspective_transform(image, vert):
    DEBUG = False
    pts2 = [[0, 240], [640, 240], [0, 0], [640, 0]]

    M = cv2.getPerspectiveTransform(np.float32(vert), np.float32(pts2))
    M_inv = cv2.getPerspectiveTransform(np.float32(pts2), np.float32(vert))
    dst = cv2.warpPerspective(image, M, (640, 240))

    if DEBUG:
        rcon = cv2.warpPerspective(dst, M_inv, (image.shape[1], image.shape[0]))
        f = plt.figure(figsize=(20, 6))
        f.add_subplot(131)
        plt.title("original")
        for vertex, color in zip(vert, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
            image = cv2.circle(image, tuple(vertex), 50, color, thickness=10)
            rcon = cv2.circle(rcon, tuple(vertex), 50, color, thickness=10)
        plt.imshow(image, cmap='gray')
        f.add_subplot(132)
        plt.title("transform")
        plt.imshow(dst, cmap='gray')
        f.add_subplot(133)
        plt.title("reconstruction")
        plt.imshow(rcon, cmap='gray')
        plt.show()

    return dst, M_inv

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10))
def hist_eq(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return equalized

def grayscale(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return gray

def blur(image):
  blur_image = cv2.GaussianBlur(image, (3,3),0)
  return blur_image

def canny(image):
  canny_image = cv2.Canny(image, 100, 150)
  return canny_image

def shadow_removal(image):
    DEBUG = False
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vectorized = np.float32(gray_image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)

    dark_centroid = np.argmin([np.linalg.norm(center[0]), np.linalg.norm(center[1])])

    diff = center[1 - dark_centroid] - center[dark_centroid]
    res = center[label.flatten()]
    res[label.flatten() == dark_centroid] = 255
    res[label.flatten() != dark_centroid] = 0
    result_image = res.reshape((gray_image.shape))

    blurred_img = cv2.GaussianBlur(image, (21, 21), 0)

    shadow_edge = canny(result_image)

    out = np.where(shadow_edge == [255, 255, 255], image, blurred_img)

    sobelx = cv2.Sobel(grayscale(out), cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Scale result to 0-255
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    out = cv2.inRange(scaled_sobel, 30, 256)

    if DEBUG:
        f = plt.figure(figsize=(20, 6))
        f.add_subplot(141)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.imshow(blurred_img)

        f.add_subplot(142)
        plt.imshow(result_image, cmap='gray')

        f.add_subplot(143)
        plt.imshow(shadow_edge, cmap='gray')
        plt.title("shad edge")

        f.add_subplot(144)
        plt.imshow(out, cmap='gray')
        plt.title("proposed soble")

    return result_image

def preprocess(image):
    DEBUG = False
    norm_image = np.zeros_like(image)
    norm_image = hist_eq(image)
    gray_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)

    shadow_mask = shadow_removal(image)

    # Detect pixels that are white in the grayscale image
    white_mask = cv2.inRange(norm_image, (200, 200, 200), (255, 255, 255))
    # white_mask = cv2.inRange(norm_image, (160,160,160), (255,255,255))

    # lenient_white = cv2.inRange(norm_image, (140,140,140), (255,255,255))
    # white_mask = cv2.bitwise_or(white_mask, cv2.bitwise_and(lenient_white, shadow_mask))

    # Detect yellow
    hsv_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2HSV)
    lower = np.array([19, 50, 0], dtype="uint8")
    upper = np.array([60, 255, 255], dtype="uint8")
    yellow_mask = cv2.inRange(hsv_image, lower, upper)

    # Combine all pixels detected above
    binary_wy = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Scale result to 0-255
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sobel_mask = cv2.inRange(scaled_sobel, 30, 256)

    output = cv2.bitwise_or(binary_wy, sobel_mask)
    output = cv2.medianBlur(output, 5)

    if DEBUG:
        f = plt.figure(figsize=(20, 6))
        f.add_subplot(231)
        plt.title("original")
        plt.imshow(cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB))

        f.add_subplot(232)
        plt.title("whites")
        plt.imshow(white_mask, cmap='gray')

        f.add_subplot(233)
        plt.title("yellows")
        plt.imshow(yellow_mask, cmap='gray')

        f.add_subplot(234)
        plt.title("w + y")
        plt.imshow(binary_wy, cmap='gray')

        f.add_subplot(235)
        plt.title("sobel")
        plt.imshow(sobel_mask, cmap='gray')

        f.add_subplot(236)
        plt.title("output")
        plt.imshow(output, cmap='gray')

    return output


def find_lane_pixels_using_histogram(binary_warped):
    DEBUG = False
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windowsf
    nwindows = 10
    # Set the width of the windows +/- margin
    #margin = image.shape[1] // 10
    margin= 720//10
    # Set minimum number of pixels found to recenter window
    minpix = 30

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    if DEBUG:
        slate = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if DEBUG:
            slate = cv2.rectangle(slate, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 255), 3)
            slate = cv2.rectangle(slate, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 255), 3)
            slate = cv2.circle(slate, (leftx_current, (win_y_low + win_y_high) // 2), 1, (0, 255, 0), 3)
            slate = cv2.circle(slate, (rightx_current, (win_y_low + win_y_high) // 2), 1, (0, 255, 0), 3)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if DEBUG:
        plt.figure()
        plt.bar(range(binary_warped.shape[1]), histogram)
        plt.imshow(slate)
    return leftx, lefty, rightx, righty


def fit_poly(binary_warped, leftx, lefty, rightx, righty):
    DEBUG = False
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty):
    DEBUG = False
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    points_left = np.vstack((left_fitx, ploty)).astype(np.int32)
    points_right = np.flip(np.vstack((right_fitx, ploty)).astype(np.int32), 1)

    for pt in points_left.T:
        cv2.circle(window_img, tuple(pt), 3, (255, 0, 0), 3)

    for pt in points_right.T:
        cv2.circle(window_img, tuple(pt), 3, (0, 0, 255), 3)

    points = np.hstack((points_left, points_right)).astype(np.int32).T

    cv2.fillPoly(window_img, [points], color=[0, 255, 0])

    result = cv2.addWeighted(out_img, 0.6, window_img, 0.4, 0)
    if DEBUG:
        f = plt.figure()
        f.add_subplot(121)
        plt.imshow(window_img)
        f.add_subplot(122)
        plt.imshow(out_img)
    ## End visualization steps ##
    return window_img


def find_street_lanes(image, prev_left_fit=None, prev_right_fit=None):
  DEBUG = False
  roi_image, roi_vert = roi(image)
  birds, m_inv = perspective_transform(image, roi_vert)
  binary_warped = preprocess(birds) // 255

  # if prev_left_fit is None or prev_right_fit is None:
  leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)

  # else:
  #   leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped, prev_left_fit, prev_right_fit)

  left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)
  painted = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)
  dewarped = cv2.warpPerspective(painted, m_inv, (image.shape[1], image.shape[0]),  flags=cv2.INTER_LINEAR)
  result = np.zeros_like(image)
  result = cv2.addWeighted(image, 0.7, dewarped, 0.3, 0)
  result[dewarped == 0] = image[dewarped == 0]

  if DEBUG:
    plt.figure(figsize=(20,6))
    plt.subplot(121)
    plt.title("birds eye view")
    plt.imshow(birds)

    plt.subplot(122)
    plt.title("binary")
    plt.imshow(binary_warped)

  return left_fit, right_fit, result
