import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

#face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")

def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def mycamshift(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x, pt_y = c + w / 2, r + h / 2
    output.write("%d,%d,%d\n" % (0, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c, r, w, h)

    # calculate the HSV histogram in the window
    roi_hist = hsv_histogram_for_window(frame, (c, r, w, h)) # this is provided for you

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while(True):
        ret, frame = v.read() # read another frame
        if ret == False:
            break

        #change colorspace to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #calculating backprojection
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        #meanshift to get new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        #drawing it on image
        pts = cv2.boxPoints(ret)

        pt_x = (pts[0][0] + pts[2][0]) / 2
        pt_y = (pts[0][1] + pts[2][1]) / 2

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, pt_x, pt_y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


def myparticle(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x, pt_y = c + w / 2, r + h / 2
    output.write("%d,%d,%d\n" % (0, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    def particleevaluator(back_proj, particle):
        return back_proj[particle[1], particle[0]]

    # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
    # c,r,w,h: obtain using detect_one_face()
    n_particles = 300
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    #change colorspace to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #calculate back projection
    hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    init_pos = np.array([c + w / 2.0, r + h / 2.0], int) # initial position to center of detected face
    particles = np.ones((n_particles, 2), int) * init_pos # init particles to center position
    f0 = particleevaluator(hist_bp, init_pos) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform at start

    while(True):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        stepsize = 20

        # Particle motion model: uniform step (TODO: find a better motion model)
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1], frame.shape[0])) - 1).astype(int)

        f = particleevaluator(hist_bp, particles.T) # evaluate all particles
        weights = np.float32(f.clip(1))             # weight the histogram response
        weights /= np.sum(weights)                  # normalize the weights
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, pos[0], pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


def mykalman(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt_x, pt_y = c + w / 2, r + h / 2
    output.write("%d,%d,%d\n" % (frameCounter, pt_x, pt_y)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # initialize the tracker
    state = np.array([c + w / 2,r + h / 2, 0 ,0], dtype='float64') # initial position
    kalman = cv2.KalmanFilter(4,2,0) # 4 state/hidden, 2 measurement, 0 control
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],  # a rudimentary constant speed model:
                                        [0., 1., 0., .1],  # x_t+1 = x_t + v_t
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1 * np.eye(2, 4)      # you can tweak these to make the tracker
    kalman.processNoiseCov = 1e-2 * np.eye(4, 4)      # respond faster to change and be less smooth
    kalman.measurementNoiseCov = 1e-2 * np.eye(2, 2)
    kalman.errorCovPost = 1 * np.eye(4, 4)
    kalman.statePost = state

    #for every frame onwards predict and measure
    while(True):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        c, r, w, h = detect_one_face(frame)
        measurement_valid = not (c == 0 and r == 0 and w == 0 and h == 0)

        prediction = kalman.predict()

        pt_x, pt_y = prediction[0][0], prediction[1][0]

        #we get measurement

        if measurement_valid: # e.g. face found
            measurement = np.array([c + w / 2, r + h / 2], dtype='float64')
            posterior = kalman.correct(measurement)
            pt_x, pt_y = posterior[0], posterior[1]

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter, pt_x, pt_y)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()


def myof(v, file_name):
    output_name = sys.argv[3] + file_name
    output = open(output_name, "w")

    frameCounter = 0
    # read first frame
    ret, frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c, r, w, h = detect_one_face(frame)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.3,
                          minDistance=4,
                          blockSize=3)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(20, 20),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)
    x = int(c + w / 2)
    y = int(r + h / 2)
    mask[y - 20: y + 20, x - 20:x + 20] = 255

    prev_pts = cv2.goodFeaturesToTrack(prev_frame, mask=mask, **feature_params)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, (c + w / 2), (r + h / 2)))  # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    while (1):
        ret, frame = v.read()
        if ret == False:
            break

        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None, **lk_params)
        # Select good points
        good_new = next_pts[st == 1]
        # draw the tracks
        x, y = 0, 0
        for a, b in good_new:
            x += a
            y += b
            
        x /= len(good_new)
        y /= len(good_new)

        prev_frame = next_frame.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        output.write("%d,%d,%d\n" % (frameCounter, int(x), int(y)))  # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        # Now update the previous frame and previous points

    output.close()


if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        mycamshift(video, "output_camshift.txt")
    elif (question_number == 2):
        myparticle(video, "output_particle.txt")
    elif (question_number == 3):
        mykalman(video, "output_kalman.txt")
    elif (question_number == 4):
        myof(video, "output_of.txt")