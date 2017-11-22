**Computer Vision**

**CV2 Python**
Getting started with openCV and matplotlib. Reading, writing and simple arithmatic on image with openCV

**Panorama**
Goal is to create 2 panoramas:
1. Using homographies and perspective warping on a common plane (3 images).
2. Using cylindrical warping (many images).

**Filters convolution ImageBlending**
Filtering in the spatial domain as well as in the frequency domain.
Laplacian Blending using Image Pyramids is a very good intro to working and thinking in frequencies, and Deconvolution is a neat trick.

we will:

1. Perform Histogram Equalization on the given input image.
2. Perform Low-Pass, High-Pass and Deconvolution on the given input image.
3. Perform Laplacian Blending on the two input images (blend them together).

**Detection and Tracking**
Goal is to:
1. Detect the face in the first frame of the movie using pre-trained Viola-Jones detector.

2. Track the face throughout the movie using:
	a. CAMShift
	b. Particle Filter
	c. Face detector + Kalman Filter
 
3. Face Detector + Optical Flow tracker (uses OF tracker whenever the face detector fails).

**Segmentation**
Goal is to perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts:
Given an image and sparse markings for foreground and background
1. Calculate SLIC over image
2. Calculate color histograms for all superpixels
3. Calculate color histograms for FG and BG
4. Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
5. Run a graph-cut algorithm to get the final segmentation

**"tool.py"**
Lets Make it interactive:Let the user draw the markings, for every interaction step (mouse click, drag, etc.)

1. recalculate only the FG-BG histograms,
2. construct the graph and get a segmentation from the max-flow graph-cut,
3. show the result immediately to the user (should be fast enough).