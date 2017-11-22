import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import sys

def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("[Input_Marking]")
   print("Path to the input marking")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=20)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def RMSD(target, master):
    # Note: grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1/2.0)

        return total_diff;

def calculate_mask_and_show():
    fgSegment, bgSegment = find_superpixels_under_marking(whiteImg, segment1)
    cumHistFg = cumulative_histogram_for_superpixels(fgSegment, colorHist1)
    cumHistBg = cumulative_histogram_for_superpixels(bgSegment, colorHist1)
    res = do_graph_cut((cumHistFg, cumHistBg), (fgSegment, bgSegment), norm_hist,
                       neighbourVertices1)
    mask = np.where(res[segment1], 255, 0).astype('uint8')
    cv2.imshow("Mask", mask)

def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, mode, redLine, blueLine

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        if mode == True:
            redLine = True
        else:
            blueLine = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(whiteImg, (ix, iy), (x, y), (255, 0, 0), 4)
                cv2.line(pixel_image, (ix, iy), (x, y), (255, 0, 0), 4)

            else:
                cv2.line(whiteImg, (ix, iy), (x, y), (0, 0, 255), 4)
                cv2.line(pixel_image, (ix, iy), (x, y), (0, 0, 255), 4)

        # cv2.imshow('canvas', whiteImg)
        cv2.imshow('astronaut', pixel_image)
        if blueLine is True and redLine is True:
            calculate_mask_and_show()
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # cv2.imshow('canvas', whiteImg)
        cv2.imshow('astronaut', pixel_image)
        if redLine is True and blueLine is True:
            calculate_mask_and_show()


if __name__ == '__main__':
   
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    mainImg = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    whiteImg = np.ones((mainImg.shape[0], mainImg.shape[1], 3), np.uint8) * 255

    center1, colorHist1, segment1, neighbourVertices1 = superpixels_histograms_neighbors(mainImg)
    norm_hist = normalize_histograms(colorHist1)

    drawing = False  # true if mouse is pressed
    mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
    ix, iy = -1, -1
    redline = False
    blueLine = False
    
    pixel_image = mainImg
    cv2.namedWindow('astronaut')
    cv2.setMouseCallback('astronaut', draw_line)
    cv2.imshow('astronaut', pixel_image)
    
    while 1:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break

    cv2.destroyAllWindows()

