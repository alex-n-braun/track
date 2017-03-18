import numpy as np

from sklearn.utils import shuffle

def split_n_shuffle(features, labels, split):
    feat_train=features[0:0]; label_train=np.array([])
    feat_test=features[0:0]; label_test=np.array([])
    for cls in range(2):
        #print(features)
        feat_cls=features[labels==cls]
        #print(feat_cls)
        n_cls = len(feat_cls)
        feat_cls_train=feat_cls[0:int(split*n_cls)]; feat_cls_test=feat_cls[int(split*n_cls):n_cls]
        #print(feat_cls_train)
        #print(feat_cls_test)
        label_cls_train=np.ones(len(feat_cls_train))*cls
        label_cls_test=np.ones(len(feat_cls_test))*cls
        feat_train=np.vstack((feat_train, feat_cls_train))
        feat_test=np.vstack((feat_test, feat_cls_test))
        label_train=np.hstack((label_train, label_cls_train))
        label_test=np.hstack((label_test, label_cls_test))
    feat_train, label_train=shuffle(feat_train, label_train)
    feat_test, label_test=shuffle(feat_test, label_test)
    return feat_train, feat_test, label_train, label_test

class Classifier():
    def __init__(self, clf, minDist=0):
        self.clf=clf
        self.minDist=minDist
        
    def predict(self, X):
        d=self.clf.decision_function(X)
        result=np.zeros(len(X))
        result[d>self.minDist]=1
        return result
    
    def score(self, X, y):
        p=self.predict(X)
        d=1-np.abs(p-y)
        return ((np.sum(d))/len(d))

def rectangle(polygon):
    xMin=polygon[0][1]; xMax=xMin
    yMin=polygon[0][0]; yMax=yMin
    for i in range(1, len(polygon)):
        xMin=xMin if xMin<=polygon[i][0] else polygon[i][0]
        xMax=xMax if xMax>=polygon[i][0] else polygon[i][0]
        yMin=yMin if yMin<=polygon[i][1] else polygon[i][1]
        yMax=yMax if yMax>=polygon[i][1] else polygon[i][1]
        
    return [[xMin, yMin],[xMax, yMax]]

# given a polygon (with exactly 4 corners!!!), cut a
# rectangular area from img that contains the complete polygon.
def region_of_interest(img, polygon):
    xMin=polygon[0][1]; xMax=xMin
    yMin=polygon[0][0]; yMax=yMin
    for i in range(1, len(polygon)):
        xMin=xMin if xMin<=polygon[i][0] else polygon[i][0]
        xMax=xMax if xMax>=polygon[i][0] else polygon[i][0]
        yMin=yMin if yMin<=polygon[i][1] else polygon[i][1]
        yMax=yMax if yMax>=polygon[i][1] else polygon[i][1]
    return img[yMin:yMax, xMin:xMax, :]

# using the 4 corners of polygon src, derive a perspective 
# scaling factor. For y at the lower border of the polygon,
# the scaling factor is 1.
def perspective_scale(y):
    src=[[326, 60], [411, 60], [812, 280], [50, 280]]
    l0=src[2][0]-src[3][0]
    l1=src[1][0]-src[0][0]
    y0=src[2][1]
    y1=src[0][1]
    a = (l1-l0)/(y1-y0)
    b = 0.5*(l1+l0-a*(y1+y0))
    return (a*y+b)/l0
    
def left_boundary(y, groundWindow=[[0,279],[240,50],[800,50],[1003,70], [1003,279]]):
    y0=groundWindow[0][1]
    y1=groundWindow[1][1]
    x0=groundWindow[0][0]
    x1=groundWindow[1][0]
    a=(x1-x0)/(y1-y0)
    b=0.5*(x1+x0-(x1-x0)/(y1-y0)*(y1+y0))
    result=int(a*y+b)
    return result if result>0 else 0

def right_boundary(y, groundWindow=[[0,279],[240,50],[800,50],[1003,70], [1003,279]]):
    y0=groundWindow[2][1]
    y1=groundWindow[3][1]
    x0=groundWindow[2][0]
    x1=groundWindow[3][0]
    a=(x1-x0)/(y1-y0)
    b=0.5*(x1+x0-(x1-x0)/(y1-y0)*(y1+y0))
    result=int(a*y+b)
    xMax=max(np.array(groundWindow)[:,0])
    return result if result<xMax else xMax

def top_boundary(groundWindow=[[0,279],[240,50],[800,50],[1003,70], [1003,279]]):
    return min(np.array(groundWindow)[:,1])

def bottom_boundary(groundWindow=[[0,279],[240,50],[800,50],[1003,70], [1003,279]]):
    return max(np.array(groundWindow)[:,1])

# Source of the code: Udacity, Self-Driving Car Engineer. Adapted to support x range as well.
# Define a single function that can extract features using hog sub-sampling and make predictions
from lib import convert_color
import cv2
import numpy as np
from lib import get_hog_features, bin_spatial
def find_cars2(img, xstart, xstop, ystart, ystop, scale, 
               svc, X_scaler, orient, pix_per_cell, cell_per_block, cells_per_step_x, cells_per_step_y, 
               spatial_size, hist_bins):
        
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step_x + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step_y + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    boxes=[]
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step_y
            xpos = xb*cells_per_step_x
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box=((xbox_left+xstart, ytop_draw+ystart), (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart))
                boxes.append(box)
                
    return boxes

def get_labeled_bboxes(labels):
    # Iterate through all detected cars
    bboxes=[]
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Add the box on the list
        bboxes.append(bbox)
    return bboxes

def draw_labeled_bboxes2(img, bboxes):
    # Iterate through all detected cars
    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 6)
    # Return the image
    return img

