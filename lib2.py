import numpy as np

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
