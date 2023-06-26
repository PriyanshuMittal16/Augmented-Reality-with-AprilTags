import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    x00=X[0][0]
    x01=X[0][1]
    x10=X[1][0]
    x11=X[1][1]
    x20=X[2][0]
    x21=X[2][1]
    x30=X[3][0]
    x31=X[3][1]

    y00=Y[0][0]   
    y01=Y[0][1]
    y10=Y[1][0]
    y11=Y[1][1]
    y20=Y[2][0]
    y21=Y[2][1]
    y30=Y[3][0]
    y31=Y[3][1]

    ax=np.array([-x00, -x01, -1, 0, 0, 0, x00*y00, x01*y00, y00 ])
    ay=np.array([0,0,0, -x00, -x01, -1, x00*y01, x01*y01, y01])

    bx=np.array([-x10, -x11, -1, 0, 0, 0, x10*y10, x11*y10, y10 ])
    by=np.array([0,0,0, -x10, -x11, -1, x10*y11, x11*y11, y11])

    cx=np.array([-x20, -x21, -1, 0, 0, 0, x20*y20, x21*y20, y20 ])
    cy=np.array([0,0,0, -x20, -x21, -1, x20*y21, x21*y21, y21])

    dx=np.array([-x30, -x31, -1, 0, 0, 0, x30*y30, x31*y30, y30 ])
    dy=np.array([0,0,0, -x30, -x31, -1, x30*y31, x31*y31, y31])

    A = np.vstack((ax,ay,bx,by,cx,cy,dx,dy))
    
    [U, S , V] = np.linalg.svd(A) 

    H=V[-1].reshape(3,3)



    ##### STUDENT CODE END #####

    return H
