import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels

        Rcw=Rwc.T
        Tcw=-Rcw*Twc
        Remove 1 column and take its inverse
    """
     # ##### STUDENT CODE START #####

    m=pixels.shape[0]
    Pw=np.zeros((m,3))
    singular=np.ones((m,1))
    pix1=np.column_stack((pixels,singular))

    lambdas=1
    R_cw=np.linalg.inv(R_wc)
    RTT=np.matmul(np.transpose(R_wc),t_wc)


    RTT=-RTT.reshape(3,1)
    R_n=R_cw[:,:-1]

    RT=np.column_stack((R_n, RTT))

    H=np.matmul(K, RT)
    H_dash=np.linalg.inv(H)
    i=0

    while i<m:
        pix1T=np.transpose(pix1[i,:])
        W=np.matmul(H_dash,pix1T)
        Pw[i,:]=np.transpose(W)
        Pw[i,:]=Pw[i,:]/Pw[i,-1]
        Pw[i,-1]=0
        i=i+1

    # ##### STUDENT CODE END #####
    return Pw

