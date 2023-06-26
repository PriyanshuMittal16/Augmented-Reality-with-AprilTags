import numpy as np

def est_Pw(s):
    """
    Estimate the world coordinates of the April tag corners, assuming the world origin
    is at the center of the tag, and that the xy plane is in the plane of the April
    tag with the z axis in the tag's facing direction. See world_setup.jpg for details.
    Input:
        s: side length of the April tag

    Returns:
        Pw: 4x3 numpy array describing the world coordinates of the April tag corners
            in the order of a, b, c, d for row order. See world_setup.jpg for details.

    """

    ##### STUDENT CODE START #####
    "Let the corners be A,B,C,D starting from top left. Half of side s=r"
    r=s/2
    D=np.array([[-r],[r],[0]])
    C=np.array([[r],[r],[0]])
    B=np.array([[r],[-r],[0]])
    A=np.array([[-r],[-r],[0]])
    A=A.T
    B=B.T
    C=C.T
    D=D.T
    # print(np.shape(C))
    # print(Pw)
    

    Pw=np.vstack((A,B,C,D))
    # print(Pw)
    ##### STUDENT CODE END #####

    return Pw

if __name__ == "__main__":
    s=1
    est_Pw(s)
