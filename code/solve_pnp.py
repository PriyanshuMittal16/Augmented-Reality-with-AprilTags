from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    Pwu=np.zeros((4,2))
    # Pwu[0][0]=Pw[0][0]/Pw[0][2]
    # Pwu[0][1]=Pw[0][1]/Pw[0][2]
    # Pwu[1][0]=Pw[1][0]/Pw[1][2]
    # Pwu[1][1]=Pw[1][1]/Pw[1][2]
    # Pwu[2][0]=Pw[2][0]/Pw[2][2]
    # Pwu[2][1]=Pw[2][1]/Pw[2][2]
    Pwu[0][0]=Pw[0][0]
    Pwu[0][1]=Pw[0][1]
    Pwu[1][0]=Pw[1][0]
    Pwu[1][1]=Pw[1][1]
    Pwu[2][0]=Pw[2][0]
    Pwu[2][1]=Pw[2][1]
    Pwu[3][0]=Pw[3][0]
    Pwu[3][1]=Pw[3][1]
  
    
    H = est_homography(Pwu,Pc)
    H=H/H[2][2]
    
    K_inv=np.linalg.inv(K)
    H_dash=np.matmul(K_inv, H)
    
    a=H_dash[:,0]
    b=H_dash[:,1]
    c=np.cross(a,b)
    
    abc=np.vstack((a,b,c))
    
    abcT=np.transpose(abc)

    U, S , V = np.linalg.svd(abcT) #did from svd mnanual

    UV=np.matmul(U,V)
    det=np.linalg.det(UV)
    m=U.shape[1]
    identity=np.eye(m)
    identity[2][2]=det 
    Right=np.matmul(identity, V)
    Left=np.matmul(U,Right)
    R=np.transpose(Left)

    norm=np.linalg.norm(H_dash[:,0])
    th=H_dash[:,2]/norm
    t=-np.matmul(R,th)
    # print (R)
    return R, t

if __name__ == "__main__":

    Pc=np.array([[0,1],[1,1],[1,0],[0,0]])
    Pw=np.array([[3,1,5],[5,1,6],[8,3,9],[7,8,1]])
    x= PnP(Pc,Pw)

