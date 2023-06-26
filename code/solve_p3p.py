import numpy as np
import math as m

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    #DEBUGG SOLVE P3P BY INPUTTING PC AND PW FIRST. THERE ARE MANY ERRORS

    p1=Pw[0,:]
    p2=Pw[1,:]
    p3=Pw[2,:]
    p4=Pw[3,:]

    a=np.linalg.norm(p3-p4)
    b=np.linalg.norm(p4-p2)
    c=np.linalg.norm(p2-p3)
    f=(K[0][0]+K[1][1])/2
    u0=K[0][2]; v0=K[1][2]

    centeroff=np.array([u0,v0])
    K_inv=np.linalg.inv(K)


    Pc1=np.array([Pc[1,0], Pc[1,1],1])
    Pc2=np.array([Pc[2,0], Pc[2,1],1])
    Pc3=np.array([Pc[3,0], Pc[3,1],1])
    uv1=np.matmul(K_inv,Pc1)
    uv2=np.matmul(K_inv,Pc2)
    uv3=np.matmul(K_inv,Pc3)

    j1=uv1/np.linalg.norm(uv1)
    j2=uv2/np.linalg.norm(uv2)
    j3=uv3/np.linalg.norm(uv3)
  
    cosalpha=np.dot(j2,j3)
    cosbeta=np.dot(j1,j3)
    cosgamma=np.dot(j1,j2)

    acm2= ((a**2)-(c**2))/(b**2)
    acp2=((a**2)+(c**2))/(b**2)
    bc2=((b**2)-(c**2))/(b**2)
    ba2=((b**2)-(a**2))/(b**2)

    ab2=(a*a)/(b*b)
    cb2=(c*c)/(b*b)

    A4=((acm2-1)*(acm2-1))-4*cb2*cosalpha*cosalpha
    A3=4*((acm2*(1-acm2)*cosbeta)-((1-acp2)*cosalpha*cosgamma)+(2*cb2*cosalpha*cosalpha*cosbeta))
    A2=2*(((acm2*acm2)-1)+(2*acm2*acm2*cosbeta*cosbeta)+(2*bc2*cosalpha*cosalpha)-(4*acp2*cosalpha*cosbeta*cosgamma)+(2*ba2*cosgamma*cosgamma))
    A1=4*(-(acm2*(1+acm2)*cosbeta)+(2*ab2*cosgamma*cosgamma*cosbeta)-((1-acp2)*cosalpha*cosgamma))
    A0=((1+acm2)*(1+acm2)-(4*ab2*cosgamma*cosgamma))  

    coefficients=np.roots(np.array([A4,A3,A2,A1,A0]))

    r=[]
    s1=[]
    s2=[]
    s3=[]
    r=coefficients[np.isreal(coefficients)].real
    
    realv=np.array(r)

    k=realv.shape[0]
    u=[]
    lR=[]
    lt=[]
    temp1=np.zeros((k,3))
    temp2=np.zeros((k,1))
    for i in range (k):
        v=realv[i]
        u.append(((-1+acm2)*v**2-2*acm2*v*cosbeta+1+acm2)/(2*(cosgamma-v*cosalpha)))
        v1=m.sqrt(c**2/(u[i]**2+1-2*u[i]*cosgamma))
        s1.append(v1)

        v2=u[i]*s1[i]
        s2.append(v2)

        v3=r[i]*s1[i]
        s3.append(v3)

        pn1=s1[i]*j1
        pn2=s2[i]*j2 
        pn3=s3[i]*j3 

        Pc_3d=np.array([pn1.T,pn2.T,pn3.T])

        R,t = Procrustes(Pw[1:,:], Pc_3d)

        lR.append(R)
        lt.append(t)

        x1= np.matmul(R, Pw[0,:].T) 

        temp1[i]=np.matmul(K,x1+t)
        temp1[i]=temp1[i]/temp1[i,2]
        temp2[i]=np.linalg.norm(temp1[i, 0:2]-Pc[0,:])
    
    pointer=np.argmin(temp2)
    R=lR[pointer]
    t=lt[pointer]

    R=np.linalg.inv(R)
    t=-np.matmul(R,t)
    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    X_=np.mean(X,axis=0).T
    Y_=np.mean(Y,axis=0).T

    X=np.transpose(X-X_) 
    Y=np.transpose(Y-Y_)
    abt=np.matmul(Y,np.transpose(X))

    [U, S ,V] = np.linalg.svd(abt)
    us=U.shape[1]
    I=np.eye(us)
    duv=np.linalg.det(np.matmul(U,V))
    I[-1][-1]=duv

    R=np.matmul(U,np.matmul(I,V))
    t=Y_.T-np.matmul(R,X_.T) 
    
    return R, t

