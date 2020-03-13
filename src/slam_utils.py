import numpy as np
from numpy.linalg import norm, inv, pinv

def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XXX_sync_KLT.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: IMU measurements in IMU frame
            with shape 3*t
        rotational_velocity: IMU measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            [fx  0 cx
            0 fy cy
            0  0  1]
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
            close to 
            [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
            with shape 4*4
    '''
    with np.load(file_name) as data:
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
    return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu

def hat_map(vec):
    '''
        Transform vec in R^3 to a skew-symmetric vec_hat in R^3x3
        
        Input:
            vec - a 3x1 vector
        Output:
            vec_hat - a 3x3 matrix
    '''
    vec_hat = np.zeros((3,3))
    vec_hat[2,1] = vec[0]
    vec_hat[1,2] = -vec[0]
    vec_hat[2,0] = -vec[1]
    vec_hat[0,2] = vec[1]
    vec_hat[0,1] = -vec[2]
    vec_hat[1,0] = vec[2]
    return vec_hat

def adjoint(p, theta):
    '''
        Map u=[p,theta]^T in R^6 to se(3) in 4x4
        
        Input:
            p     - a 3x1 vector
            theta - a 3x1 vector
        Output:
            u_adjoint - a 6x6 matrix in SE(3)
    ''' 
    p_hat     = hat_map(p)
    theta_hat = hat_map(theta)
    u_adjoint = np.zeros((6,6))
    u_adjoint[:3,:3] = theta_hat
    u_adjoint[3:,3:] = theta_hat
    u_adjoint[:3,3:] = p_hat
    return u_adjoint

def twist(p, theta):
    '''
        Map u=[p,theta]^T in R^6 to the adjoint of SE(3) in 6x6
        
        Input:
            p     - a 3x1 vector
            theta - a 3x1 vector
        Output:
            twist - a 4x4 matrix in se(3)
    '''
    twist        = np.zeros((4,4))
    theta_hat    = hat_map(theta)
    twist[:3,:3] = theta_hat
    twist[:3,3]  = p
    return twist

def rodrigues_3(p, theta):
    '''
        u = [p,theta]^T
        Rodrigues formula for u where u is 6x1
    '''
    u  = twist(p,theta) # get u in SE(3)
    u2 = np.dot(u, u)   # u^2
    u3 = np.dot(u, u2)  # u^3
    u2_coeff = (1-np.cos(norm(theta)))/(norm(theta)**2)
    u3_coeff = (norm(theta)-np.sin(norm(theta)))/(np.power(norm(theta),3))
    
    T = np.eye(4) + u + u2_coeff*u2 + u3_coeff*u3
    return T

def approx_rodrigues_3(p, theta):
    '''
        u = [v,w]^T
        Approximate Rodrigues formula for u where u is 6x1 to avoid nan
    '''
    u  = twist(p,theta) # get u in SE(3)
    T = np.eye(4) + u
    return T

def rodrigues_6(p, theta):
    '''
        Rodrigues formula for u where u is 6x6
    '''
    u = adjoint(p, theta) # get u in adjoint of SE(3)
    u2 = np.dot(u, u)     # u^2
    u3 = np.dot(u, u2)    # u^3
    u4 = np.dot(u, u3)    # u^4
    u_coeff  = (3*np.sin(norm(theta)) - norm(theta)*np.cos(norm(theta))) / (2 * norm(theta))
    u2_coeff = (4 - norm(theta)*np.sin(norm(theta)) - 4*np.cos(norm(theta))) / (2 * norm(theta)**2)
    u3_coeff = (np.sin(norm(theta)) - norm(theta)*np.cos(norm(theta))) / (2 * np.power(norm(theta),3))
    u4_coeff = (2 - norm(theta)*np.sin(norm(theta)) - 2*np.cos(norm(theta))) / (2 * np.power(norm(theta),4))
    
    T = np.eye(6) + u_coeff*u + u2_coeff*u2 + u3_coeff*u3 + u4_coeff*u4
    return T
    
def mu_predict(mu, v, w, dt):
    '''
        EKF mean prediction
        
        Input:
            mu - current mean
            v  - current linear velocity
            w  - current rotational velocity
            dt - time interval
        Outputs:
            mu_pred - predicted mean
    '''
    p       = -dt * v
    theta   = -dt * w
    mu_pred = np.dot(rodrigues_3(p, theta), mu)
    return mu_pred

def cov_predict(cov, v, w, dt, noise):
    '''
        EKF covariance prediction
        
        Input:
            cov   - current covariance
            v     - current linear velocity
            w     - current rotational velocity
            dt    - time interval
            niose - motion noise covariance
        Outputs:
            cov_pred - predicted covariance
    '''
    p        = -dt * v
    theta    = -dt * w
    cov_pred = np.dot(rodrigues_6(p, theta), cov)
    cov_pred = np.dot(cov_pred, rodrigues_6(p, theta).T)
    cov_pred = cov_pred + noise
    return cov_pred

def get_calibration(K, b):
    '''
        Get calibration matrix M from K and b
        
        Input:
            K - camera calibration matrix
            b - stereo baseline
        Output:
            M - stereo camera calibration matrix
    '''
    M   = np.vstack([K[:2], K[:2]])
    arr = np.array([0, 0, -K[0,0]*float(b), 0]).reshape((4,1))
    M   = np.hstack([M, arr])
    return M

def projection(q):
    '''
        Get the projection of a vector in R^4
        
        Input:
            q  - a vector in R^4
        Output:
            pi - corresponding projection
    '''
    pi = q / q[2]
    return pi

def d_projection(q):
    '''
        Take a R^4 vector and return the derivative of its projection function
        
        Input:
            q  - a vector in R^4
        Output:
            dq - corresponding derivative of the projection function, size 4x4
    '''
    dq = np.zeros((4,4))
    dq[0,0] = 1
    dq[1,1] = 1
    dq[0,2] = -q[0]/q[2]
    dq[1,2] = -q[1]/q[2]
    dq[3,2] = -q[3]/q[2]
    dq[3,3] = 1
    dq = dq / q[2]
    return dq

def pixel_to_world(p, i_T_w, o_T_i, K, b):
    '''
        Get homogeneous coordinates xyz in world frame from left right pixels
        
        Input:
            p     - left right pixels, size 1x4
            i_T_w - current pose of IMU, size 4x4
            o_T_i - imu to optical frame rotation, size 3x3
            K     - camera calibration matrix
            b     - stereo baseline
        Output:
            m_w   - homogeneous coordinates
    '''
    uL, vL, uR, vR = p
    fsu = K[0,0]
    fsv = K[1,1]
    cu  = K[0,2]
    cv  = K[1,2]
    z   = (fsu*b) / (uL-uR)
    x   = z * (uL-cu) / fsu
    y   = z * (vL-cv) / fsv
    m_o = np.array([x,y,z,1]).reshape([4,1])
    m_i = np.dot(inv(o_T_i), m_o)
    m_w = np.dot(inv(i_T_w), m_i)
    return m_w

def get_Ht(H_list, num_feature, isSLAM=False):
    '''
        Get the model Jacobian
    '''
    if isSLAM:
        Nt = len(H_list)
        Ht = np.zeros([4*Nt, 3*num_feature+6])
        for i in range(Nt):
            j = H_list[i][0]      # landmark index
            H_obs  = H_list[i][1]
            H_pose = H_list[i][2]
            Ht[i*4:(i+1)*4, 3*j:3*(j+1)] = H_obs
            Ht[i*4:(i+1)*4, -6:] = H_pose
    else:
        Nt = len(H_list)
        Ht = np.zeros([4*Nt, 3*num_feature])
        for i in range(Nt):
            j = H_list[i][0]      # landmark index
            H = H_list[i][1]      # current Hij
            Ht[i*4:(i+1)*4,3*(j):3*(j+1)] = H
    return Ht

def get_Kt(cov, Ht, v):
    '''
        Get the Kalman gain
    '''
    V_noise  = np.eye(Ht.shape[0]) * v
    inv_term = np.dot(Ht, np.dot(cov, Ht.T)) + V_noise
    Kt       = np.dot(np.dot(cov, Ht.T), inv(inv_term))
    return Kt

def circle(m):
    '''
        circle operator
        
        Input:
            m - a vector in R^4, [x,y,z,1]
        Output:
            result - a matrix of size 4x6
    '''
    s      = m[:3]
    s_hat  = hat_map(s)
    result = np.hstack((np.eye(3), -s_hat))
    result = np.vstack((result, np.zeros((1,6))))
    return result