from math import pi
import numpy as np
import cv2
import lmfit

class Camera:
    def __init__(self, w, h, f, D, t, r):
        self.width = w
        self.height = h
        self.focal_length = f
        self.K = np.array([[f, 0, w / 2 - 0.5],
                           [0, f, h / 2.0 - 0.5],
                           [0, 0, 1]], dtype=np.float)
        self.t = np.array(t, dtype=float)
        self.r = np.array(r, dtype=float)
        self.P = self.create_proj_mat(self.K, self.t, self.r)
        self.D = np.array(D, dtype=float)

    @staticmethod
    def create_proj_mat(K, t, r):
        KA = np.concatenate((K, np.zeros((3, 1))), axis=1)
        R, _ = cv2.Rodrigues(r)
        T = np.eye(4, dtype=np.float)
        T[:3, :3] = R
        T[:3, 3] = t
        return np.matmul(KA, T)

class StereoCameras:
    def __init__(self, w, h, f, D, t_r, r_r, t_l=(0, 0, 0), r_l=(0, 0, 0)):
        self.cam_r = Camera(w, h, f, D, t_r, r_r)
        self.cam_l = Camera(w, h, f, D, t_l, r_l)

def objective_func(params, q_l, q_r, P_l, P_r):
    hom_Q = np.transpose(np.array([[params['x'].value, params['y'].value, params['z'].value, 1]], dtype=np.float))
    proj_q_l = np.matmul(P_l, hom_Q)
    proj_q_r = np.matmul(P_r, hom_Q)
    residual = []
    residual.append(q_l[0] - proj_q_l[0] / proj_q_l[2])
    residual.append(q_l[1] - proj_q_l[1] / proj_q_l[2])
    residual.append(q_r[0] - proj_q_r[0] / proj_q_r[2])
    residual.append(q_r[1] - proj_q_r[1] / proj_q_r[2])
    return residual

def define_stereo_cameras():
    # w, h, f = width, height, focal length
    # D = tuple of 5 distortion parameters
    # t_r, r_r = XYZ translatation and RPY rotation of right camera respective to left
    # return StereoCameras(w, h, f, D, t_r, r_r)

# def generate_points(x_range, y_range, z_range, n):
    # np.array of n xyz points within the ranges
    # return Qs

# def project_points_to_image_plane(Qs, cam):
    # Use cv2.projectPoints to project the 3D points to 2D
    # correct the shape of the output (squeeze and transpose)
    # return qs

# def add_noise_to_image_points(qs, sigma):
    # return qs + random Gaussian noise with standard deviation = sigma

# def triangulate_points(qs_l, qs_r, cams):
    # Use cv2.triangulatePoints to find 3D points
    # Un-homogenize the points and correct shape
    # return triangulated points

# def evaluate_euclidean_distance(Qs1, Qs2):
    # return L2-norm of difference between points

# def evaluate_reprojection_error(Qs, qs_l, qs_r, cams):
    # Project points to 2D and calculate the rms distance between the optimal and projected points
    # return rpe_l + rpe_r

# def optimize_points(Qs, qs_l, qs_r, cams):
    # Qs_opt = []
    # for q_l, q_r, Q_init in zip(qs_l, qs_r, Qs):
        # Use lmfit with the provided objective function to find a point minimizing the reprojection error
    # return np.array(Qs_opt, dtype=np.float)

def main():
    cams = define_stereo_cameras()
    # orig_Qs = generate_points((0.05, 0.1), (0.18, 0.22), (1.98, 2.02), 5)

    # qs_l = project_points_to_image_plane(orig_Qs, cams.cam_l)
    # qs_r = project_points_to_image_plane(orig_Qs, cams.cam_r)

    # noisy_qs_l = add_noise_to_image_points(qs_l, 1)
    # noisy_qs_r = add_noise_to_image_points(qs_r, 1)

    # triangulated_Qs = triangulate_points(qs_l, qs_r, cams)
    # noisy_triangulated_Qs = triangulate_points(noisy_qs_l, noisy_qs_r, cams)

    # triangulated_dist = evaluate_euclidean_distance(triangulated_Qs, orig_Qs)
    # noisy_dist = evaluate_euclidean_distance(noisy_triangulated_Qs, orig_Qs)

    # triangulated_rpe = evaluate_reprojection_error(triangulated_Qs, qs_l, qs_r, cams)
    # noisy_rpe = evaluate_reprojection_error(noisy_triangulated_Qs, qs_l, qs_r, cams)

    # optimal_Qs = optimize_points(noisy_triangulated_Qs, np.transpose(noisy_qs_l), np.transpose(noisy_qs_r), cams)
    # optimal_dist = evaluate_euclidean_distance(optimal_Qs, orig_Qs)
    # optimal_rpe = evaluate_reprojection_error(optimal_Qs, qs_l, qs_r, cams)

if __name__ == "__main__":
    main()
