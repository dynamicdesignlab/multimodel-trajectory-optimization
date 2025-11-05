import scipy.interpolate
import scipy.io as sio
import scipy.spatial
import numpy as np
import casadi
import matplotlib.pyplot as plt


class World:
    
    def __init__(self, filename, name, edge_buffer_m=0.0, diagnostic_plotting=True):

        self.params = {}
        self.params["filename"] = filename
        self.params["name"]     = name
        
        self.name = name
        self.data = sio.loadmat(filename, squeeze_me=True)
        del self.data["__header__"]
        del self.data["__version__"]
        del self.data["__globals__"]

        self.length_m  = self.data["length_m"]
        self.gps_xyz_ref_m = self.data["gpsXYZRef_m"]
        
        
        # Array to hold the local track attitude and associated derivatives
        attitude_ = np.zeros((9, len(self.data["s_m"])))   # Centerline s-coordinate
        attitude_[0, :]  = np.unwrap(self.data["psi_rad"]) # Road heading angle
        attitude_[1, :]  = self.data["psi_s_radpm"]        # Road heading angle first derivative w.r.t. s
        attitude_[2, :]  = self.data["psi_ss_radpm2"]      # Road heading angle second derivative w.r.t. s
        attitude_[3, :]  = self.data["grade_rad"]          # Road grade angle
        attitude_[4, :]  = self.data["grade_s_radpm"]      # Road grade angle first derivative w.r.t. s
        attitude_[5, :]  = self.data["grade_ss_radpm2"]    # Road grade angle second derivative w.r.t. s
        attitude_[6, :]  = self.data["bank_rad"]           # Road bank angle
        attitude_[7, :]  = self.data["bank_s_radpm"]       # Road bank angle first derivative w.r.t. s
        attitude_[8, :]  = self.data["bank_ss_radpm2"]     # Road bank angle second derivative w.r.t. s


        # Define interpolating functions for position and attitude of centerline as a function of s-coordinate
        self.posE_m_interp_fcn   = scipy.interpolate.interp1d(self.data["s_m"], self.data["posE_m"], axis=0, kind="linear", fill_value="extrapolate")
        self.posN_m_interp_fcn   = scipy.interpolate.interp1d(self.data["s_m"], self.data["posN_m"], axis=0, kind="linear", fill_value="extrapolate")
        self.posU_m_interp_fcn   = scipy.interpolate.interp1d(self.data["s_m"], self.data["posU_m"], axis=0, kind="linear", fill_value="extrapolate")
        self.psi_rad_interp_fcn  = scipy.interpolate.interp1d(self.data["s_m"], self.data["psi_rad"], axis=0, kind="linear", fill_value="extrapolate")
        self.k_1pm_interp_fcn    = scipy.interpolate.interp1d(self.data["s_m"], self.data["psi_s_radpm"], axis=0, kind="linear", fill_value="extrapolate")
        self.attitude_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], attitude_, axis=1, kind="linear", fill_value="extrapolate")
        
        
        # Generate virtual road bounds that are a constant offset from the input bounds according to edge_buffer_m
        theta = edge_buffer_m/self.data["track_width_m"]
        theta = theta.reshape(len(theta), 1)
        self.data["virtual_inner_bounds_m"] = (1-theta)*self.data["inner_bounds_m"] + (theta)*self.data["outer_bounds_m"]
        self.data["virtual_outer_bounds_m"] = (theta)*self.data["inner_bounds_m"]   + (1-theta)*self.data["outer_bounds_m"]
        self.data["virtual_track_width_m"] = self.data["track_width_m"] - 2*edge_buffer_m


        # Define inetrpolating functions for track bounds as a function of s-coordinate
        self.inner_bounds_posE_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["inner_bounds_m"][:, 0], kind="linear", fill_value="extrapolate")
        self.inner_bounds_posN_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["inner_bounds_m"][:, 1], kind="linear", fill_value="extrapolate")
        self.inner_bounds_posU_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["inner_bounds_m"][:, 2], kind = "linear", fill_value = "extrapolate")
        self.outer_bounds_posE_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["outer_bounds_m"][:, 0], kind="linear", fill_value="extrapolate")
        self.outer_bounds_posN_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["outer_bounds_m"][:, 1], kind="linear", fill_value="extrapolate")
        self.outer_bounds_posU_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["outer_bounds_m"][:, 2], kind = "linear", fill_value = "extrapolate")

        self.virtual_inner_bounds_posE_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["virtual_inner_bounds_m"][:, 0], kind="linear", fill_value="extrapolate")
        self.virtual_inner_bounds_posN_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["virtual_inner_bounds_m"][:, 1], kind="linear", fill_value="extrapolate")
        self.virtual_inner_bounds_posU_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["virtual_inner_bounds_m"][:, 2], kind = "linear", fill_value = "extrapolate")
        self.virtual_outer_bounds_posE_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["virtual_outer_bounds_m"][:, 0], kind="linear", fill_value="extrapolate")
        self.virtual_outer_bounds_posN_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["virtual_outer_bounds_m"][:, 1], kind="linear", fill_value="extrapolate") 
        self.virtual_outer_bounds_posU_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["virtual_outer_bounds_m"][:, 2], kind = "linear", fill_value = "extrapolate") 


        # Useful lookup tables for Casadi
        self.s_m_2laps = np.zeros(len(self.data["s_m"])*2-1)
        self.s_m_2laps[0:len(self.data["s_m"])] = self.data["s_m"]
        self.s_m_2laps[len(self.data["s_m"]):] = self.data["s_m"][1:] + self.length_m
        
        self.attitude_2laps = np.zeros((9, len(self.data["s_m"])*2-1))
        self.attitude_2laps[:, 0:len(self.data["s_m"])] = attitude_
        self.attitude_2laps[:, len(self.data["s_m"]):] = attitude_[:, 1:]
        
        self.psi_rad_LUT       = casadi.interpolant("psi_rad_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[0, :])
        self.psi_s_radpm_LUT   = casadi.interpolant("psi_s_radpm_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[1, :])
        self.psi_ss_radpm2_LUT = casadi.interpolant("psi_ss_radpm2_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[2, :])
        
        self.grade_rad_LUT       = casadi.interpolant("grade_rad_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[3, :])
        self.grade_s_radpm_LUT   = casadi.interpolant("grade_s_radpm_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[4, :])
        self.grade_ss_radpm2_LUT = casadi.interpolant("grade_ss_radpm2_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[5, :])
        
        self.bank_rad_LUT       = casadi.interpolant("bank_rad_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[6, :])
        self.bank_s_radpm_LUT   = casadi.interpolant("bank_s_radpm_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[7, :])
        self.bank_ss_radpm2_LUT = casadi.interpolant("bank_ss_radpm2_LUT", "linear", [self.s_m_2laps], self.attitude_2laps[8, :])      

        track_width_m_2laps = np.zeros((len(self.data["s_m"])*2-1))
        track_width_m_2laps[0:len(self.data["s_m"])] = self.data["track_width_m"]
        track_width_m_2laps[len(self.data["s_m"]):] = self.data["track_width_m"][1:]
        self.track_width_m_LUT       = casadi.interpolant("track_width_m_LUT", "linear", [self.s_m_2laps], track_width_m_2laps)


        # Miscellaneous
        self.virtual_track_width_m_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.data["virtual_track_width_m"], kind = "linear", fill_value = "extrapolate") 
        
        
        # Local track rotation matrix as defined by Heading/Grade/Bank Euler angles
        self.local_track_rotation_matrix = np.zeros((len(self.data["s_m"]), 3, 3)) # Rotation matrix ^I R ^B to describe the relation between the inertial i1i2i3 (NED) frame (I) and local frame p1p2p3 (P)

        self.local_track_rotation_matrix[:, 0, 0] = np.cos(-self.data["psi_rad"])*np.cos(-self.data["grade_rad"])
        self.local_track_rotation_matrix[:, 0, 1] = -np.sin(-self.data["psi_rad"])*np.cos(self.data["bank_rad"]) + np.sin(-self.data["grade_rad"])*np.sin(self.data["bank_rad"])*np.cos(-self.data["psi_rad"])
        self.local_track_rotation_matrix[:, 0, 2] = np.sin(-self.data["psi_rad"])*np.sin(self.data["bank_rad"]) + np.sin(-self.data["grade_rad"])*np.cos(-self.data["psi_rad"])*np.cos(self.data["bank_rad"])
        self.local_track_rotation_matrix[:, 1, 0] = np.sin(-self.data["psi_rad"])*np.cos(-self.data["grade_rad"])
        self.local_track_rotation_matrix[:, 1, 1] = np.cos(-self.data["psi_rad"])*np.cos(self.data["bank_rad"]) + np.sin(-self.data["psi_rad"])*np.sin(-self.data["grade_rad"])*np.sin(self.data["bank_rad"])
        self.local_track_rotation_matrix[:, 1, 2] = -np.sin(self.data["bank_rad"])*np.cos(-self.data["psi_rad"]) + np.sin(-self.data["psi_rad"])*np.sin(-self.data["grade_rad"])*np.cos(self.data["bank_rad"])
        self.local_track_rotation_matrix[:, 2, 0] = -np.sin(-self.data["grade_rad"])
        self.local_track_rotation_matrix[:, 2, 1] = np.sin(self.data["bank_rad"])*np.cos(-self.data["grade_rad"])
        self.local_track_rotation_matrix[:, 2, 2] = np.cos(-self.data["grade_rad"])*np.cos(self.data["bank_rad"])

        self.local_track_rotation_matrix_interp_fcn = scipy.interpolate.interp1d(self.data["s_m"], self.local_track_rotation_matrix, axis=0, kind="linear", fill_value="extrapolate")



        if(diagnostic_plotting):
            fig, ax = plt.subplots(1, 1, num="Track overhead", constrained_layout=True)
            ax.plot(self.data["inner_bounds_m"][:, 0], self.data["inner_bounds_m"][:, 1], label = "Inner bounds", color = "b", marker = "x", ms = 2)
            ax.plot(self.data["outer_bounds_m"][:, 0], self.data["outer_bounds_m"][:, 1], label = "Outer bounds", color = "r", marker = "x", ms = 2)
            ax.plot(self.data["virtual_inner_bounds_m"][:, 0], self.data["virtual_inner_bounds_m"][:, 1], label = "Virtual inner bounds", linestyle = "--", color = "b", alpha = .5, marker = "x", ms = 2)
            ax.plot(self.data["virtual_outer_bounds_m"][:, 0], self.data["virtual_outer_bounds_m"][:, 1], label = "Virtual outer bounds", linestyle = "--", color = "r", alpha = .5, marker = "x", ms = 2)
            ax.plot(self.data["posE_m"], self.data["posN_m"], marker = "x", label = "Centerline", color = "k", ms = 2)
            ax.scatter([self.data["posE_m"][0]], [self.data["posN_m"][0]], label = "Start point ({}, {})".format(round(self.data["posE_m"][0]), round(self.data["posN_m"][0])), color = "k")
            ax.legend(loc = "best")
            ax.set_aspect("equal")
            ax.set_xlabel("East position [m]")
            ax.set_ylabel("North position [m]")
            plt.show()
            
               
    
    def map_match_vectorized(self, s_vals, e_vals):
        """
        Vectorized version of map matching
        
        Inputs:
            - s_vals: np.array of s values
            - e_vals: np.array of corresponding e values
            
        Outputs:
             - x_vals: np.array of x (East) coordinates
             - y_vals: np.array of y (North) coordiantes
             - z_vals: np.array of z (Up) coordinates
        """
        
        # Centerline locations
        x_cl = self.posE_m_interp_fcn(s_vals % self.length_m)
        y_cl = self.posN_m_interp_fcn(s_vals % self.length_m)
        z_cl = self.posU_m_interp_fcn(s_vals % self.length_m)

        psi_cl = self.psi_rad_interp_fcn(s_vals % self.length_m)

        # Extract the local path frame NED direction
        t_hat_NED = self.local_track_rotation_matrix_interp_fcn(s_vals % self.length_m)[:, :, 0] # The transformed North vector aligns with the centerline tangent
        e_hat_NED = -self.local_track_rotation_matrix_interp_fcn(s_vals % self.length_m)[:, :, 1] # The transformed East vector aligns with the negative "e" direction
        b_hat_NED = -self.local_track_rotation_matrix_interp_fcn(s_vals % self.length_m)[:, :, 2] # The transformed Down vector aligns with the negative Up direction
        
        # Convert NED coordinates to ENU
        t_hat_ENU = np.hstack((t_hat_NED[:, 1].reshape(-1, 1), t_hat_NED[:, 0].reshape(-1, 1), -t_hat_NED[:, 2].reshape(-1, 1)))
        e_hat_ENU = np.hstack((e_hat_NED[:, 1].reshape(-1, 1), e_hat_NED[:, 0].reshape(-1, 1), -e_hat_NED[:, 2].reshape(-1, 1)))
        b_hat_ENU = np.hstack((b_hat_NED[:, 1].reshape(-1, 1), b_hat_NED[:, 0].reshape(-1, 1), -b_hat_NED[:, 2].reshape(-1, 1)))
        
        x_vals = x_cl + e_vals*e_hat_ENU[:, 0]
        y_vals = y_cl + e_vals*e_hat_ENU[:, 1]
        z_vals = z_cl + e_vals*e_hat_ENU[:, 2]

        return x_vals, y_vals, z_vals
    
    
    def get_racetrack_params_as_dict(self):
        return self.params
    
    