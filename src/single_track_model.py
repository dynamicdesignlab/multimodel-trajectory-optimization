from yaml import safe_load
import numpy as np
from casadi import cos, sin, tanh, atan2, sqrt, log, exp, vertcat
import fiala_functions

G_MPS2 = 9.81
BETA   = 2.0
FY_XI  = 0.95


class SingleTrackModel:
    
    def __init__(self, params_file):
        
        with open(params_file, "r") as stream:
            full_params_dict = safe_load(stream)
            try:
                self.params_dict = full_params_dict["single_track_params"]
            except:
                raise Exception("Error: Paramters .yaml file needs to have a heading called 'single_track_params'!")
        
        print("---------------")
        print("Successfully loaded in vehicle from file: {}\n".format(params_file))
        for key, value in self.params_dict.items(): print("{}: {}".format(key, value))
        print("---------------")
        
        # Populate additional (derived) parameters
        self.params_dict["l_m"] = self.params_dict["a_m"] + self.params_dict["b_m"] # wheelbase
        self.params_dict["track_avg_m"] = 0.5*(self.params_dict["track_front_m"] + self.params_dict["track_rear_m"])
        self.params_dict["fz_f_static_kn"] = self.params_dict["m_Mg"]*G_MPS2*self.params_dict["b_m"]/self.params_dict["l_m"]
        self.params_dict["fz_r_static_kn"] = self.params_dict["m_Mg"]*G_MPS2*self.params_dict["a_m"]/self.params_dict["l_m"]

        # Compute roll center properties
        self.params_dict["h_rc_o_m"] = (1/self.params_dict["l_m"])*(self.params_dict["b_m"]*self.params_dict["h_rc_f_m"] + self.params_dict["a_m"]*self.params_dict["h_rc_r_m"]);
        self.params_dict["h_l_m"] = self.params_dict["h_cm_m"] - self.params_dict["h_rc_o_m"]
        self.params_dict["R_phi_radpmps2"] = self.params_dict["m_Mg"]*self.params_dict["h_l_m"]/(self.params_dict["K_phi_f_knmprad"] + self.params_dict["K_phi_r_knmprad"] - self.params_dict["m_Mg"]*G_MPS2*self.params_dict["h_l_m"])
        
        # Compute lateral weight transfer per m/s^2 of lateral acceleration
        self.params_dict["dfz_lat_sens_knpmps2"] = self.params_dict["m_Mg"]*self.params_dict["h_cm_m"]/self.params_dict["track_avg_m"] + self.params_dict["m_Mg"]*G_MPS2*self.params_dict["h_l_m"]*self.params_dict["R_phi_radpmps2"]/self.params_dict["track_avg_m"]

    def dx_dt_casadi(self,
                     x,
                     u,
                     z,
                     mu_f,
                     mu_r,
                     auxiliary_outputs=False):
        
        """
        Implements single-track vehicle model dynamics for use with Casadi.
        
        Given     
            State       x := [vx, vy, r, t, e, dpsi, dfz_long]
            Control     u := [delta, fx],
            Topopgraphy z := [psi, theta, phi, k_psi, k_theta, k_phi],
            
        this method returns the time state derivative,
            dx/dt.
            
        """
         
        # Assemble relevant tire parameters
        ca_0_f = self.params_dict["ca_0_f_knprad"]
        ca_1_f = self.params_dict["ca_1_f_knpradpkn"]
        
        ca_0_r = self.params_dict["ca_0_r_knprad"]
        ca_1_r = self.params_dict["ca_1_r_knpradpkn"]

        # Unpack the state vector
        vx        = x[0] # m/s
        vy        = x[1] # m/s
        r         = x[2] # rad/s
        t         = x[3] # s
        e         = x[4] # m
        dpsi      = x[5] # rad
        dfz_long  = x[6] # kn
        
        # Unpack the control vector
        delta     = u[0] # rad
        fx        = u[1] # kn   
        
        psi     = z[0]
        theta   = z[1]
        phi     = z[2]
        k_psi   = z[3]
        k_theta = z[4]
        k_phi   = z[5]

        # Axle normal load calculation
        fz_f = self.params_dict["b_m"]/self.params_dict["l_m"]*self.params_dict["m_Mg"]*G_MPS2 - dfz_long
        fz_r = self.params_dict["a_m"]/self.params_dict["l_m"]*self.params_dict["m_Mg"]*G_MPS2 + dfz_long
        
        # Calculate drive and brake force partitioning
        chi_f = (self.params_dict["drive_front_frac"] - self.params_dict["brake_front_frac"])/2*tanh(2*(fx + 0.5)) + (self.params_dict["drive_front_frac"] + self.params_dict["brake_front_frac"])/2
        chi_r = (self.params_dict["brake_rear_frac"]  - self.params_dict["drive_rear_frac"])/2*tanh(-2*(fx + 0.5)) + (self.params_dict["drive_rear_frac"] + self.params_dict["brake_rear_frac"])/2
        fx_f = chi_f*fx
        fx_r = chi_r*fx
            
        # Calculate slip angles
        alpha_f = atan2((vy + self.params_dict["a_m"]*r), vx) - delta
        alpha_r = atan2((vy - self.params_dict["b_m"]*r), vx)

        # Saturate the force command Fx if it's above the Fx limit of the tire
        eps_f = self.params_dict["eps_f"]
        eps_r = self.params_dict["eps_r"]
        fx_f = 1/2*(sqrt((fx_f + mu_f*fz_f*cos(alpha_f))**2 + eps_f**2) - sqrt((fx_f - mu_f*fz_f*cos(alpha_f))**2 + eps_f**2))
        fx_r = 1/2*(sqrt((fx_r + mu_r*fz_r*cos(alpha_r))**2 + eps_r**2) - sqrt((fx_r - mu_r*fz_r*cos(alpha_r))**2 + eps_r**2))
        
        # Calculate the lateral force produced on each tire   
        fy_f = fiala_functions.calculate_fy(ca_0_f, ca_1_f, mu_f, alpha_f, fx_f, fz_f, FY_XI)
        fy_r = fiala_functions.calculate_fy(ca_0_r, ca_1_r, mu_r, alpha_r, fx_r, fz_r, FY_XI)

        # Calculate drag and perceived topographical force terms
        f_rr   = 0.001*self.params_dict["cd0_n"] # Rolling resistance
        f_aero = 0.001*(self.params_dict["cd1_nspm"]*vx + self.params_dict["cd2_ns2pm2"]*vx**2) # Aerodynamic drag
        f_grade = -self.params_dict["m_Mg"]*G_MPS2*sin(theta) # Perceived force due to grade
        
        f_d = f_rr + f_aero + f_grade
        f_l = -self.params_dict["m_Mg"]*G_MPS2*cos(theta)*sin(phi)
        
        # Calculate the body accelerations
        ax = (fx_f*cos(delta) - fy_f*sin(delta) + fx_r - f_d)/self.params_dict["m_Mg"]
        ay = (fy_f*cos(delta) + fx_f*sin(delta) + fy_r + f_l)/self.params_dict["m_Mg"]
        
        # Calculate the braking yaw moment
        fx_f_braking_only = -1/BETA*log(1 + exp(-BETA*fx_f))
        fx_r_braking_only = -1/BETA*log(1 + exp(-BETA*fx_r))
        
        dfz_lat = self.params_dict["dfz_lat_sens_knpmps2"]*ay
        
        mz_f_diff_braking = fx_f_braking_only*self.params_dict["gamma"]    *self.params_dict["track_avg_m"]*dfz_lat/fz_f
        mz_r_diff_braking = fx_r_braking_only*(1-self.params_dict["gamma"])*self.params_dict["track_avg_m"]*dfz_lat/fz_r
        
        mz_diff_braking = mz_f_diff_braking + mz_r_diff_braking
        alphaz = (self.params_dict["a_m"]*(fy_f*cos(delta) + fx_f*sin(delta)) - self.params_dict["b_m"]*fy_r + mz_diff_braking)/self.params_dict["Izz_Mgm2"]
        
        
        # Evaluate the state derivatives
        s_dot     = (vx*cos(dpsi) - vy*sin(dpsi))/(1-k_psi*e)
        vx_dot    = ((fx_f*cos(delta) - fy_f*sin(delta) + fx_r - f_d)/self.params_dict["m_Mg"] + r*vy)
        vy_dot    = ((fy_f*cos(delta) + fx_f*sin(delta) + fy_r + f_l)/self.params_dict["m_Mg"] - r*vx)
        r_dot     = (alphaz)
        t_dot     = 1
        e_dot     = (vx*sin(dpsi) + vy*cos(dpsi))
        dpsi_dot  = r - k_psi*s_dot
        dfz_long_dot = (1/self.params_dict["tau_long_weight_transfer_s"])*(self.params_dict["m_Mg"]*ax*self.params_dict["h_cm_m"]/self.params_dict["l_m"] - dfz_long)
        
        dx_dt = vertcat(vx_dot,
                        vy_dot,
                        r_dot,
                        t_dot,
                        e_dot,
                        dpsi_dot,
                        dfz_long_dot)
        
       
        if(auxiliary_outputs):
            return dx_dt, fx_f, fy_f, fx_r, fy_r, fz_f, fz_r, ax, ay, alphaz, alpha_f, alpha_r, mz_diff_braking
        else:
            return dx_dt
        

    def dx_ds_casadi(self, x, u, z, mu_f, mu_r, auxiliary_outputs=False):
        
        if(auxiliary_outputs):
            dx_dt, fx_f, fy_f, fx_r, fy_r, fz_f, fz_r, ax, ay, alphaz, alpha_f, alpha_r, mz_diff_braking = self.dx_dt_casadi(x, u, z, mu_f, mu_r, auxiliary_outputs=auxiliary_outputs)
            
            s_dot = (x[0]*cos(x[5]) - x[1]*sin(x[5]))/(1-z[3]*x[4])
            dx_ds = dx_dt/s_dot
            dx_ds[3] = 1/s_dot
            
            return dx_ds, fx_f, fy_f, fx_r, fy_r, fz_f, fz_r, ax, ay, alphaz, alpha_f, alpha_r, mz_diff_braking
        
        else:
            dx_dt = self.dx_dt_casadi(x, u, z, mu_f, mu_r, auxiliary_outputs=auxiliary_outputs)
        
            s_dot = (x[0]*cos(x[5]) - x[1]*sin(x[5]))/(1-z[3]*x[4])
            dx_ds = dx_dt/s_dot
            dx_ds[3] = 1/s_dot
        
            return dx_ds


    def dx_ds_sim(self,
                  s,
                  x,
                  x_ref_interpolant,
                  u_ref_interpolant,
                  z_ref_interpolant,
                  track_width_interpolant,
                  fb_gains_dict,
                  mu_road_interpolant,
                  road_length):
        
        """
        Computes the state derivative assuming closed-loop feedback to the
        reference trajectory.
        """
    
        
        # Interpolate into references at current s position
        x_ref = x_ref_interpolant(s % road_length)
        u_ref = u_ref_interpolant(s % road_length)
        z_ref = z_ref_interpolant(s % road_length)
        mu_f = mu_road_interpolant((s + self.params_dict["a_m"]*cos(x[5])) % road_length)
        mu_r = mu_road_interpolant((s - self.params_dict["b_m"]*cos(x[5])) % road_length)
        
        # Compute feedback control effort        
        delta_fb = - (  fb_gains_dict["K_delta_e"]*(x[4] - x_ref[4])
                      + fb_gains_dict["K_delta_dpsi"]*(x[5] - x_ref[5]))

        
        fx_fb =    - (  fb_gains_dict["K_fx_ux"]*(x[0] - x_ref[0]))
        
        u_sim = u_ref
        u_sim[0] += delta_fb
        u_sim[1] += fx_fb
        delta_max_rad = np.radians(self.params_dict["max_delta_deg"])
        u_sim[0] = np.clip(u_sim[0], -delta_max_rad, delta_max_rad)
        
        dx_ds = self.dx_ds_casadi(x, u_sim, z_ref, mu_f, mu_r)
        dx_ds = np.asarray(dx_ds).squeeze()

        return dx_ds