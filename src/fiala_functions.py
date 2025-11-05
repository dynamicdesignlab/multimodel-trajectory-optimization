from casadi import tan, fabs, sqrt, atan2, if_else, sign


def calculate_fy(ca_0, ca_1, mu, alpha, fx, fz, fy_xi):
    """
    Calculates the lateral tire force using the Fx-derated lateral Fiala model
    """
    
    ca = ca_0 + ca_1*fz # Affine cornering stiffness w.r.t. normal load
    
    tan_alpha       = tan(alpha)
    tan_alpha_cubed = tan_alpha**3
    abs_tan_alpha   = fabs(tan_alpha)
    
    fy_max = sqrt((mu*fz)**2 - (0.99*fx)**2)
    alpha_star = atan2(3*fy_max*fy_xi, ca) # Saturation slip angle
    
    fy = if_else(
        fabs(alpha) <= alpha_star,
        (- ca*tan_alpha + ca**2/3/fy_max*abs_tan_alpha*tan_alpha - ca**3/27/fy_max**2*tan_alpha_cubed),
        (- ca*(1 - 2*fy_xi + fy_xi**2)*tan_alpha + fy_max*(-3*fy_xi**2 + 2*fy_xi**3)*sign(alpha))
        )
    
    return fy