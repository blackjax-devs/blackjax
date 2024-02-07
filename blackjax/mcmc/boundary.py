
import jax.numpy as jnp




class Boundary():
    """Forms a transformation map which will bound the parameter space (this transformation will be applied in the position_update of the Hamiltonian dynamics integration)"""
  
    def __init__(self, d,
                 where_positive = None,
                 where_reflect = None,
                 where_periodic = None,
                 a = None, b = None,
                 ):
        
        """
        where_positive: indices of positively constrained parameters
        where_reflect: indices of rectangularly constrained parameters (with reflective boundary). Use if parameter is constrained to an interval (for example 0 < x < 1), but it is not periodic.
        where_periodic: indices of rectangularly constrained parameters (with periodic boundary). Use for example for angles.
        a: lower bounds
        b: upper bounds
        
        Example:
            We have parameters
                x = [x0, x1, x2, x3, x4, x5, x6] 
            and we want constraints:
                x0 unconstrained
                x1 > 0
                0 < x2 < 2 pi (periodic)
                x3 unconstrained
                0 < x4 < 1 (not periodic)
                -1 < x5 < 1 (not periodic)
                x6 > 0
                
            We should use:
            where_positive = jnp.array([1, 6])
            where_reflect = jnp.array([4, 5])
            where_periodic = jnp.array([2, ])
            a = jnp.array([0.,       0.,-1.])
            b = jnp.array([2 jnp.pi, 1., 1.])
        """


        self.d = d
        
        self.mask_positive = self.to_mask(where_positive)
        self.mask_reflect = self.to_mask(where_reflect)
        self.mask_periodic = self.to_mask(where_periodic)


        self.a, self.b = self.extend_bounds(jnp.logical_or(self.mask_reflect, self.mask_periodic), a, b)
        
    
    def map(self, x):
        """maps R^d to the constrained region
            Args: 
                x: unconstrained parameter vector
            Returns:
                x': constrained parameter vector
                sgn: array of signs (+1 or -1), indicating which component of the velocity should be fliped.
        """
        
        # These functions map R^d to the constrained region (the unconstrained parameters are also maped but this will be ignored later). 
        # They also return a boolean array (r) which indicate which components of the velocity should be fliped.
        x0, r0 = x, False
        x1, r1 = self._positive(x)
        x2, r2 = self._reflect(x)
        x3, r3 = self._periodic(x)

        combine = lambda y0, y1, y2, y3: self.mask_positive * y1 + self.mask_reflect * y2 + self.mask_periodic * y3 + (1- (self.mask_positive + self.mask_reflect + self.mask_periodic)) * y0
                
        return combine(x0, x1, x2, x3), 1 - 2 * combine(r0, r1, r2, r3)
        


    def _positive(self, x):
        return jnp.abs(x), x < 0.
    
    def _periodic(self, x):
        return jnp.mod(x - self.a, self.b - self.a) + self.a, False
    
    def _reflect(self, x):
        y = jnp.mod((x - self.a) / (self.b - self.a), 2.)
        z = 1 - jnp.abs(1. - y)
        return z * (self.b-self.a) + self.a, y > 1.
    


    def extend_bounds(self, mask, a, b):
        A = jnp.zeros(len(mask))
        B = jnp.ones(len(mask))
        
        if a != None:
            A = A.at[mask].set(a)
            B = B.at[mask].set(b)

        return A, B
    
    
    def to_mask(self, where):
        
        mask = jnp.zeros(self.d, dtype = bool)
        
        if where == None:
            return mask 
        else:   
            return mask.at[where].set(True)
