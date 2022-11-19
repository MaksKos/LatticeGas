import numpy as np
from numba import njit

class LatticeGas():
    
    _a = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
    _v = np.array([[1, 1], [1, 0], [1, -1], [0, 1], 
                      [0, 0], 
                      [0, -1], [-1, 1], [-1, 0], [-1, -1]])

    def __init__(self, parametrs: dict, obstacle: dict) -> None:
        """Initialized filed of flow and
        macro prametrs

        Args:
            parametrs (dict): filed's settings 
            obstacle (dict): circle's parametrs
        """
        self.n_x = parametrs['nx']
        self.n_y = parametrs['ny']
        self.u_lb = parametrs['u_lb']
        self.Re = parametrs['Re']
        #self.xc = obstacle['xc']
        #self.yc = obstacle['yc']
        #self.r = obstacle['r']

        self.vi = self.u_lb*obstacle['r']/self.Re
        self.omega = 1/(3*self.vi + 0.5)

        self.f_in = np.zeros((9, self.n_x, self.n_y))
        self.f_out = np.zeros((9, self.n_x, self.n_y))
        self.density = np.ones((self.n_x, self.n_y))
        self.obstacle = self.add_cylinder(obstacle['xc'], obstacle['yc'], obstacle['r'], (self.n_x, self.n_y))
        self.__init_velo()
        self.__init_f_in()

    def __init_velo(self):
        """
        Init vectors filed of velosity 'u' in each pint
        with small disturbance
        Shape (nx, ny, 2)
        """
        y = np.arange(self.n_y)
        v_init = self.u_lb*(1 + 1e-4*np.sin(2*np.pi/(self.n_y-1)*y))
        zeros = np.zeros(self.n_y)
        self.u = np.array([np.column_stack((v_init, zeros)) for _ in range(self.n_x)])

    def __init_f_in(self):
        """
        Init field with 9 possible directions
        of propagation
        Shape (9, nx, ny)
        """
        for i in range(9):
            self.f_in[i,:,:] = self.density*self._a[i]*(1 + 3*(self.u@self._v[i]) + 4.5*(self.u@self._v[i])**2 \
                                - 1.5*np.linalg.norm(self.u, axis=2)**2)
    
    @staticmethod
    def add_cylinder(xc: int, yc: int, r: int, shape: tuple) -> dict:
        """Add circle oh filed

        Args:
            xc (int): x coordinate of center
            yc (int): y coordinate of center
            r (int): radius
            shape (tuple): (nx, ny) shape of field 

        Raises:
            ValueError: circle out of field size

        Returns:
            dict: {'x': list, 'y': list}
        """
        if  xc+r >= shape[0]-1 or\
            xc-r <= 0 or\
            yc+r >= shape[1]-1 or\
            yc-r <= 0:
            raise ValueError("Circle out of field")

        occupied = {'x':[], 'y':[]}
        for x in range(xc-r-1, xc+r+1):
            for y in range(yc-r-1, yc+r+1):
                if (x-xc)**2 + (y-yc)**2 <= r**2:
                    occupied['x'].append(x)
                    occupied['y'].append(y)

        return occupied

    @staticmethod
    def calc_outflow(f_in: np.ndarray):
        """Calculate outflow 
        boundary condition

        Args:
            f_in (ndarray): filed of possible directions
            of propagation
        """
        for i in range(3):
            f_in[8-i,-1,:] = f_in[i,-1,:]
    
    @staticmethod
    def calc_u(density: np.ndarray, f_in: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Calculate field of velosity vectors 'u'

        Args:
            density (np.ndarray): field of density
            f_in (np.ndarray): filed of possible directions
            v (np.ndarray): list of normalize vectors of possible
            directions

        Returns:
            np.ndarray: field of vectors ''u
        """
        u = np.zeros((*f_in.shape, 2))
        for i in f_in.shape[0]:
            for j in f_in.shape[1]:
                for k in f_in.shape[2]:
                    u[i, j, k] = f_in[i, j, k]*v[i]/density[j,k]
        u = np.sum(u, axis=0)
        return u

    def calc_f_eq_i(self, i: int, u: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Calculate f equal in i-direction

        Args:
            i (int): index of direction
            u (np.ndarray): filed of vectors felosity
            density (np.ndarray): field of density
        
        Shape of u (x, y, 2) and density (x, y)

        Returns:
            np.ndarray: f_i equal with sahpe (x, y)
        """
        if u.shape[:-1] != density.shape:
            raise ValueError(f"incorrect shape 'u' ({u.shape[:-1]}) and 'density' ({density.shape})")

        return density*self._a[i]*(1 + 3*u@self._v[i] + 4.5*(u@self._v[i])**2 - 1.5*np.linalg.norm(u, axis=-1)**2)

    def calc_inflow(self):
        """
        Calculate inflow boundary condition
        """
        for i in range(3):
            self.f_in[i,0,:] = self.calc_f_eq_i(i, self.u[0,:,:], self.density[0,:]) \
                                + (self.f_in[8-i, 0, :] - self.calc_f_eq_i(8-i, self.u[0, :, :], self.density[0, :]))
            
    def calc_f_out(self):
        """
        Calculate pre-collision (f out)
        """
        for i in range(9):
            self.f_out[i,:,:] = self.f_in[i,:,:] - self.omega*(self.f_in[i,:,:]\
                                 - self.calc_f_eq_i(i, self.u, self.density))
    
    def bounce_back(self):
        """
        Bounce-back boundary condition on 
        solid obstacle
        """
        for i in range(9):
            self.f_out[i][self.obstacle['x'], self.obstacle['y']] = \
                -self.f_in[8-i][self.obstacle['x'], self.obstacle['y']]

    def collision(self):
        """
        Calculate post-collision process
        """
        for i, direct in enumerate(self._v):
            self.f_in[i] = np.roll(np.roll(self.f_out[i], direct[1], axis=1), direct[0], axis=0)


    def solve(self, n_step=100_000, step_frame = 100):

        for time in range(n_step):
            self.calc_outflow(self.f_in)
            self.density = np.sum(self.f_in, axis=0)
            self.u = self.calc_u(self.density, self.f_in, self._v)
            self.calc_inflow()
            self.calc_f_out()
            self.bounce_back()
            self.collision()

            if time%step_frame == 0:
                self.__save()

    def __save(self):
        if not hasattr(self, 'field_u'):
            self.field_u = []
        if not hasattr(self, 'field_den'):
            self.field_den = []
        #if not hasattr(self, 'field_vec_u'):
        #   self.field_vec_u = []
        
        self.field_u.append(np.linalg.norm(self.u, axis=-1))
        self.field_den.append(self.density)

    @property
    def field_p(self):
        return np.array(self.field_den)/3