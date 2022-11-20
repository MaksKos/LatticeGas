import numpy as np
from torch import zeros,tensor,roll,sin,sqrt,linspace
from torch import sum as tsum
import torch as tr
# num of directions
N = 9
DEVICE = 'cpu' #tr.device('cuda' if tr.cuda.is_available() else 'cpu')


class LatticeGas():
    
    _a = tr.tensor([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
    _index = tr.arange(N)
    _vx = tr.tensor([1, 1, 1, 0, 0, 0, -1, -1, -1])
    _vy = tr.tensor([1, 0, -1, 1, 0, -1, 1, 0, -1])
    _ind_right = tr.tensor([8, 7, 6])
    _ind_middle = tr.tensor([3, 4, 5])
    _ind_left = tr.tensor([0, 1, 2])

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

        self.vi = self.u_lb*obstacle['r']/self.Re
        self.omega = 1/(3*self.vi + 0.5)

        self.f_in = tr.ones((self.n_x, self.n_y, N))
        self.f_out = tr.ones((self.n_x, self.n_y, N))
        self.f_equil = tr.ones((self.n_x, self.n_y, N))
        self.density = tr.ones((self.n_x, self.n_y))
        self.obstacle = self.add_cylinder(obstacle['xc'], obstacle['yc'], obstacle['r'], (self.n_x, self.n_y))
        self.__init_velo()
        self.__init_f_in()

    def __init_velo(self):
        """
        Init vectors filed of velosity 'u' in each pint
        with small disturbance
        Shape ux and uy (nx, ny)
        """
        self.u_y = tr.zeros((self.n_x, self.n_y))
        self.u_x = tr.zeros((self.n_x, self.n_y))
        self.v_init = self.u_lb*(1 + 1e-4*tr.sin(2*tr.pi/(self.n_y-1)*tr.arange(self.n_y)))
        self.u_x[:,] += self.v_init 

    def __init_f_in(self):
        """
        Init field with N possible directions
        of propagation
        Shape (nx, ny, N)
        """
        for i, vx, vy, a in zip(self._index, self._vx, self._vy, self._a):
            dot  = self.u_x*vx + self.u_y*vy
            norma = self.u_x**2 + self.u_y**2
            self.f_in[:, :, i] = self.density*a*(1 + 3*dot + 4.5*dot**2 - 1.5*norma)
    
    @staticmethod
    def add_cylinder(xc: int, yc: int, r: int, shape: tuple) -> np.ndarray:
        """Add circle oh filed

        Args:
            xc (int): x coordinate of center
            yc (int): y coordinate of center
            r (int): radius
            shape (tuple): (nx, ny) shape of field 

        Raises:
            ValueError: circle out of field size

        Returns:
            np.ndarray <bool>: mask of cylinder nodes
        """
        if  xc+r >= shape[0]-1 or\
            xc-r <= 0 or\
            yc+r >= shape[1]-1 or\
            yc-r <= 0:
            raise ValueError("Circle out of field")

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0])) 
        cylinder = (x - xc)**2 + (y - yc)**2 <= r**2
        return cylinder

    @staticmethod
    def calc_outflow(f_in: tr.TensorType):
        """Calculate outflow 
        boundary condition

        Args:
            f_in (ndarray): filed of possible directions
            of propagation
        """
        index = tr.tensor([6, 7, 8])
        f_in[index, -1, :] = f_in[index, -2, :]
        return f_in
    
    @staticmethod
    def calc_u(density: tr.TensorType, f_in: tr.TensorType, vx: tr.TensorType, vy: tr.TensorType) -> tr.TensorType:
        """Calculate field of velosity vectors 'u'

        Args:
            density (np.ndarray): field of density
            f_in (np.ndarray): filed of possible directions
            vx (np.ndarray): x coordinates of possible directions
            vy (np.ndarray): x coordinates of possible directions

        Returns:
            list: ux and ux fields
        """
        ux = tr.sum(f_in*vx, axis=-1).to(DEVICE)
        ux /= density
        uy = tr.sum(f_in*vy, axis=-1).to(DEVICE)
        uy /= density
        return [ux, uy]

    def calc_f_equil(self) -> np.ndarray:
        """Calculate f equilimbrium
        Returns:
            np.ndarray: f equilimbrium with sahpe (x, y, N)
        """
        norma = self.u_x**2 + self.u_y**2
        for i, vx, vy, a in zip(self._index, self._vx, self._vy, self._a):
            dot  = self.u_x*vx + self.u_y*vy
            self.f_equil[:,:,i] = self.density*a*(1 + 3*dot + 4.5*dot**2 - 1.5*norma)

    def calc_inflow(self):
        """
        Calculate inflow boundary condition
        """
        self.u_x[0] = self.v_init
        self.u_y[0,:] = 0
        rho_2 = tr.sum(self.f_in[0,:,self._ind_middle], axis=0).to(DEVICE)
        rho_3 = tr.sum(self.f_in[0,:,self._ind_right], axis=0).to(DEVICE)
        self.density[0, :] = 1/(1 - self.u_x[0]) * (rho_2 + 2*rho_3)

        self.calc_f_equil()

        self.f_in[0, :, self._ind_left] = self.f_equil[0,:, self._ind_left] +\
                                            self.f_in[0,:, self._ind_right] -\
                                            self.f_equil[0,:, self._ind_right]
            
    def calc_f_out(self):
        """
        Calculate pre-collision (f out)
        """
        self.f_out = self.f_in - self.omega*(self.f_in - self.f_equil)
    
    def bounce_back(self):
        """
        Bounce-back boundary condition on 
        solid obstacle
        """
        bndry = self.f_in[self.obstacle,:]
        self.f_out[self.obstacle,:] = bndry[:, self._index[::-1]]

    def collision(self):
        """
        Calculate post-collision process
        """
        for i, vx, vy in zip(self._index, self._vx, self._vy):
            self.f_in[:,:,i] = tr.roll(
                                    tr.roll(self.f_out[:,:,i], vy, axis=1), 
                                    vx, axis=0
                                )

    def solve(self, n_step=100_000, step_frame = 100):

        for time in range(n_step):

            if time%step_frame == 0:
                self.__save()

            self.f_in = self.calc_outflow(self.f_in)
            self.density = tr.sum(self.f_in, axis=-1)
            self.u_x, self.u_y = self.calc_u(self.density, self.f_in, self._vx, self._vy)
            self.calc_inflow()
            self.calc_f_out()
            self.bounce_back()
            self.collision()

    def __save(self):
        if not hasattr(self, 'field_u'): self.field_u = []
        if not hasattr(self, 'field_den'): self.field_den = []
        if not hasattr(self, 'field_ux'): self.field_ux = []
        if not hasattr(self, 'field_uy'): self.field_uy = []
        
        self.field_u.append(np.sqrt(self.u_x**2 + self.u_y**2))
        self.field_den.append(self.density)
        self.field_ux.append(self.u_x)
        self.field_uy.append(self.u_y)

    @property
    def field_p(self):
        return np.array(self.field_den)/3