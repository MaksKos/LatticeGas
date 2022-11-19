import pytest
import latticegas as lgas
import numpy as np

@pytest.mark.parametrize(
    ('xc', 'yc'),
    [
        (3, 6), (5, 6), (14, 6), (18, 6),
        (10, 3), (10, 5), (10, 9), (10, 11),
    ]
)
def test_add_cylinder_bound(xc, yc):

    radius = 5
    shape = (20 , 15)
    with pytest.raises(ValueError):
        lgas.LatticeGas.add_cylinder(xc, yc, radius, shape)

def test_add_cylinder():

    radius = 3
    shape = (10 , 10)
    xc, yc = 5, 5
    occup_cells = lgas.LatticeGas.add_cylinder(xc, yc, radius, shape)
    answer = {'x': [2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8], 
            'y': [5, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 5]}
    assert occup_cells == answer

def test_calc_outflow():

    f_in = []
    shape = (10, 5)
    for i in range(9):
        f_in.append(np.full(shape, i))
    f_in = np.array(f_in)
    correct = np.copy(f_in)
    for i in range(3):
        correct[8-i, -1, :] = i
    lgas.LatticeGas.calc_outflow(f_in)

    assert np.allclose(f_in, correct)

def test_initial():

    parametrs = {'nx':11, 'ny':7, 'u_lb':0.1, 'Re':20}
    obstacle = {'xc':5, 'yc':3, 'r':2}

    model = lgas.LatticeGas(parametrs, obstacle)

    assert model.u.shape == (parametrs['nx'], parametrs['ny'], 2)
    assert model.u[:, :, -1].sum().astype(int) == 0
    assert model.f_in.shape == (9, parametrs['nx'], parametrs['ny'])

def test_calc_u():

    shape = (5, 3)
    density = np.ones(shape)
    v = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    f_in = np.ones((len(v), *shape))
    u = lgas.LatticeGas.calc_u(density[:,:,np.newaxis], f_in[:,:,:,np.newaxis], v)
    answer = np.zeros((*shape, 2))
    assert u.shape == (*shape, 2)
    assert np.allclose(answer, u)

def test_calc_f_eq_i():

    i = 0
    shape = (5, 5)
    parametrs = {'nx':shape[0], 'ny':shape[1], 'u_lb':0.1, 'Re':20}
    obstacle = {'xc':2, 'yc':2, 'r':1}

    density = np.ones(shape)
    u = np.full((*shape, 2), 0.5)

    model = lgas.LatticeGas(parametrs, obstacle)
    model._a[i] = 1
    model._v[i] = np.array([1,1])
    f_eq = model.calc_f_eq_i(i, u, density)
    answer = np.full(shape, 7.75)

    assert np.allclose(f_eq, answer, atol=1e-2)

def test_calc_inflow():
    
    shape = (5, 5)
    parametrs = {'nx':shape[0], 'ny':shape[1], 'u_lb':0.1, 'Re':20}
    obstacle = {'xc':2, 'yc':2, 'r':1}

    model = lgas.LatticeGas(parametrs, obstacle)
    model._a = np.ones(9)
    model._v = np.ones((9, 2))
    model.f_in = np.zeros_like(model.f_in)
    for i in range(3):
        model.f_in[i,0,:] = np.ones(shape[1])
    model.calc_inflow()
    assert np.allclose(model.f_in, np.zeros_like(model.f_in))