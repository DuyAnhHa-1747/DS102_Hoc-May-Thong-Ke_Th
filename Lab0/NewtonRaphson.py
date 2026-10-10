import numpy as np


def f(u):
    return u[0]**2 + 2*u[1]**2 - u[0]*u[1] - 4*u[0]


def grad(u):
    return np.array([2*u[0] - u[1] - 4, 4*u[1] - u[0]])


def hessian(u):
    return np.array([
        [2, -1],
        [-1, 4]
    ])


def newton_raphson(grad_fn, hessian_fn, u_0, tol=1e-8, max_iter=100):
    u = np.array(u_0, dtype=float)
    his = [u.copy()]
    for _ in range(max_iter):
        g = grad_fn(u)
        if np.linalg.norm(g) <= tol:
            break
        H = hessian_fn(u)
        a = np.linalg.solve(H, g)
        u = u - a
        his.append(u.copy())
    return u, his


if __name__ == "__main__":
    u_start = [1.0, 2.0]
    u_opt, history = newton_raphson(grad, hessian, u_start)
    print("Gia tri cuc tri la:", f(u_opt))
    print("Toa do diem cuc tri la:", u_opt)
