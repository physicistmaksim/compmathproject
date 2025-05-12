import random
from math import sqrt, pi

from sem_2 import *


def eratosthenes_sieve(n):
    """Generate prime numbers up to n using Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


def H(z, p, s=5):
    """Compute H(z) function for Korobov grid coefficients (Eq. 3.35)."""
    total = 0.0
    for k in range(1, (p - 1) // 2 + 1):
        product = 1.0
        for i in range(1, s + 1):
            # Iterative modulo to avoid overflow
            kz = k
            for _ in range(i):
                kz = (kz * z) % p
            frac = kz / p
            product *= (1 - 2 * (frac - int(frac))) ** 2

        total += product

    return total


def find_korobov_coefficients(p):
    """Find Korobov grid coefficients for prime p."""
    H_values = []
    for z in range(1, (p - 1) // 2 + 1):
        H_values.append((H(z, p), z))
    min_H = min(H_values, key=lambda x: x[0])[0]
    b_candidates = [h[1] for h in H_values if h[0] == min_H]
    b = b_candidates[0]  # Choose first candidate
    a = [1]
    current = b
    for _ in range(1, p):
        a.append(current)
        current = (current * b) % p
    return a


def initialize_distribution(N, xi_cut, T_tilde=0.5, condition=1):
    """Initialize distribution function f (Eq. 3.35 or 3.36)."""
    f = np.zeros((3, N))
    xi, dxi = create_velocity_grid(N, N, N, xi_cut)
    xi_sq = xi ** 2
    if np.sqrt(xi_sq).all() <= xi_cut:
        if condition == 1:  # Eq. 3.35
            f = np.exp(-0.5 * xi_sq) + T_tilde ** (-1.5) * np.exp(-0.5 * xi_sq / T_tilde)
        else:  # Eq. 3.36
            f = np.exp(-0.5 * xi_sq) if xi[0].all() > 0 else T_tilde ** (-1.5) * np.exp(-0.5 * xi_sq / T_tilde)

    volume = dxi[0] * dxi[1] * dxi[2]
    norm = volume * np.sum(f, axis = 0)
    f /= norm  # Normalize to n=1
    return f, xi, dxi


def find_interpolation_nodes(xi_prime, xi_1_prime, xi_grid, delta_xi, xi_cut):
    """Find interpolation nodes and coefficient r_v (Eq. 2.4, 2.5, 2.8)."""
    N = len(xi_grid)

    def get_node_idx(xi_val):
        idx = int((xi_val + xi_cut) / delta_xi)
        idx = max(0, min(N - 2, idx))
        return idx

    lambda_idx = [get_node_idx(xi_prime[i]) for i in range(3)]
    mu_idx = [get_node_idx(xi_1_prime[i]) for i in range(3)]
    s_v = [1 if xi_prime[i] > xi_grid[lambda_idx[i]] else -1 for i in range(3)]
    p_v = [1 if xi_1_prime[i] > xi_grid[mu_idx[i]] else -1 for i in range(3)]
    lambda_s_idx = [lambda_idx[i] + s_v[i] for i in range(3)]
    mu_p_idx = [mu_idx[i] + p_v[i] for i in range(3)]
    E0 = np.sum(xi_prime ** 2) + np.sum(xi_1_prime ** 2)
    E1 = sum(xi_grid[lambda_idx[i]] ** 2 for i in range(3)) + sum(xi_grid[mu_idx[i]] ** 2 for i in range(3))
    E2 = sum(xi_grid[lambda_s_idx[i]] ** 2 for i in range(3)) + sum(xi_grid[mu_p_idx[i]] ** 2 for i in range(3))
    if E1 <= E0 <= E2:
        r_v = (E0 - E1) / (E2 - E1) if E2 != E1 else 0.5
    else:
        r_v = 0.5  # Fallback
    return lambda_idx, mu_idx, s_v, p_v, r_v


def compute_collision_integral(f, xi_grid, dxi, xi_cut, N_v, b_max=1.0, tau=0.02):
    """Implement scheme (3.22) for collision integral."""
    N = f.shape[0]
    V_sph = 4 * pi * xi_cut ** 3 / 3
    volume = dxi[0] * dxi[1] * dxi[2]
    N_0 = round(V_sph / volume)
    C = (b_max ** 2 * volume * N_0 ** 2 / N_v * tau) / (2 ** 2.5)
    p = int(4 * N_v)  # Approximate Korobov grid size
    primes = eratosthenes_sieve(p + 100)
    p = min([x for x in primes if x >= p], default=p)
    a = find_korobov_coefficients(p)
    grid = np.array([((i * a[j]) % p) / p for j in range(5) for i in range(1, p + 1)]).reshape(-1, 5)
    grid[:, 0] *= 2 * pi  # epsilon
    grid[:, 1] *= b_max ** 2  # S = b^2
    grid[:, 2:5] = grid[:, 2:5] * 2 * xi_cut - xi_cut  # xi_1
    random.shuffle(grid)  # Shuffle grid blocks
    f_new = f.copy
    for v in range(min(N_v, len(grid))):
        epsilon, S, xi_1 = grid[v, 0], grid[v, 1], grid[v, 2:5]
        if np.sqrt(np.sum(xi_1 ** 2)) > xi_cut:
            continue

        alpha_idx = [random.randint(0, N - 1) for _ in range(3)]  # массив из 3 чисел
        xi_alpha = np.array([xi_grid[i][alpha_idx[i]] for i in range(3)])

        d = 1.
        a = create_random_params(N_v)
        normalize_params(a, d, xi_cut)
        xi, xi1, g = compute_relative_velocities(a)
        theta = compute_theta(a[6], d)
        g_new = transform_g(g, a[7], theta)

        xi_prime, xi_1_prime = compute_post_collision_velocities(xi_alpha, xi_1, g_new)
        if np.sqrt(np.sum(xi_prime ** 2)) > xi_cut or np.sqrt(np.sum(xi_1_prime ** 2)) > xi_cut:
            continue

        lambda_idx, mu_idx, s_v, p_v, r_v = find_interpolation_nodes(xi_prime, xi_1_prime, xi_grid, dxi, xi_cut)
        lambda_s_idx = [lambda_idx[i] + s_v[i] for i in range(3)]
        mu_p_idx = [mu_idx[i] + p_v[i] for i in range(3)]
        f_alpha = f[tuple(alpha_idx)]
        f_beta = f[tuple([int((xi_1[i] + xi_cut) / dxi) for i in range(3)])]
        f_lambda = f[tuple(lambda_idx)]
        f_mu = f[tuple(mu_idx)]
        f_lambda_s = f[tuple(lambda_s_idx)]
        f_mu_p = f[tuple(mu_p_idx)]
        if f_lambda * f_mu == 0:
            continue
        Omega = ((f_lambda * f_mu) ** (1 - r_v) * (f_lambda_s * f_mu_p) ** r_v - f_alpha * f_beta) * np.linalg.norm(
                xi_alpha - xi_1)
        delta_f = np.zeros_like(f)
        delta_f[tuple(alpha_idx)] += C * Omega
        delta_f[tuple([int((xi_1[i] + xi_cut) / dxi) for i in range(3)])] += C * Omega
        delta_f[tuple(lambda_idx)] -= (1 - r_v) * C * Omega
        delta_f[tuple(mu_idx)] -= (1 - r_v) * C * Omega
        delta_f[tuple(lambda_s_idx)] -= r_v * C * Omega
        delta_f[tuple(mu_p_idx)] -= r_v * C * Omega
        f_temp = f_new + delta_f
        if np.any(f_temp < 0):
            continue

        f_new = f_temp
    return f_new


def compute_macro_parameters(f, xi_grid, dxi):
    """Compute macroscopic parameters n, u, T."""
    volume = dxi[0] * dxi[1] * dxi[2]
    n = np.sum(f) * volume
    u = np.sum(f * xi_grid, axis = 0) * volume
    u /= n
    T = np.sum(f * (xi_grid - u) ** 2) * volume
    T /= (3 * n)
    return n, u, T


def newton_method(xi_grid, dxi, u_guess, T_guess):
    """Find u*, T* for Maxwellian distribution using Newton's method."""

    def compute_y(u_star, T_star):
        volume = dxi[0] * dxi[1] * dxi[2]
        f_M = np.exp(-0.5 * np.sum((xi_grid - u_star) ** 2, axis=0) / T_star)
        norm = np.sum(f_M) * volume
        f_M /= norm
        xi_mean = np.sum(f_M * xi_grid) * volume
        xi_sq_mean = np.sum(f_M * np.sum(np.array(xi_grid) ** 2, axis=0) * volume)
        y1 = (xi_sq_mean - u_star[0] ** 2) / 3 - T_guess
        y2 = xi_mean - u_guess[0]
        return y1, y2

    N = xi_grid.shape[1]
    u_star = np.zeros((3, N))
    u_star[0][0] = u_guess[0]
    T_star = T_guess
    for _ in range(10):
        y1, y2 = compute_y(u_star, T_star)
        if abs(y1).all() < 1e-6 and abs(y2) < 1e-6:
            break
        # Compute partial derivatives numerically
        eps = 1e-6
        y1_u, y2_u = compute_y(u_star[0] + eps, T_star)
        y1_T, y2_T = compute_y(u_star, T_star + eps)
        dy1_du = (y1_u - y1) / eps
        dy1_dT = (y1_T - y1) / eps
        dy2_du = (y2_u - y2) / eps
        dy2_dT = (y2_T - y2) / eps
        J = np.array([[dy1_dT[0], dy1_du[0]], [dy2_dT, dy2_du]])
        delta = np.linalg.solve(J, [-y1[0], -y2])
        T_star += delta[0]
        u_star[0] += delta[1]
    return u_star, T_star


def compute_norms(f, f_M, xi_grid, dxi, xi_cut):
    """Compute differentiated norms (Eq. 3.38)."""
    norms = []
    volume = dxi[0] * dxi[1] * dxi[2]
    for k in range(1, 4):
        mask = ((k - 1) * xi_cut / 3 <= np.sqrt(xi_grid[k-1] ** 2)) & \
               (np.sqrt(xi_grid[k-1] ** 2) <= k * xi_cut / 3)

        # print(f[k-1] - f_M[k-1], "\n")
        norm = np.sqrt((f[k-1] - f_M[k-1]) ** 2 * mask) * volume
        norms.append(norm)
    return norms


def main():
    N = 20
    xi_cut = 4.8
    tau = 0.02
    T_tilde = 0.95
    time_steps = int(0.4 / tau)
    # f_max = 1 / ((2 * pi) ** 1.5)
    # V_sph = 4 * pi * xi_cut ** 3 / 3
    N_0 = 4224
    W_min = (N_0 * xi_cut ** 4) / (6 * sqrt(pi))
    N_v = int(W_min * tau)
    f, xi_grid, dxi = initialize_distribution(N, xi_cut, T_tilde, condition=1)
    n, u, T = compute_macro_parameters(f, xi_grid, dxi)
    u_star, T_star = newton_method(xi_grid, dxi, u, T)
    f_M = np.exp(-0.5 * (xi_grid - u_star) ** 2 / T_star)
    volume = dxi[0] * dxi[1] * dxi[2]
    f_M /= np.sum(f_M, axis = 0) * volume
    # print(np.sum((f - f_M) ** 2))
    initial_norms = compute_norms(f, f_M, xi_grid, dxi, xi_cut)
    delta_k = []
    for t in range(time_steps):
        f = compute_collision_integral(f, xi_grid, dxi, xi_cut, N_v, tau=tau)
        norms = compute_norms(f, f_M, xi_grid, dxi, xi_cut)
        delta_k.append([norm / init_norm for norm, init_norm in zip(norms, initial_norms)])

    # Conservation check
    n_new, u_new, T_new = compute_macro_parameters(f, xi_grid, dxi)
    print(f"Conservation check: Δn={abs(n_new - n)}, Δu={np.linalg.norm(u_new - u)}, ΔT={abs(T_new - T)}")

    # Symmetry check
    f_0 = f.copy()
    tau_sym = 1e-6
    time_steps_sym = int(100 * tau_sym / tau)
    for _ in range(time_steps_sym):
        f = compute_collision_integral(f, xi_grid, dxi, xi_cut, N_v, tau=tau_sym)
    I = (f - f_0) / (100 * tau_sym)
    sym_error = sqrt(np.sum((I[:N // 2, :, :] - I[N - 1:N // 2 - 1:-1, :, :]) ** 2)) / sqrt(0.5 * np.sum(I ** 2))
    print(f"Symmetry error: {sym_error}")

    # Convergence and other checks would require additional runs with different N and p
    # Plotting delta_k vs model solution

    t = np.array([i * tau for i in range(time_steps)])
    model = -t * (16 / (5 * sqrt(2 * pi))) * sqrt(T)
    # Plotting would be done here using matplotlib
    # Repeat for t=2 and condition=2 as needed


if __name__ == "__main__":
    main()
