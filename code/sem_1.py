import numpy as np


def create_random_params(n: int) -> np.ndarray:
    """
    Создаёт массив случайных параметров размером (8, n) в [0, 1].

    Параметры:
    n : int
        Количество столкновений (пар частиц).

    Возвращает:
    a : np.ndarray
        Массив параметров.
    """
    return np.random.rand(8, n)


def normalize_params(a: np.ndarray, d: float, xi_cut: float) -> None:
    """
    Нормирует параметры массива:
    - a[:6]: скорости в [-xi_cut, xi_cut]
    - a[6]: прицельное расстояние в [0, d]
    - a[7]: азимутальный угол в [0, 2π]

    Параметры:
    a : np.ndarray
        Исходный массив параметров (8, n).
    d : float
        Прицельное расстояние (диаметр).
    xi_cut : float
        Скорость обрезания.
    """
    # Скорости
    a[:6] = 2 * xi_cut * a[:6] - xi_cut
    # b ∈ [0, d]
    a[6] *= d
    # ε ∈ [0, 2π]
    a[7] *= 2 * np.pi


def compute_relative_velocities(a: np.ndarray):
    """
    Вычисляет скорости xi, xi1 и относительные скорости g.

    Параметры:
    a : np.ndarray
        Нормализованный массив параметров (8, n).

    Возвращает:
    xi, xi1 : np.ndarray, shape (n, 3)
    g : np.ndarray, относительные скорости (n, 3)
    """
    xi = a[:3].T
    xi1 = a[3:6].T
    g = xi1 - xi
    return xi, xi1, g


def compute_theta(b: np.ndarray, d: float) -> np.ndarray:
    """
    Вычисляет углы отклонения θ по формуле для твёрдых сфер.

    Параметры:
    b : np.ndarray, shape (n)
        Прицельные расстояния.
    d : float
        Диаметр частиц.

    Возвращает:
    theta : np.ndarray, shape (n)
    """
    # θ = 2 arccos(b/d) если b ≤ d, иначе 0
    return np.where(b <= d, 2 * np.arccos(b / d), 0.0)


def transform_g(g: np.ndarray, eps: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Трансформирует относительные скорости после столкновения по формулам 1.12a и 1.12b.

    Параметры:
    g : np.ndarray, shape (n, 3)
        Относительные скорости до столкновения.
    eps : np.ndarray, shape (n)
        Азимутальные углы.
    theta : np.ndarray, shape (n)
        Углы отклонения.

    Возвращает:
    g_new : np.ndarray, shape (n, 3)
        Относительные скорости после столкновения.
    """
    n = g.shape[0]
    g_new = np.zeros_like(g)
    for i in range(n):
        gi = g[i]
        norm_g = np.linalg.norm(gi)
        if norm_g < 1e-10:
            continue
        ti, ei = theta[i], eps[i]
        cos_t, sin_t = np.cos(ti), np.sin(ti)
        cos_e, sin_e = np.cos(ei), np.sin(ei)
        gx, gy, gz = gi
        # общая формула
        if np.abs(gx) + np.abs(gy) > 1e-10:
            g_xy = np.hypot(gx, gy)
            gx_new = (gx * cos_t - (gx * gz / g_xy) * cos_e * sin_t + (gy / g_xy) * norm_g * sin_e * sin_t)
            gy_new = (gy * cos_t - (gy * gz / g_xy) * cos_e * sin_t - (gx / g_xy) * norm_g * sin_e * sin_t)
            gz_new = gz * cos_t + g_xy * cos_e * sin_t
        else:
            # вдоль z
            gx_new = norm_g * np.sin(ti) * sin_e
            gy_new = norm_g * np.sin(ti) * cos_e
            gz_new = norm_g * np.cos(ti)
        g_new[i] = [gx_new, gy_new, gz_new]
    return g_new


def compute_post_collision_velocities(xi: np.ndarray, xi1: np.ndarray, g_new: np.ndarray):
    """
    Вычисляет скорости частиц после столкновения по формуле 1.8.

    Параметры:
    xi, xi1 : np.ndarray, shape (n, 3)
        Скорости до столкновения.
    g_new : np.ndarray, shape (n, 3)
        Новые относительные скорости.

    Возвращает:
    xi_prime, xi1_prime : np.ndarray, speed (n, 3)
    """
    xi_prime = 0.5 * (xi + xi1) - 0.5 * g_new
    xi1_prime = 0.5 * (xi + xi1) + 0.5 * g_new
    return xi_prime, xi1_prime


def check_and_save(xi: np.ndarray, xi1: np.ndarray, xi_prime: np.ndarray,
                   xi1_prime: np.ndarray, check_conservation: bool, save_to_file: bool) -> None:
    """
    Проверяет законы сохранения и при необходимости сохраняет результаты в файлы.
    """
    if check_conservation:
        g_before = xi1 - xi
        g_after = xi1_prime - xi_prime
        eb_before = np.sum(xi ** 2 + xi1 ** 2, axis=1)
        eb_after = np.sum(xi_prime ** 2 + xi1_prime ** 2, axis=1)
        mb_before = xi + xi1
        mb_after = xi_prime + xi1_prime
        print("Проверки сохранения:")
        print(
            f"Относительная скорость: {np.max(np.abs(np.linalg.norm(g_before, axis=1) - np.linalg.norm(g_after, axis=1))):.2e}")
        print(f"Энергия: {np.max(np.abs(eb_before - eb_after)):.2e}")
        print(f"Импульс: {np.max(np.abs(mb_before - mb_after)):.2e}")

    if save_to_file:
        np.savetxt("xi_prime.csv", xi_prime, delimiter=",")
        np.savetxt("xi1_prime.csv", xi1_prime, delimiter=",")
        print("Данные сохранены в xi_prime.csv и xi1_prime.csv")


def simulate_collisions(n: int = 10000,
                        d: float = 1.0,
                        xi_cut: float = 4.8,
                        check_conservation: bool = False,
                        save_to_file: bool = False):
    """
    Основная функция симуляции столкновений, разбитая на этапы.
    """
    # 1
    a = create_random_params(n)
    # 2
    normalize_params(a, d, xi_cut)
    # 3
    xi, xi1, g = compute_relative_velocities(a)
    # 4
    theta = compute_theta(a[6], d)
    # 5
    g_new = transform_g(g, a[7], theta)
    # 6
    xi_prime, xi1_prime = compute_post_collision_velocities(xi, xi1, g_new)
    # 7
    check_and_save(xi, xi1, xi_prime, xi1_prime, check_conservation, save_to_file)
    return xi_prime, xi1_prime
