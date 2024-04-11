import numpy as np
import sympy as sp
import cv2

x, y, vx, vy, ax, ay, jx, jy, sx, sy = sp.symbols('x y vx vy ax ay jx jy sx sy')


class extended_kalman_for_tracking_in_image:
    def __init__(self, f, g, mu_0, sigma_0, Q, R, F, G):
        """

        :param f
        :param h
        :param mu_0: počáteční stav
        :param sigma_0: počáteční kovarianční matice
        :param Q: kovarianční matice šumu procesu
        :param R: kovarianční matice šumu měření
        :param F: Jacobian f
        """
        self.f = f
        self.g = g
        self.mu_prev_k = mu_0
        self.sigma_prev_k = sigma_0
        self.Q = Q
        self.R = R
        self.F = F
        self.G = G
        self.dt = 1

    def step(self, measurement):
        measurement = np.array(measurement).reshape(2, 1)
        mu_star = self.f.subs(
            {x: self.mu_prev_k[0], y: self.mu_prev_k[1]})
        sigma_star = self.F @ self.sigma_prev_k @ self.F.T + self.Q
        self.K = sigma_star @ self.G.T @ np.linalg.inv(self.G @ sigma_star @ self.G.T + self.R)
        mu_k = mu_star + self.K @ (measurement - self.g.subs({x: mu_star[0], y: mu_star[1]}))
        sigma_k = sigma_star - self.K @ self.G @ sigma_star
        self.mu_prev_k = mu_k
        self.sigma_prev_k = sigma_k

    def predict(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        return mu, sigma

    def predict_and_update(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        self.mu_prev_k = mu
        self.sigma_prev_k = sigma
        return mu, sigma

    def get_mu(self):
        return self.mu_prev_k

    def get_sigma(self):
        return self.sigma_prev_k



class extended_kalman_for_tracking_in_image_with_velocity(extended_kalman_for_tracking_in_image):

    def __init__(self, f, g, mu_0, sigma_0, Q, R, F, G):
        super().__init__(f, g, mu_0, sigma_0, Q, R, F, G)

    def step(self, measurement):
        measurement = np.array(measurement).reshape(2, 1)
        mu_star = self.f.subs(
            {x: self.mu_prev_k[0], y: self.mu_prev_k[1], vx: self.mu_prev_k[2], vy: self.mu_prev_k[3]})
        sigma_star = self.F @ self.sigma_prev_k @ self.F.T + self.Q
        self.K = sigma_star @ self.G.T @ np.linalg.inv(self.G @ sigma_star @ self.G.T + self.R)
        mu_k = mu_star + self.K @ (measurement - self.g.subs({x: mu_star[0], y: mu_star[1]}))
        sigma_k = sigma_star - self.K @ self.G @ sigma_star
        self.mu_prev_k = mu_k
        self.sigma_prev_k = sigma_k

    def predict(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        return mu, sigma

    def predict_and_update(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        self.mu_prev_k = mu
        self.sigma_prev_k = sigma
        return mu, sigma

class extended_kalman_for_tracking_in_image_with_acceleration(extended_kalman_for_tracking_in_image):

    def __init__(self, f, g, mu_0, sigma_0, Q, R, F, G):
        super().__init__(f, g, mu_0, sigma_0, Q, R, F, G)

    def step(self, measurement):
        measurement = np.array(measurement).reshape(2, 1)
        mu_star = self.f.subs(
            {x: self.mu_prev_k[0], y: self.mu_prev_k[1], vx: self.mu_prev_k[2], vy: self.mu_prev_k[3], ax: self.mu_prev_k[4], ay: self.mu_prev_k[5]})
        sigma_star = self.F @ self.sigma_prev_k @ self.F.T + self.Q
        self.K = sigma_star @ self.G.T @ np.linalg.inv(self.G @ sigma_star @ self.G.T + self.R)
        mu_k = mu_star + self.K @ (measurement - self.g.subs({x: mu_star[0], y: mu_star[1]}))
        sigma_k = sigma_star - self.K @ self.G @ sigma_star
        self.mu_prev_k = mu_k
        self.sigma_prev_k = sigma_k

    def predict(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3], ax: mu[4], ay: mu[5]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        return mu, sigma

    def predict_and_update(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3], ax: mu[4], ay: mu[5]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        self.mu_prev_k = mu
        self.sigma_prev_k = sigma
        return mu, sigma

class extended_kalman_for_tracking_in_image_with_acceleration_and_jerk(extended_kalman_for_tracking_in_image):

        def __init__(self, f, g, mu_0, sigma_0, Q, R, F, G):
            super().__init__(f, g, mu_0, sigma_0, Q, R, F, G)

        def step(self, measurement):
            measurement = np.array(measurement).reshape(2, 1)
            mu_star = self.f.subs(
                {x: self.mu_prev_k[0], y: self.mu_prev_k[1], vx: self.mu_prev_k[2], vy: self.mu_prev_k[3], ax: self.mu_prev_k[4], ay: self.mu_prev_k[5], jx: self.mu_prev_k[6], jy: self.mu_prev_k[7]})
            sigma_star = self.F @ self.sigma_prev_k @ self.F.T + self.Q
            self.K = sigma_star @ self.G.T @ np.linalg.inv(self.G @ sigma_star @ self.G.T + self.R)
            mu_k = mu_star + self.K @ (measurement - self.g.subs({x: mu_star[0], y: mu_star[1]}))
            sigma_k = sigma_star - self.K @ self.G @ sigma_star
            self.mu_prev_k = mu_k
            self.sigma_prev_k = sigma_k

        def predict(self, number_of_steps):
            mu = self.mu_prev_k
            sigma = self.sigma_prev_k
            for i in range(number_of_steps):
                mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3], ax: mu[4], ay: mu[5], jx: mu[6], jy: mu[7]})
                sigma = self.F @ sigma @ self.F.T + self.Q
            return mu, sigma

        def predict_and_update(self, number_of_steps):
            mu = self.mu_prev_k
            sigma = self.sigma_prev_k
            for i in range(number_of_steps):
                mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3], ax: mu[4], ay: mu[5], jx: mu[6], jy: mu[7]})
                sigma = self.F @ sigma @ self.F.T + self.Q
            self.mu_prev_k = mu
            self.sigma_prev_k = sigma
            return mu, sigma

class extended_kalman_for_tracking_in_image_with_acceleration_and_jerk_and_snap(extended_kalman_for_tracking_in_image):

    def __init__(self, f, g, mu_0, sigma_0, Q, R, F, G):
        super().__init__(f, g, mu_0, sigma_0, Q, R, F, G)

    def step(self, measurement):
        measurement = np.array(measurement).reshape(2, 1)
        mu_star = self.f.subs(
            {x: self.mu_prev_k[0], y: self.mu_prev_k[1], vx: self.mu_prev_k[2], vy: self.mu_prev_k[3], ax: self.mu_prev_k[4], ay: self.mu_prev_k[5], jx: self.mu_prev_k[6], jy: self.mu_prev_k[7], sx: self.mu_prev_k[8], sy: self.mu_prev_k[9]})
        sigma_star = self.F @ self.sigma_prev_k @ self.F.T + self.Q
        self.K = sigma_star @ self.G.T @ np.linalg.inv(self.G @ sigma_star @ self.G.T + self.R)
        mu_k = mu_star + self.K @ (measurement - self.g.subs({x: mu_star[0], y: mu_star[1]}))
        sigma_k = sigma_star - self.K @ self.G @ sigma_star
        self.mu_prev_k = mu_k
        self.sigma_prev_k = sigma_k

    def predict(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3], ax: mu[4], ay: mu[5], jx: mu[6], jy: mu[7], sx: mu[8], sy: mu[9]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        return mu, sigma

    def predict_and_update(self, number_of_steps):
        mu = self.mu_prev_k
        sigma = self.sigma_prev_k
        for i in range(number_of_steps):
            mu = self.f.subs({x: mu[0], y: mu[1], vx: mu[2], vy: mu[3], ax: mu[4], ay: mu[5], jx: mu[6], jy: mu[7], sx: mu[8], sy: mu[9]})
            sigma = self.F @ sigma @ self.F.T + self.Q
        self.mu_prev_k = mu
        self.sigma_prev_k = sigma
        return mu, sigma


