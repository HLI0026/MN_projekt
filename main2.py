import numpy as np
import sympy as sp
import cv2
import time as t
from my_kalman import extended_kalman_for_tracking_in_image_with_acceleration

x, y, vx, vy, ax, ay = sp.symbols('x y vx vy ax ay')
f = sp.Matrix([x + vx + (ax**2)/2, y + vy + (ay**2)/2, vx + ax, vy+ay, ax, ay])
g = sp.Matrix([x, y])
mu_0 = np.array([1280/2, 720/2,0,0, 0, 0])
sigma_0 = np.eye(6)*1

dt = 1
Q = np.array([[3**2, 0, 0, 0, 0, 0],
                [0, 3**2, 0, 0, 0, 0],
                [0, 0, 3**2, 0, 0, 0],
                [0, 0, 0, 3**2, 0, 0],
                [0, 0, 0, 0, 0.01**2, 0],
                [0, 0, 0, 0, 0, 0.01**2]])

R = np.array([[10**2, 0],
              [0, 10**2]])

F = np.array([[1, 0, dt, 0, dt**2/2, 0],
                [0, 1, 0, dt, 0, dt**2/2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])

G = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]])


ekf = extended_kalman_for_tracking_in_image_with_acceleration(f, g, mu_0, sigma_0, Q, R, F, G)

scale = 1.3
n_neighbors = 5
min_size = (70, 70)
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
turn_pred = False
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
steps = 3

h = 70
w = 70

while True:
    beg = t.time()
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale, n_neighbors, cv2.CASCADE_SCALE_IMAGE, min_size)

    if  cv2.waitKey(1) & 0xFF == ord('w'):
        turn_pred = not turn_pred

    if len(faces) == 0 or turn_pred:

        mu_predicted, sigma_predicted = ekf.predict_and_update(1)

        cv2.putText(frame, 'Predicted', (int(mu_predicted[0]), int(mu_predicted[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (int(mu_predicted[0]), int(mu_predicted[1])),
                      (int(mu_predicted[0] + w), int(mu_predicted[1] + h)), (255, 0, 0), 2)

    else:
        for (xx, yy, w, h) in faces:
            mu_prev, sigma_prev = ekf.predict(1)
            ekf.step(np.array([xx, yy]))
            mu = ekf.get_mu()

            mu_future, sigma_future = ekf.predict(steps)

            cv2.rectangle(frame, (int(mu[0]), int(mu[1])), (int(mu[0] + w), int(mu[1] + h)), (255, 0, 0), 2)
            cv2.rectangle(frame, (xx, yy), (xx + w, yy + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(mu_future[0]), int(mu_future[1])), (int(mu_future[0] + w), int(mu_future[1] + h)), (0, 0, 255), 2)




    sgm = ekf.get_sigma()
    mu = ekf.get_mu()

    cv2.putText(frame, "x = " + str(mu[0]) + " y = " + str(mu[1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                2, cv2.LINE_AA)
    cv2.putText(frame, "vx = " + str(mu[2]) + " vy = " + str(mu[3]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                2, cv2.LINE_AA)
    cv2.putText(frame, "ax = " + str(mu[4]) + " ay = " + str(mu[5]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                2,  cv2.LINE_AA)
    cv2.putText(frame, "fps = " + str(1/(t.time()-beg)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, "sigma diag = " + str(np.diag(ekf.sigma_prev_k)), (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

