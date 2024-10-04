Please see `doc/guideline.pdf` for instructions.

## Usage
```bash
conda activate <env>
python /test/student grader.py -q q1 # Run AutoGrader for question 1
python /test/student grader.py -q q1 --graphics # Run AutoGrader for question 1 with graphics
python /test/student grader.py -q q3 --graphics --iter-step=20 # To specify simulation iteration steps for faster debugging
```

## Solution

### Question 1
Simulation of the robot dynamics.

Motion model:

$$
\begin{align*} 
    s_t = 
    \begin{pmatrix}
        x_t \\
        y_t \\
        \theta_t
    \end{pmatrix}
    = g(s_{t-1}, u_t) = 
    \begin{pmatrix}
        x_{t-1} + \Delta_t \cos(\theta_{t-1} + \delta_t) \\
        y_{t-1} + \Delta_t \sin(\theta_{t-1} + \delta_t) \\
        \theta_{t-1} + \delta_t
    \end{pmatrix}
\end{align*}
$$

$$
s_t = g(s_{t-1}, u_t) + \epsilon_t \quad \text{where} \quad \epsilon_t \sim \mathcal{N}(0, \Sigma)
$$

### Question 2 ~ 4
EKF algorithm

$$
\begin{align*}
    \text{Prediction update:} \quad
    & \hat G_t = g(\hat \mu_{t-1}, u_t) \\
    & \hat \Sigma_t = G_t \hat \Sigma_{t-1} G_t^T + R_t \\
    \text{Measurement update:} \quad
    & K_t = \hat \Sigma_t H_t^T (H_t \hat \Sigma_t H_t^T + Q_t)^{-1} \\
    & \hat \mu_t = \hat \mu_t + K_t (z_t - h(\hat \mu_t)) \\
    & \hat \Sigma_t = (I - K_t H_t) \hat \Sigma_t
\end{align*}
$$