% Define system parameters from Table 1
R_m = 8.4;     % Motor armature resistance
K_t = 0.042;   % Motor torque constant
K_m = 0.042;   % Back-emf constant
m_p = 0.024;   % Pendulum mass
l_p = 0.0645;  % Distance from pivot to center of gravity
J_p = 3.3282e-5; % Pendulum moment of inertia
J_r = 5.7198e-5; % Rotary arm moment of inertia
B_p = 0.0005;  % Pendulum damping coefficient
B_r = 0.0015;  % Viscous friction
r = 0.085;     % Rotary arm length
g = 9.81;      % Gravitational acceleration

% State-space matrices for the downward position
A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, (g*l_p^2*m_p^2*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r), -(B_r*J_p*R_m + J_p*K_m*K_t + B_r*R_m*l_p^2*m_p + K_m*K_t*l_p^2*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_p*l_p*m_p*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r);
     0, -(R_m*g*l_p*m_p^2*r^2 + J_r*R_m*g*l_p*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_r*R_m*l_p*m_p*r + K_m*K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), -(B_p*R_m*m_p*r^2 + B_p*J_r*R_m)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];

B = [0; 0; (K_t*m_p*l_p^2 + J_p*K_t)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)); -(K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];

C = eye(4);
D = [0; 0; 0; 0];
dt = 0.1;

% Discretize the state-space model
sys = ss(A, B, C, D);
sysd = c2d(sys, dt);

% Extract discrete-time system matrices
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

%% Simulate data generation
% Time settings
T_total = 100;    % Total time
t = 0:dt:T_total;
N = length(t);    % Number of time steps

% Random input
m = size(B, 2);
p = size(C, 1);
u = randn(N, m);  % Random input

% Initialize state and output
x = zeros(size(Ad, 1), N);  % State
y = zeros(p, N);            % Output

% Noise settings
mu_bar = 0.01^2;  % Variance
mu_limit = 0.01;  % Truncation limit
cov_matrix = mu_bar * eye(p);  % Covariance matrix
noise_matrix = zeros(N, p);    % Initialize noise matrix

% Generate truncated noise matrix
for i = 1:N
    while true
        mu_k = mvnrnd(zeros(1, p), cov_matrix);  % Sample noise from normal distribution
        if max(abs(mu_k)) <= mu_limit  % Check truncation condition
            noise_matrix(i, :) = mu_k;
            break;
        end
    end
end

% Simulate the discrete-time system response
for k = 1:N-1
    x(:, k+1) = Ad * x(:, k) + Bd * u(k, :)';  % Update state
    y(:, k) = Cd * x(:, k) + Dd * u(k, :)';    % Output calculation
end
y(:, N) = Cd * x(:, N) + Dd * u(N, :)';  % Output for the last time step
y1 = y';
y = y + noise_matrix';  % Add noise to the output

%% Initialization for prediction
T_ini = 1;   % Initial input/output data length
T_pred = 10; % Prediction length (only first step)
p_steps = 200; % Total number of time steps
theta_amp = deg2rad(20);  % Amplitude of oscillation
theta_freq = 0.1;  % Frequency of oscillation

% Initialize input/output data
u_ini = repmat(zeros(m, 1), T_ini, 1);
y_ini = repmat(zeros(p, 1), T_ini, 1);

% Build Hankel matrices
U_d = build_hankel(u', T_ini, T_pred);
Y_d = build_hankel(y', T_ini, T_pred);
Y_dd = build_hankel(y1', T_ini, T_pred);

U_p = U_d(1:m*T_ini, :);
U_f = U_d(m*T_ini+1:end, :);
Y_p = Y_d(1:p*T_ini, :);
Y_pp = Y_dd(1:p*T_ini, :);
Y_f = Y_d(p*T_ini+1:end, :);

% Constraints setup
x_new = zeros(4, p_steps);
A_eq = [U_d; ones(1, size(U_d, 2))];  % Equality constraint
b_eq = [zeros(size(A_eq, 1), 1)];

% Upper bound constraint
G = [eye(p); -eye(p)];
A_ub = [G * Y_d(1:p, :); eye(size(U_d, 2)); -eye(size(U_d, 2))];
b_ub = [0.001 * ones(2 * p, 1); ones(size(U_d, 2), 1); zeros(size(U_d, 2), 1)];

% Polyhedron constraints for optimization
A_mu = Polyhedron('A', A_ub, 'b', b_ub, 'Ae', A_eq, 'be', b_eq);

% Prediction settings
L = T_ini + T_pred;
E_mu_l_sets = cell(1, L);
lb = [-100; -0.0284; -0.5443; -0.2274] + mu_limit;
ub = [100; 0.0284; 0.5443; 0.2274] - mu_limit;

% Combine sets for optimization
Z_l = Polyhedron('lb', lb, 'ub', ub);
A_combined = [];
b_combined = [];

% Iterate over prediction steps
for l = 2:L
    Y_d_l = Y_d((l-1)*p+1:l*p, :);  % Subset of Hankel matrix
    E_mu_l = Y_d_l * A_mu;
    E_mu_l_sets{l} = E_mu_l;
    Z_hat_l = Z_l - E_mu_l;
    A = Z_hat_l.A;
    b = Z_hat_l.b;
    A_combined = blkdiag(A_combined, A);  % Combine constraints
    b_combined = [b_combined; b];
end
