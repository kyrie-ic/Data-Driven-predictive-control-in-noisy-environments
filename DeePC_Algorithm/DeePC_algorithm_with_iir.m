% Define system parameters from Table 1
R_m = 8.4;     % Motor armature resistance
K_t = 0.042;   % Motor torque constant
K_m = 0.042;   % Back-emf constant
m_p = 0.024;   % Pendulum mass
l_p = 0.0645;  % Distance from pivot to CG
J_p = 3.3282e-5; % Pendulum moment of inertia
J_r = 5.7198e-5; % Rotary arm moment of inertia
B_p = 0.0005;  % Pendulum damping coefficient
B_r = 0.0015;  % Viscous friction
r = 0.085;     % Rotary arm length
g = 9.81;      % Gravitational constant
data = load('C:\path_to_file\y_clean.mat'); % Load cleaned data
y_clean = data.y_new_all;

% State-space matrices for the downward position
A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, (g*l_p^2*m_p^2*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r), -(B_r*J_p*R_m + J_p*K_m*K_t + B_r*R_m*l_p^2*m_p + K_m*K_t*l_p^2*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_p*l_p*m_p*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r);
     0, -(R_m*g*l_p*m_p^2*r^2 + J_r*R_m*g*l_p*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_r*R_m*l_p*m_p*r + K_m*K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), -(B_p*R_m*m_p*r^2 + B_p*J_r*R_m)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];

B = [0; 0; (K_t*m_p*l_p^2 + J_p*K_t)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)); -(K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];
C = eye(4);
D = zeros(4,1);
dt = 0.1;

% Discretize state-space model
sys = ss(A, B, C, D);
sysd = c2d(sys, dt);

% Discrete-time system matrices
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

% Simulation data setup
T_total = 30;   % Total time
t = 0:dt:T_total;
N = length(t);  % Time steps
m = size(B, 2);
p = size(C, 1);

% Random input
u = randn(N, m);

% Initialize state and output
x = zeros(size(Ad, 1), N);  % State
y = zeros(p, N);            % Output
mu_bar = 0.001^2;           % Variance
mu_limit = 0.001;           % Noise limit

% Covariance matrix
cov_matrix = mu_bar * eye(p);

% Initialize noise matrix
noise_matrix = zeros(N, p);

% Generate truncated noise
for i = 1:N
    while true
        mu_k = mvnrnd(zeros(1, p), cov_matrix);  % Sample noise
        if max(abs(mu_k)) <= mu_limit
            noise_matrix(i, :) = mu_k;
            break;
        end
    end
end

% Simulate discrete-time system
for k = 1:N-1
    x(:, k+1) = Ad * x(:, k) + Bd * u(k, :)';
    y(:, k) = Cd * x(:, k) + Dd * u(k, :)';
end
y(:, N) = Cd * x(:, N) + Dd * u(N, :)';  % Last time step output
y = y' + noise_matrix;  % Add noise

% Initialize DeePC parameters
T_ini = 1;   % Initial data length
T_pred = 20; % Prediction horizon
p_steps = 200; % Total time steps
theta_amp = deg2rad(20);  % Amplitude
theta_freq = 0.1;         % Frequency

W = kron(eye(T_pred), diag([1, 1, 1, 1]));
options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');

% Initialize input/output data
u_ini = repmat(zeros(m,1),T_ini,1);
y_ini = repmat(zeros(p,1),T_ini,1);

% Build Hankel matrix
U_d = build_hankel(u', T_ini, T_pred);
Y_d = build_hankel(y', T_ini, T_pred);

U_p = U_d(1:m*T_ini, :);
U_f = U_d(m*T_ini+1:end, :);
Y_p = Y_d(1:p*T_ini, :);
Y_f = Y_d(p*T_ini+1:end, :);

x_new = zeros(4, p_steps);
target_angel_all = [];
Q = W' * W;
f = @(target_angle) -2 * (target_angle' * Q * Y_f)';
H_combined = [U_p; Y_p; U_f];
Pi = pinv(H_combined) * H_combined;
lamda_ii = 3000;
H = 2 * (Y_f' * Q * Y_f + 0.00001 * (U_f' * U_f) + lamda_ii * (eye(size(Pi)) - Pi)' * (eye(size(Pi)) - Pi));
H = (H + H') / 2;
error = 0;

% Recursive DeePC loop
for t_step = 1:p_steps
    theta_ref = theta_amp * square(2 * pi * theta_freq * t(t_step));
    target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
    target_angel_all = [target_angel_all; [theta_ref, 0, 0, 0]];

    % Equality constraints
    Aeq = [U_p; Y_p];
    beq = [u_ini(:); y_ini(:)];
    A = kron(eye(T_pred * 2), [0,1,0,0;0,0,1,0;0,0,0,1]) * [Y_f; -Y_f];
    b = kron(ones(T_pred * 2, 1),[0.0284;0.5443;0.2274]) - mu_limit;

    % Solve optimization problem
    [g_opt, fval] = quadprog(H, f(target_angle), [], [], Aeq, beq, [], [], [], options);
    sum(g_opt);

    % Compute optimal inputs
    u_opt = U_f * g_opt;
    y_opt = Y_f * g_opt;

    % Update input/output data
    u_opt_all = [u_opt_all; u_opt(1:m)'];
    y_opt_all = [y_opt_all; y_opt(1:p)'];
    x_new(:, t_step+1) = Ad * x_new(:, t_step) + Bd * u_opt(1:m)';
    y_new = Cd * x_new(:, t_step) + Dd * u_opt(1:m)';
    u_ini = [u_ini(m+1:end); u_opt(1:m)'];
    y_ini = [y_ini(p+1:end); y_new];
    y_new_all = [y_new_all; y_new'];
    error = error + ((y_new(1) -theta_ref).^2);
end

error = error/p_steps;

% Plot results
figure;
for i = 1:p
    subplot(2, 2, i);
    plot(0:dt:(length(y_opt_all)-1)*dt, y_new_all(:, i));
    hold on;
    plot(0:dt:(length(y_opt_all)-1)*dt, target_angel_all(:,i), '--');
    xlabel('Time (seconds)');
    ylabel(['Output ', num2str(i), ' (radians)']);
    legend(['Output ', num2str(i)], 'Reference');
    title(['DeePC Control of Inverted Pendulum - Output ', num2str(i)]);
end

function H = build_hankel(data, T_ini, T_pred)
    [num_data, num_samples] = size(data);
    H = zeros(num_data * (T_ini + T_pred), num_samples - T_ini - T_pred + 1);
    for i = 1:(num_samples - T_ini - T_pred + 1)
        H(:, i) = reshape(data(:, i:i + T_ini + T_pred - 1), [], 1);
    end
end
