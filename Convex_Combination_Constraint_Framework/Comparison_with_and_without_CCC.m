% Define system parameters from Table 1
R_m = 8.4;     % Motor armature resistance
K_t = 0.042;   % Motor torque constant
K_m = 0.042;   % Back-emf constant
m_p = 0.024;   % Pendulum mass
l_p = 0.0645;  % Distance from pivot to center of gravity
J_p = 3.3282e-5; % Pendulum moment of inertia
J_r = 5.7198e-5; % Rotary arm moment of inertia
B_p = 0.0005;  % Pendulum damping coefficient
B_r = 0.0015;  % Viscous friction force
r = 0.085;     % Rotary arm length
g = 9.81;      % Gravitational constant

% State-space matrices for downward position
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

% Discrete-time system matrices
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

%% Simulate data generation
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

mu_bar = 0.05^2;  % Variance
mu_limit = 0.05;  % Truncation limit

% Covariance matrix
cov_matrix = mu_bar * eye(p);

% Initialize noise matrix
noise_matrix = zeros(N, p);

% Generate truncated noise matrix
for i = 1:N
    while true
        mu_k = mvnrnd(zeros(1, p), cov_matrix);  % Sample from normal distribution
        if max(abs(mu_k)) <= mu_limit  % Truncation condition
            noise_matrix(i, :) = mu_k;
            break;
        end
    end
end

% Simulate system response (without noise)
for k = 1:N-1
    x(:, k+1) = Ad * x(:, k) + Bd * u(k, :)';
    y(:, k) = Cd * x(:, k) + Dd * u(k, :)';
end
y(:, N) = Cd * x(:, N) + Dd * u(N, :)';  % Last time step output
y_no_noise = y';  % Save noise-free data

% Add noise
y_noisy = y_no_noise + noise_matrix;

%% Initialization
T_ini = 1;        % Initial input/output data length
T_pred = 20;      % Prediction horizon
p_steps = 200;    % Total number of time steps
theta_amp = deg2rad(20);  % Amplitude
theta_freq = 0.1;         % Frequency

% Target angle
W = kron(eye(T_pred), diag([1, 1, 1, 1]));

% Optimization setup
options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
error = 0;

% Initialize input/output
u_opt_all = [];
y_opt_all = [];
y_new_all = [];
y_no_constraint_all = [];
y_no_noise_all = [];

u_ini = repmat(zeros(m,1), T_ini, 1);
y_ini = repmat(zeros(p,1), T_ini, 1);

% Build Hankel matrix (noise-free data)
U_d_no_noise = build_hankel(u', T_ini, T_pred);
Y_d_no_noise = build_hankel(y_no_noise', T_ini, T_pred);
U_p_no_noise = U_d_no_noise(1:m*T_ini, :);
U_f_no_noise = U_d_no_noise(m*T_ini+1:end, :);
Y_p_no_noise = Y_d_no_noise(1:p*T_ini, :);
Y_f_no_noise = Y_d_no_noise(p*T_ini+1:end, :);

% Build Hankel matrix (noisy data)
U_d = build_hankel(u', T_ini, T_pred);
Y_d = build_hankel(y_noisy', T_ini, T_pred);
U_p = U_d(1:m*T_ini, :);
U_f = U_d(m*T_ini+1:end, :);
Y_p = Y_d(1:p*T_ini, :);
Y_f = Y_d(p*T_ini+1:end, :);

x_new = zeros(4, p_steps);
target_angle_all = [];

Q = W' * W;
H_combined = [U_p; Y_p; U_f];
H = 2 * (Y_f' * Q * Y_f + 0.00001 * (U_f' * U_f));
H = (H + H') / 2;
f = @(target_angle) -2 * (target_angle' * Q * Y_f)';

%% Recursive DeePC (with noise and convex combination constraint)
for t_step = 1:p_steps
    theta_ref = theta_amp * square(2 * pi * theta_freq * t(t_step));
    target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
    target_angle_all = [target_angle_all; [theta_ref, 0, 0, 0]];
    
    % Equality constraint
    Aeq = [U_p; Y_p; ones(1, size(Y_p, 2))];
    beq = [u_ini(:); y_ini(:); 1];
    
    % Inequality constraint
    A1 = [-eye(size(U_f, 2)); eye(size(U_f, 2))];
    b1 = [zeros(size(U_f, 2), 1); ones(size(U_f, 2), 1)];
    
    % Solve optimization problem
    [g_opt, fval] = quadprog(H, f(target_angle), A1, b1, Aeq, beq, [], [], [], options);
    
    % Compute optimal input sequence
    u_opt = U_f * g_opt;
    y_opt = Y_f * g_opt;
    
    % Apply input and simulate response
    u_opt_all = [u_opt_all; u_opt(1:m)'];
    y_opt_all = [y_opt_all; y_opt(1:p)'];

    x_new(:, t_step+1) = Ad * x_new(:, t_step) + Bd * u_opt(1:m)';
    y_new = Cd * x_new(:, t_step) + Dd * u_opt(1:m)';
    y_new_all = [y_new_all; y_new'];
    
    % Update initial input/output
    u_ini = [u_ini(m+1:end); u_opt(1:m)'];
    y_ini = [y_ini(p+1:end); y_new];
end

%% Without convex combination constraint (with noise)
x_no_constraint = zeros(4, p_steps);
u_ini = repmat(zeros(m, 1), T_ini, 1);
y_ini = repmat(zeros(p, 1), T_ini, 1);

H_no_constraint = 2 * (Y_f' * Q * Y_f);
H_no_constraint = (H_no_constraint + H_no_constraint') / 2;

for t_step = 1:p_steps
    theta_ref = theta_amp * square(2 * pi * theta_freq * t(t_step));
    target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
    
    Aeq = [U_p; Y_p];
    beq = [u_ini(:); y_ini(:)];
    
    [g_opt, fval] = quadprog(H_no_constraint, f(target_angle), [], [], Aeq, beq, [], [], [], options);
    
    u_opt = U_f * g_opt;
    y_opt = Y_f * g_opt;
    
    y_no_constraint_all = [y_no_constraint_all; y_opt(1:p)'];
    
    x_no_constraint(:, t_step+1) = Ad * x_no_constraint(:, t_step) + Bd * u_opt(1:m)';
    y_new_no_constraint = Cd * x_no_constraint(:, t_step) + Dd * u_opt(1:m)';
    
    u_ini = [u_ini(m+1:end); u_opt(1:m)'];
    y_ini = [y_ini(p+1:end); y_new_no_constraint];
    y_new_no_constraint_all = [y_new_no_constraint_all; y_new_no_constraint'];
end

%% Without convex combination constraint (noise-free)
x_no_noise = zeros(4, p_steps);
u_ini = repmat(zeros(m, 1), T_ini, 1);
y_ini = repmat(zeros(p, 1), T_ini, 1);

H_no_noise = 2 * (Y_f_no_noise' * Q * Y_f_no_noise);
H_no_noise = (H_no_noise + H_no_noise') / 2;

for t_step = 1:p_steps
    theta_ref = theta_amp * square(2 * pi * theta_freq * t(t_step));
    target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
    
    Aeq = [U_p_no_noise; Y_p_no_noise];
    beq = [u_ini(:); y_ini(:)];
    
    [g_opt, fval] = quadprog(H_no_noise, f(target_angle), [], [], Aeq, beq, [], [], [], options);
    
    u_opt = U_f_no_noise * g_opt;
    y_opt = Y_f_no_noise * g_opt;
    
    y_no_noise_all = [y_no_noise_all; y_opt(1:p)'];
    
    x_no_noise(:, t_step+1) = Ad * x_no_noise(:, t_step) + Bd * u_opt(1:m)';
    y_new_no_noise = Cd * x_no_noise(:, t_step) + Dd * u_opt(1:m)';
    
    u_ini = [u_ini(m+1:end); u_opt(1:m)'];
    y_ini = [y_ini(p+1:end); y_opt(1:p)'];
end

%% Plot results
figure;
plot(0:dt:(length(y_opt_all)-1)*dt, target_angle_all(:, 1), '--', 'LineWidth', 1, 'DisplayName', 'Reference');
hold on;
plot(0:dt:(length(y_opt_all)-1)*dt, y_new_all(:, 1), 'LineWidth', 1, 'DisplayName', 'With Convex Combination');
plot(0:dt:(length(y_opt_all)-1)*dt, y_new_no_constraint_all(:, 1), 'Color', [0.4660, 0.6740, 0.1880], 'LineWidth', 1, 'DisplayName', 'Without Convex Combination');
plot(0:dt:(length(y_opt_all)-1)*dt, y_no_noise_all(:, 1), '--', 'Color', [0.4940, 0.1840, 0.8560], 'LineWidth', 1, 'DisplayName', 'Nominal Deterministic');
xlabel('Time (seconds)');
ylabel('Output 1 (radians)');
legend();

% Hankel matrix construction function
function H = build_hankel(data, T_ini, T_pred)
    [num_data, num_samples] = size(data);
    H = zeros(num_data * (T_ini + T_pred), num_samples - T_ini - T_pred + 1);
    for i = 1:(num_samples - T_ini - T_pred + 1)
        H(:, i) = reshape(data(:, i:i + T_ini + T_pred - 1), [], 1);
    end
end
