% System parameters from Table 1
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

% State-space matrices for the downward position
A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, (g*l_p^2*m_p^2*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r), -(B_r*J_p*R_m + J_p*K_m*K_t + B_r*R_m*l_p^2*m_p + K_m*K_t*l_p^2*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_p*l_p*m_p*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r);
     0, -(R_m*g*l_p*m_p^2*r^2 + J_r*R_m*g*l_p*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_r*R_m*l_p*m_p*r + K_m*K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), -(B_p*R_m*m_p*r^2 + B_p*J_r*R_m)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];
B = [0; 0; (K_t*m_p*l_p^2 + J_p*K_t)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)); -(K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];
C = eye(4);  % Identity matrix
D = [0; 0; 0; 0];
dt = 0.1;

% Discretize the state-space model
sys = ss(A, B, C, D);
sysd = c2d(sys, dt);

% Discrete system matrices
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

%% Simulate data generation
% Time settings
T_total = 30;    % Total simulation time
t = 0:dt:T_total;
N = length(t);   % Number of time steps

% Random input
m = size(B, 2);
p = size(C, 1);
u = randn(N, m);  % Random input sequence

% Initialize state and output
x = zeros(size(Ad, 1), N);  % State
y = zeros(p, N);  % Output

% Simulate discrete-time system response
for k = 1:N-1
    x(:, k+1) = Ad * x(:, k) + Bd * u(k, :)';
    y(:, k) = Cd * x(:, k) + Dd * u(k, :)';
end
y(:, N) = Cd * x(:, N) + Dd * u(N, :)';  % Output at final time step
y = y';

%% Initialization
T_ini = 1;       % Initial input/output data length
T_pred = 20;     % Prediction horizon
p_steps = 200;   % Number of time steps
theta_amp = deg2rad(20);  % Amplitude in radians
theta_freq = 0.1;  % Frequency

% Target angles
W = kron(eye(T_pred), diag([1, 1, 1, 1]));

% Optimization settings
options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');

% Initialize input/output
u_opt_all = [];
y_opt_all = [];
y_new_all = [];
u_ini = repmat(zeros(m,1), T_ini, 1);
y_ini = repmat(zeros(p,1), T_ini, 1);

% Construct Hankel matrix
U_d = build_hankel(u', T_ini, T_pred);
Y_d = build_hankel(y', T_ini, T_pred);
U_p = U_d(1:m*T_ini, :);
U_f = U_d(m*T_ini+1:end, :);
Y_p = Y_d(1:p*T_ini, :);
Y_f = Y_d(p*T_ini+1:end, :);

% Recursive DeePC
x_new = zeros(4, p_steps);
target_angle_all = [];
Q = W' * W;
H = 2 * (Y_f' * Q * Y_f + 0.00001 * (U_f' * U_f));
H = (H + H') / 2;
f = @(target_angle) -2 * (target_angle' * Q * Y_f)';

cost_all = [];  % Store cost for each time step

for t_step = 1:p_steps
    theta_ref = theta_amp * square(2 * pi * theta_freq * t(t_step));
    target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
    target_angle_all = [target_angle_all; [theta_ref, 0, 0, 0]];
    
    Aeq = [U_p; Y_p];
    beq = [u_ini(:); y_ini(:)];
    
    [g_opt, fval] = quadprog(H, f(target_angle), [], [], Aeq, beq, [], [], [], options);
    
    u_opt = U_f * g_opt;
    y_opt = Y_f * g_opt;
    
    % Store cost
    cost = fval;
    cost_all = [cost_all; cost];
    
    % Apply input to the system and simulate new response
    u_opt_all = [u_opt_all; u_opt(1:m)'];
    y_opt_all = [y_opt_all; y_opt(1:p)'];
    x_new(:, t_step+1) = Ad * x_new(:, t_step) + Bd * u_opt(1:m)';
    y_new = Cd * x_new(:, t_step) + Dd * u_opt(1:m)';
    u_ini = [u_ini(m+1:end); u_opt(1:m)'];
    y_ini = [y_ini(p+1:end); y_new];
    y_new_all = [y_new_all; y_new'];
end

%% Plot results
figure;
for i = 1:p
    subplot(3, 2, i);
    plot(0:dt:(length(y_opt_all)-1)*dt, y_new_all(:, i));
    hold on;
    plot(0:dt:(length(y_opt_all)-1)*dt, target_angle_all(:, i), '--');
    xlabel('Time (seconds)');
    ylabel(['Output ', num2str(i), ' (radians)']);
    legend(['Output ', num2str(i)], 'Reference');
    title(['DeePC Control of Inverted Pendulum - Output ', num2str(i)]);
end

% Plot cost curve
subplot(3, 2, 5:6);  % Occupy two slots
plot(0:dt:(length(cost_all)-1)*dt, cost_all, '-o');
xlabel('Time (seconds)');
ylabel('Cost');
title('Cost Over Time');
legend('Cost');

% Hankel matrix construction function
function H = build_hankel(data, T_ini, T_pred)
    [num_data, num_samples] = size(data);
    H = zeros(num_data * (T_ini + T_pred), num_samples - T_ini - T_pred + 1);
    for i = 1:(num_samples - T_ini - T_pred + 1)
        H(:, i) = reshape(data(:, i:i + T_ini + T_pred - 1), [], 1);
    end
end

