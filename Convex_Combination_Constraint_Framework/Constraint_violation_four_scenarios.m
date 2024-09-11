% Define system parameters
R_m = 8.4;
K_t = 0.042;
K_m = 0.042;
m_p = 0.024;
l_p = 0.0645;
J_p = 3.3282e-5;
J_r = 5.7198e-5;
B_p = 0.0005;
B_r = 0.0015;
r = 0.085;
g = 9.81;

% State-space matrices
A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, (g*l_p^2*m_p^2*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r), -(B_r*J_p*R_m + J_p*K_m*K_t + B_r*R_m*l_p^2*m_p + K_m*K_t*l_p^2*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_p*l_p*m_p*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r);
     0, -(R_m*g*l_p*m_p^2*r^2 + J_r*R_m*g*l_p*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_r*R_m*l_p*m_p*r + K_m*K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), -(B_p*R_m*m_p*r^2 + B_p*J_r*R_m)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];
B = [0; 0; (K_t*m_p*l_p^2 + J_p*K_t)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)); -(K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];
C = eye(4);
D = zeros(4, 1);

% Discretize the system
dt = 0.1;
sys = ss(A, B, C, D);
sysd = c2d(sys, dt);

Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

% Simulation parameters
T_ini = 1;
T_pred = 20;
p_steps = 200;
theta_amp = deg2rad(20);
theta_freq = 0.1;
W = kron(eye(T_pred), diag([1, 1, 1, 1]));
mu_limit = 0.01;
num_simulations = 10000;

% Initialize storage variables
violations_method1 = zeros(num_simulations, 1);
violations_method2 = zeros(num_simulations, 1);
violations_method3 = zeros(num_simulations, 1);
violations_method4 = zeros(num_simulations, 1);

for i = 1:num_simulations
    % Generate random input
    u = randn(p_steps, size(Bd, 2));
    
    % Initialize states and outputs
    x = zeros(size(Ad, 1), p_steps);
    y = zeros(p_steps, size(Cd, 1));
    
    % Initialize noise matrix
    mu_bar = 0.01^2; % Variance
    cov_matrix = mu_bar * eye(size(Cd, 1));
    noise = zeros(p_steps, size(Cd, 1));
    
    % Generate truncated noise
    for j = 1:p_steps
        while true
            mu_k = mvnrnd(zeros(1, size(Cd, 1)), cov_matrix);
            if max(abs(mu_k)) <= mu_limit
                noise(j, :) = mu_k;
                break;
            end
        end
    end
    
    % Simulate discrete system response
    for k = 1:p_steps-1
        x(:, k+1) = Ad * x(:, k) + Bd * u(k, :)';
        y(k, :) = Cd * x(:, k) + Dd * u(k, :)';
    end
    y(p_steps, :) = Cd * x(:, p_steps) + Dd * u(p_steps, :)';
    y = y + noise;
    
    % Build Hankel matrix
    U_d = build_hankel(u', T_ini, T_pred);
    Y_d = build_hankel(y', T_ini, T_pred);
    U_p = U_d(1:size(Bd, 2)*T_ini, :);
    U_f = U_d(size(Bd, 2)*T_ini+1:end, :);
    Y_p = Y_d(1:size(Cd, 1)*T_ini, :);
    Y_f = Y_d(size(Cd, 1)*T_ini+1:end, :);
    

    violations_method1(i) = method1(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f);
    violations_method2(i) = method2(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f);
    violations_method3(i) = method3(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f);
    violations_method4(i) = method4(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f);
end

% Calculate average constraint violations
avg_violations_method1 = mean(violations_method1);
avg_violations_method2 = mean(violations_method2);
avg_violations_method3 = mean(violations_method3);
avg_violations_method4 = mean(violations_method4);
disp(['Average violations - Method 1: ', num2str(avg_violations_method1)]);
disp(['Average violations - Method 2: ', num2str(avg_violations_method2)]);
disp(['Average violations - Method 3: ', num2str(avg_violations_method3)]);
disp(['Average violations - Method 4: ', num2str(avg_violations_method4)]);

% Define methods and average violations
methods = {'PDWTC', 'PDWOTC', 'CDWTC', 'CDWOTC'};
avg_violations = [avg_violations_method1, avg_violations_method2, avg_violations_method3, avg_violations_method4];

% Create bar chart
figure;
b = bar(avg_violations, 'FaceColor', 'flat');

% Set bar colors (IEEE colors)
b.CData(1, :) = [0, 0.4470, 0.7410]; % Blue
b.CData(2, :) = [0.8500, 0.3250, 0.0980]; % Orange
b.CData(3, :) = [0.9290, 0.6940, 0.1250]; % Yellow

% Add value labels
xtips = b.XEndPoints;
ytips = b.YEndPoints;
labels = string(b.YData);
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
% Set labels and title
ylabel('Average Violations');
set(gca, 'XTickLabel', methods);
% Adjust y-axis limits
ylim([0, max(avg_violations) + 1]);
% Show grid
grid on;
set(gca, 'Position', [0.13 0.11 0.775 0.815]);

function num_violations = method1(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f)
    m = size(Bd, 2);
    p = size(Cd, 1);
    
    u_ini = repmat(zeros(m, 1), T_ini, 1);
    y_ini = repmat(zeros(p, 1), T_ini, 1);
    x_new = zeros(4, p_steps);
    Q = W' * W;
    lambda_ii = 3000;
    H_combined = [U_p; Y_p; U_f];
    Pi = pinv(H_combined) * H_combined;
    H = 2 * (Y_f' * Q * Y_f + 0.00001 * (U_f' * U_f) + lambda_ii * (eye(size(Pi)) - Pi)'*(eye(size(Pi)) - Pi));
    H = (H + H') / 2;
    
    y_new_all = [];
    for t_step = 1:p_steps
        theta_ref = theta_amp * square(2 * pi * theta_freq * t_step * dt);
        target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
        f = -2 * (target_angle' * Q * Y_f)';
        Aeq = [U_p; Y_p];
        beq = [u_ini(:); y_ini(:)];
        A = kron(eye(T_pred * 2), [0,1,0,0;0,0,1,0;0,0,0,1]) * [Y_f; -Y_f];
        b = kron(ones(T_pred * 2, 1),[0.0284;0.5443;0.2274]) - mu_limit;
        options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
        [g_opt, ~] = quadprog(H, f, A, b, Aeq, beq, [], [], [], options);
        
        u_opt = U_f * g_opt;
        y_opt = Y_f * g_opt;
        
        u_ini = [u_ini(m+1:end); u_opt(1:m)];
        x_new(:, t_step+1) = Ad * x_new(:, t_step) + Bd * u_opt(1:m);
        y_new = Cd * x_new(:, t_step) + Dd * u_opt(1:m);
        y_ini = [y_ini(p+1:end); y_new];
        y_new_all = [y_new_all; y_new'];
    end
    
second_column = y_new_all(:, 2);
constraint_violations1 = abs(second_column) > 0.0284;
constraint_violations2=abs(y_new_all(:, 3)) > 0.5443;
constraint_violations3=abs(y_new_all(:, 4)) > 0.2274;
num_violations = sum(constraint_violations1)+sum(constraint_violations2)+sum(constraint_violations3);
end
function [num_violations,y_new_all] = method2(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f)
    m = size(Bd, 2);
    p = size(Cd, 1);
   
    u_ini = repmat(zeros(m, 1), T_ini, 1);
    y_ini = repmat(zeros(p, 1), T_ini, 1);
    x_new = zeros(4, p_steps);
    Q = W' * W;
    lambda_u = 50;
    lambda_y = 50;
    lambda_ii = 3000;
    H_combined = [U_p; Y_p; U_f];
    Pi = pinv(H_combined) * H_combined;
    H = 2 * (Y_f' * Q * Y_f + 0.00001 * (U_f' * U_f) + lambda_u * (U_p' * U_p) + lambda_y * (Y_p' * Y_p) + lambda_ii * (eye(size(Pi)) - Pi)'*(eye(size(Pi)) - Pi));
    H = (H + H') / 2;
    y_new_all = [];
    for t_step = 1:p_steps
        theta_ref = theta_amp * square(2 * pi * theta_freq * t_step * dt);
        target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
        f = -2 * (target_angle' * Q * Y_f)' - 2 * (lambda_u * U_p' * u_ini + lambda_y * Y_p' * y_ini);
        A = kron(eye(T_pred * 2), [0,1,0,0;0,0,1,0;0,0,0,1]) * [Y_f; -Y_f];
        b = kron(ones(T_pred * 2, 1),[0.0284;0.5443;0.2274]) - mu_limit;
        options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
        [g_opt, ~] = quadprog(H, f, A, b, [], [], [], [], [], options);
        
        u_opt = U_f * g_opt;
        y_opt = Y_f * g_opt;
        
        u_ini = [u_ini(m+1:end); u_opt(1:m)];
        x_new(:, t_step+1) = Ad * x_new(:, t_step) + Bd * u_opt(1:m);
        y_new = Cd * x_new(:, t_step) + Dd * u_opt(1:m);
        y_ini = [y_ini(p+1:end); y_new];
        y_new_all = [y_new_all; y_new'];
    end
    
second_column = y_new_all(:, 2);
constraint_violations1 = abs(second_column) > 0.0284;
constraint_violations2=abs(y_new_all(:, 3)) > 0.5443;
constraint_violations3=abs(y_new_all(:, 4)) > 0.2274;

num_violations = sum(constraint_violations1)+sum(constraint_violations2)+sum(constraint_violations3);
end
function num_violations = method3(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f)
    m = size(Bd, 2);
    p = size(Cd, 1);
    
    u_ini = repmat(zeros(m, 1), T_ini, 1);
    y_ini = repmat(zeros(p, 1), T_ini, 1);
    x_new = zeros(4, p_steps);
    Q = W' * W;
    lambda_ii = 3000;
    H_combined = [U_p; Y_p; U_f];
    Pi = pinv(H_combined) * H_combined;
    H = 2 * (Y_f' * Q * Y_f + 0.00001 * (U_f' * U_f) + lambda_ii * (eye(size(Pi)) - Pi)'*(eye(size(Pi)) - Pi));
    H = (H + H') / 2;
    
    y_new_all = [];
    for t_step = 1:p_steps
        theta_ref = theta_amp * square(2 * pi * theta_freq * t_step * dt);
        target_angle = repmat([theta_ref; 0; 0; 0], T_pred, 1);
        f = -2 * (target_angle' * Q * Y_f)';
        Aeq = [U_p; Y_p];
        beq = [u_ini(:); y_ini(:)];
        options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
        [g_opt, ~] = quadprog(H, f, [], [], Aeq, beq, [], [], [], options);
        
        u_opt = U_f * g_opt;
        y_opt = Y_f * g_opt;
        
        u_ini = [u_ini(m+1:end); u_opt(1:m)];
        x_new(:, t_step+1) = Ad * x_new(:, t_step) + Bd * u_opt(1:m);
        y_new = Cd * x_new(:, t_step) + Dd * u_opt(1:m);
        y_ini = [y_ini(p+1:end); y_new];
        y_new_all = [y_new_all; y_new'];
    end
    
second_column = y_new_all(:, 2);

constraint_violations1 = abs(second_column) > 0.0284;
constraint_violations2=abs(y_new_all(:, 3)) > 0.5443;
constraint_violations3=abs(y_new_all(:, 4)) > 0.2274;
num_violations = sum(constraint_violations1)+sum(constraint_violations2)+sum(constraint_violations3);
end
function H = build_hankel(data, T_ini, T_pred)
    [num_data, num_samples] = size(data);
    H = zeros(num_data * (T_ini + T_pred), num_samples - T_ini - T_pred + 1);
    for i = 1:(num_samples - T_ini - T_pred + 1)
        H(:, i) = reshape(data(:, i:i + T_ini + T_pred - 1), [], 1);
    end
end