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

% State-space matrix definition
A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, (g*l_p^2*m_p^2*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r), -(B_r*J_p*R_m + J_p*K_m*K_t + B_r*R_m*l_p^2*m_p + K_m*K_t*l_p^2*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_p*l_p*m_p*r)/(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r);
     0, -(R_m*g*l_p*m_p^2*r^2 + J_r*R_m*g*l_p*m_p)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), (B_r*R_m*l_p*m_p*r + K_m*K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)), -(B_p*R_m*m_p*r^2 + B_p*J_r*R_m)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];
B = [0; 0; (K_t*m_p*l_p^2 + J_p*K_t)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r)); -(K_t*l_p*m_p*r)/(R_m*(J_r*m_p*l_p^2 + J_p*m_p*r^2 + J_p*J_r))];
C = eye(4);
D = zeros(4, 1);

% Discretize the state-space model
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
num_simulations = 100;

% Initialize storage
violations_method2 = zeros(num_simulations, 1);

% Range of lambda_u and lambda_y values
lambda_values = logspace(log10(0.1), log10(3000), 30);

% Initialize results storage
results = zeros(length(lambda_values), length(lambda_values));

% Iterate over lambda_u and lambda_y
for idx_u = 1:length(lambda_values)
    for idx_y = 1:length(lambda_values)
        lambda_u = lambda_values(idx_u);
        lambda_y = lambda_values(idx_y);
        
        % Run simulation
        total_errors = 0;
        for i = 1:num_simulations
            % Generate random input
            u = randn(p_steps, size(Bd, 2));
            
            % Initialize state and output
            x = zeros(size(Ad, 1), p_steps);
            y = zeros(p_steps, size(Cd, 1));
            
            % Initialize noise
            mu_bar = 0.01^2;
            cov_matrix = mu_bar * eye(size(Cd, 1));
            noise = zeros(p_steps, size(Cd, 1));
            
            % Generate bounded noise
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
            
            % Generate Hankel matrix
            U_d = build_hankel(u', T_ini, T_pred);
            Y_d = build_hankel(y', T_ini, T_pred);
            U_p = U_d(1:size(Bd, 2)*T_ini, :);
            U_f = U_d(size(Bd, 2)*T_ini+1:end, :);
            Y_p = Y_d(1:size(Cd, 1)*T_ini, :);
            Y_f = Y_d(size(Cd, 1)*T_ini+1:end, :);
            
            % Method 2
            errors = method2(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f, lambda_u, lambda_y);
            total_errors = total_errors + errors;
        end
        
        % Store results
        results(idx_u, idx_y) = total_errors / num_simulations;
    end
end
results = results - 0.01;

% Plot results
figure;
colormap(parula);
[X, Y] = meshgrid(lambda_values, lambda_values);
surf(Y, X, results, 'EdgeColor', 'none', 'FaceAlpha', 0.85);
colorbar;
grid on;
xlabel('$\lambda_u$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$\lambda_y$', 'Interpreter', 'latex', 'FontSize', 12);
zlabel('Average realized error');

hold on;
lambda_y_highlight = 40;
[~, idx] = min(abs(lambda_values - lambda_y_highlight));

h_line = plot3(lambda_values, lambda_y_highlight * ones(size(lambda_values)), results(:, idx), 'r', 'LineWidth', 3);

legend(h_line, '$\lambda_y = 40$', 'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
set(gca, 'FontSize', 12, 'GridColor', [0.5 0.5 0.5], 'GridAlpha', 0.6);
view(130, 30);

function errors = method2(Ad, Bd, Cd, Dd, dt, T_ini, T_pred, p_steps, theta_amp, theta_freq, W, mu_limit, U_p, U_f, Y_p, Y_f, lambda_u, lambda_y)
    m = size(Bd, 2);
    p = size(Cd, 1);
    
    % Initialization
    u_ini = repmat(zeros(m, 1), T_ini, 1);
    y_ini = repmat(zeros(p, 1), T_ini, 1);
    x_new = zeros(4, p_steps);
    Q = W' * W;
    lambda_ii = 3000;
    H_combined = [U_p; Y_p; U_f];
    Pi = pinv(H_combined) * H_combined;
    H = 2 * (Y_f' * Q * Y_f + 0.00001 * (U_f' * U_f) + lambda_u * (U_p' * U_p) + lambda_y * (Y_p' * Y_p) + lambda_ii * (eye(size(Pi)) - Pi)'*(eye(size(Pi)) - Pi));
    H = (H + H') / 2;
    errors = 0;
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
        errors = errors + ((y_new(1) -theta_ref).^2);
     end
    errors = errors / p_steps;
end

function H = build_hankel(data, T_ini, T_pred)
    [num_data, num_samples] = size(data);
    H = zeros(num_data * (T_ini + T_pred), num_samples - T_ini - T_pred + 1);
    for i = 1:(num_samples - T_ini - T_pred + 1)
        H(:, i) = reshape(data(:, i:i + T_ini + T_pred - 1), [], 1);
    end
end
