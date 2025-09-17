%[R1].Z. Wei, W. Yuan, S. Li, J. Yuan and D. W. K. Ng, "Off-Grid Channel Estimation With Sparse Bayesian Learning for OTFS Systems," in IEEE Transactions on Wireless Communications, vol. 21, no. 9, pp. 7407-7426, Sept. 2022, doi: 10.1109/TWC.2022.3158616.

clc
close all
clear

%% OTFS Framework Design

variant='RCP'; %OTFS variant: (RZP / RCP / CP / ZP)


%% Dealy-Doppler Domain Grid Design
% number of Doppler bins (time slots)
N = 32;
% number of delay bins (subcarriers)
M = 32;
% subcarrier spacing
delta_f = 15e3;
% block duration
T = 1 / delta_f;
% carrier frequency
fc = 3e9;  % 4 GHz
% speed of light
c = 299792458;  % m/s
car_fre = 3*10^9;% Carrier frequency
% OTFS grid delay and Doppler resolution
delay_resolution = 1 / (M * delta_f);  % Delay resolution
doppler_resolution = 1 / (N * T);      % Doppler resolution

% M_mod: size of QAM constellation
M_mod = 4;
M_bits = log2(M_mod);
% average energy per data symbol
eng_sqrt = (M_mod==2)+(M_mod~=2)*sqrt((M_mod-1)/6*(2^2));

%% delay-Doppler grid symbol placement + Pilot symbols design(导频图案设计)

%区分：保护区域/数据/导频，思路：先确定保护区域—再区分数据或者导频
% 步骤1：先划分保护区域（ZP/CP）
if(strcmp(variant,'ZP'))

    length_ZP = M/16; % ZP保护区域长度（时延维度）
    length_CP = 0;
    prot_rows = (M - length_ZP + 1) : M;  % ZP保护区域的行索引

elseif(strcmp(variant,'CP'))

    length_ZP = 0;
    length_CP = M/16; % CP长度（不占用DD域网格，仅在时域添加）
    prot_rows = [];  % CP不占用DD域网格，保护区域为空

else
    length_ZP = 0;
    length_CP = 0;
    prot_rows = [];

end

M_data = M - length_ZP;  % 有效区域
L_prot=0; % CP/ZP需要设置参数为length_ZP/length_CP

% 步骤2：初始化网格（0=保护区域，1=有效区域待分配）
grid = zeros(M, N);
grid(1:M_data, :) = 1;

% 单导频

k_p = floor(N/2) + 1;
l_p = floor(M/2) + 1;

k_max = 3;
l_max = 4;

% guard zone placement
prot_k_range = k_p - 2*k_max : k_p + 2*k_max;
prot_l_range = l_p - l_max : l_p + l_max;


for l = prot_l_range
    for k = prot_k_range
        if l == l_p && k == k_p
            grid(l, k) = 2;
        else
            grid(l, k) = 0;
        end
    end
end


prot_grid = (grid == 0);
pilot_grid = (grid == 2);
data_grid = (grid == 1);

% 多个嵌入式导频(略)

% 符号数量计算
num_pilots = sum(pilot_grid(:));         % 导频符号数
num_data = sum(data_grid(:));            % 数据符号数
num_prot = sum(prot_grid(:));            % 保护区域符号数（全0）
N_syms_perfram = num_data;
N_bits_perfram = num_data * M_bits;


% SNR and variance of the noise
% SNR = P/\sigma^2; P: avg. power of albhabet transmitted

SNR_dB = 10:10:10;
SNR = 10.^(SNR_dB/10);
sigma_2 = (abs(eng_sqrt)^2)./SNR;

%% Initializing simulation error count variables

N_fram = 1;% N_fram_test = 1

%% Normalized DFT matrix

Fn=dftmtx(N);  % Generate the DFT matrix
Fn=Fn./norm(Fn);  % normalize the DFT matrix

%% Main Loop

for iesn0 = 1:length(SNR_dB)
    for ifram = 1:N_fram

        % random input bits generation
        trans_into_bit = randi([0,1],N_bits_perfram,1);
        % 2D QAM symbols generation
        data_symbols = qammod(reshape(trans_into_bit,M_bits,N_syms_perfram),M_mod,'gray','InputType','bit');

        % 生成导频符号（已知QAM符号，与数据同星座）
        pilot_bits = randi([0,1], log2(M_mod), num_pilots);  % 1个导频→2比特
        pilot_symbols = qammod(pilot_bits, M_mod, 'gray', 'InputType', 'bit');
        pilot_symbols = pilot_symbols(:);
        pilot_power_gain = 1;  % 10dB对应的电压增益
        pilot_symbols = pilot_symbols * pilot_power_gain;

        % %测试：数据符号全设为0（无需随机比特）
        % data_symbols = zeros(num_data, 1);  % 数据符号全0（列向量）
        % %测试：导频符号全设为1（复数1，与4QAM星座兼容）
        % pilot_symbols = ones(num_pilots, 1);  % 导频符号全1（列向量）

        % tips：先将其转换为数据流，再装填到DD网格中
        X = Generate_2D_pilot_data_grid(N, M, pilot_symbols, pilot_grid, data_symbols, data_grid);

        % %  OTFS modulation method1
        % s = OTFS_modulation(N,M,X);

        %  OTFS modulation method2
        X_tilda=X*Fn';                     %equation (2) in [R1]
        s = reshape(X_tilda,N*M,1);        %equation (4) in [R1]

        % Channel Parameters Design
        taps = 5;
        l_max = 4;
        k_max = 3;
        chan_coef= 1/sqrt(2)*(randn(1,taps)+1i.*randn(1,taps));

        % % G和gs的测试代码：
        delay_taps = [0,1,2,3,4];
        delay_spread = max(delay_taps);
        % doppler_taps = [-2.55 , -1.54 , 0.64 , 1.39 , 2.98];
        % % test
        doppler_taps = [-2,-1,0,1,2];
        % doppler_taps = [-2.5,-1.5,-1,1.5,2.5];

        for i = 1:taps
            tap = doppler_taps(i);
            % 直接取floor作为整数部分（与信道生成时的原始值对应）
            k_int(i) = floor(tap);  % 不再根据小数部分调整整数部分
            % 离格分量 = 原始值 - 整数部分（严格等于信道生成时的离格部分）
            kappa(i) = tap - k_int(i);  % 无 rounding，保持精确
        end

        % 时域信道输出计算
        [gs G]=Gen_time_domain_channel_OTFSvariants(N,M,delay_taps,doppler_taps,chan_coef,L_prot,variant);
        % % 可视化 h（k,l）& h（t，v）
        % visualize_channel_responses(N, M, taps, delay_taps, doppler_taps, chan_coef);
        % %  method1:TDL
        r = Gen_channel_output_OTFSvariants(N, M, delay_taps, gs, s, sigma_2(iesn0), variant, L_prot);

        % % method2；use Matrix G
        % r_G = G * s;
        % % noise = sqrt(sigma_2(iesn0)/2) * (randn(size(s)) + 1i*randn(size(s)));
        % % r = r + noise;
        % err=r-r_G;
        % err_DD = sum(sum(abs(err)))  %只有时延抽头为【不重复的时候，两种计算等价】

        %说明；G的生成逻辑：所有可能的时延抽头，r仅遍历随机生成的delay_taps，若delay_taps存在重复值，或未覆盖0 : delay_spread的所有整数，则必然导致误差。
        %% OTFS Demodulation

        Y_tilda=reshape(r,M,N);
        Y = Y_tilda*Fn; % 解调后得到DD域信道

        %% DD域信道矩阵

        % 1. 构建行-列交织矩阵P
        P = zeros(N*M, N*M);  % P的维度：(N*M)×(N*M)，与时域样本总数一致
        for j = 1:N  % j：DD域多普勒维度索引（对应时间符号数N）
            for i = 1:M  % i：DD域时延维度索引（对应子载波数M）
                E = zeros(M, N);  % 生成M×N的零矩阵E（作为P的子块）
                E(i, j) = 1;      % 将E的第(i,j)位置设为1，形成“单位矩阵块”（仅(i,j)为1，其余为0）
                P((j-1)*M + 1 : j*M, (i-1)*N + 1 : i*N) = E;
            end
        end

        % 2. 将时域信道矩阵G转换为时延-时间域信道矩阵H_tilda
        H_tilda = (P' * G * P);

        % 3. 将时延-时间域信道矩阵H_tilda转换为DD域信道矩阵H_DD
        H_DD = kron(eye(M), Fn) * (P' * G * P) * kron(eye(M), Fn');

        % % 4. 可视化DD域信道矩阵H_DD的幅度（观察信道在DD域的稀疏性，OTFS信道通常是稀疏的）
        % mesh(abs(H_DD))

        % 5. 验证DD域输入输出关系：Y1 = H_DD * X1 是否成立（误差应接近0）
        Y1 = reshape(Y.' , [], 1);  % 按“多普勒维度（行）→时延维度（列）”读取，匹配H_DD的索引
        % X.'：同理，对DD域发送矩阵X转置后reshape为列向量X1
        X1 = reshape(X.' , [], 1);
        % 计算理论接收符号（H_DD*X1）与实际接收符号（Y1）的误差
        error_Y = Y1 - H_DD * X1;
        % 计算误差的绝对值和（验证域转换正确性，若接近0则说明H_DD建模准确）
        sum(sum(abs(error_Y))) % 趋于0

        %% 信号在DD域的输出补偿验证计算(DD域符号输入输出关系)

        h_hat = zeros(M, N);
        N_i = N / 2;  % 多普勒扩展补偿范围
        z = exp(1i * 2 * pi / (N * M));  % 复指数基（与信道生成的z一致）
        taps = length(delay_taps);

        for m = 1:M                     % 接收端时延维度
            for n = 1:N                 % 接收端多普勒维度
                for i = 1:taps            % 遍历多径
                    g_i = chan_coef(i);
                    l_i = delay_taps(i) + 1;  % 时延抽头（+1适应MATLAB索引）
                    k_i = k_int(i);     % 整数部分（与信道生成的整数部分一致）
                    kv = kappa(i);      % 离格分量（与信道生成的离格部分一致）
                    % 此时 k_i + kv = doppler_taps(i)（与信道生成严格一致）

                    % 多普勒扩展补偿
                    for q = -N_i : N_i - 1  % k-k'∈（-16，15）
                        % 计算加权系数alpha（与参考程序一致）
                        if -q - kv == 0
                            beta = N;
                        else
                            beta = (exp(-1i * 2 * pi * (-q - kv)) - 1) / (exp(-1i * 2 * pi / N * (-q - kv)) - 1);
                        end
                        alpha = beta / N;

                        % 多普勒索引映射（与信道生成的循环逻辑对齐）
                        n_idx = mod(n - k_i - 1 + q, N) + 1;

                        % 时延索引映射与相位补偿（严格匹配信道生成的相位计算）
                        if m >= l_i
                            m_idx = m - l_i + 1;
                            % 相位 = z^((k_i + kv)*(m - l_i)) = z^(doppler_taps(i)*(m - l_i))
                            % 与信道生成的gs相位项 z^(k_i*(q - l_i)) 逻辑一致
                            phase = z^((k_i + kv) * (m - l_i));
                            h_hat(m, n) = h_hat(m, n) + g_i * phase * alpha * X(m_idx, n_idx);

                        elseif m < l_i && l_i <= M + m
                            m_idx = mod(m - l_i, M) + 1;
                            phase = z^((k_i + kv) * (m - l_i)) * exp(-1i * 2 * pi * (n_idx - 1) / N);
                            h_hat(m, n) = h_hat(m, n) + g_i * phase * alpha * X(m_idx, n_idx);

                        else
                            m_idx = mod(m - l_i, M) + 1;
                            phase = z^((k_i + kv) * (m - l_i)) * exp(-1i * 4 * pi * (n_idx - 1) / N);
                            h_hat(m, n) = h_hat(m, n) + g_i * phase * alpha * X(m_idx, n_idx);
                        end
                    end
                end
            end
        end

        % 验证误差
        error_Y_s = h_hat - Y;
        total_error = sum(abs(error_Y_s(:)));
        fprintf('error: %.4e\n', total_error);

        % DD domain 补偿输出正常

        %% A. Channel Estimation Problem Formulation：将OTFS信道估计建模为1D离格稀疏信号恢复（SSR）问题，分离在格/离格分量


        % % 1. 截断区域设置

        l_grid = l_p : 1 : l_p + l_max; %  范围（Kp ± kmax，时延l_p+l_max）对应[R1]中公式 15 的T集合
        k_grid = k_p - k_max : 1 : k_p + k_max;
        k_start = k_p-k_max;
        k_end = k_p + k_max;
        l_start = l_p;
        l_end = l_p + l_max;
        MT = length(l_grid);
        NT = length(k_grid);
        % T集合可看作是导频输出信号的完备集，故使用T集合能够正确估计
        % 截断区域：MT*NT

        %%  First-Order Linear Approximation
        %  基于虚拟网格的SSR问题
        %  off-grid问题：在虚拟采样网格上，通过泰勒展开分离信道路径参数的 “格点分量”（虚拟网格点）和 “离格分量”（网格偏差）

        % % 2.建立虚拟网格

        delta_l_grid = 1;  % 延迟网格分辨率
        l_virtual = 0:delta_l_grid:l_max;
        L_virt = length(l_virtual);

        % doppler off grid
        delta_k_grid = 0.5;  % 多普勒网格分辨率（小于1，捕捉离格分量）
        k_virtual = -k_max:delta_k_grid:k_max;
        K_virt = length(k_virtual);

        % 虚拟网格总点数
        grid_total = L_virt * K_virt;

        %% 1.Phi_T

        Phi_T = zeros(MT * NT, grid_total);
        % get pilot symbols
        % 提取导频符号（假设x是延迟-多普勒域的发送符号矩阵）
        x_pilot = X(l_p, k_p);
        col_idx = 1; % 第一列：代表虚拟格点（1，1）-（1，2）固定delay
        % 遍历虚拟网格（延迟方向 -> 多普勒方向）

        for l_virt_idx = 1:L_virt
            l_tau = l_virtual(l_virt_idx);  % 当前虚拟延迟偏移
            for k_virt_idx = 1:K_virt
                k_nu = k_virtual(k_virt_idx);  % 当前虚拟多普勒偏移

              
                % 初始化当前虚拟点对应的列向量
                phi_col = zeros(MT * NT, 1);

                % 发射符号
                for l_idx = 1:MT
                    l = l_grid(l_idx);  % 截断区域内的延迟索引
                    % 遍历截断区域内的所有(k, l)
                    for k_idx = 1:NT
                        k = k_grid(k_idx);  % 截断区域内的多普勒索引-k(receive)

                        % 计算偏移量
                        delta_k = k - k_p - k_nu;  % k(receive)-k(trans)-%真实的分数doppler偏移（未知）？—用虚拟网格上的候选偏移近似？？【误差大】
                        delta_l = l - l_p - l_tau;  % 延迟方向偏移（因无分数时延，应为整数）

                        % 输出端 k ∈（14-20）【截断区域】，输入k'=kp=17 得到
                        % k-kp=（-3，3）-还是直接用on grid的doppler分量
                        % 计算采样函数
                        if -(k - k_p) -k_nu == 0
                            beta = N;
                        else
                            numerator = exp(-1i * 2 * pi * (-(k - k_p)-k_nu)) - 1;
                            denominator = exp(-1i * 2 * pi * (-(k - k_p)-k_nu) / N) - 1;
                            beta = numerator / denominator;
                        end
                        % 计算alpha
                        alpha = beta / N;
                        % ？？采样函数用原始DD域参数还是截断区域的局部参数
                        % 先l后k [7次]
                        % 在此处补偿相位
                        linear_idx = (l_idx - 1) * NT + k_idx;
                        phi_col(linear_idx) = X(l_p,k_p) * alpha; % (l,k)发射符号的位置为：（l_p,k_p）还是（l,k）
                    end
                end
                % 填入观测矩阵
                Phi_T(:, col_idx) = phi_col;
                col_idx = col_idx + 1;
            end
        end
        
        
        %% 2. Phi_Prime_doppler

        Phi_Prime_doppler = zeros(MT * NT, grid_total);
        kappa_i = zeros(1, grid_total);                  % 离网格偏差
        col_idx = 1;

        for l_virt_idx = 1:L_virt
            l_tau = l_virtual(l_virt_idx);
            for k_virt_idx = 1:K_virt
                % 获取当前的doppler_fratical_parameters
                k_nu = 0;
                phi_prime_col = zeros(MT * NT, 1);
                for l_idx = 1:MT
                    l = l_grid(l_idx);
                    for k_idx = 1:NT
                        k = k_grid(k_idx);

                        delta = -(k - k_p) - k_nu;
                        if delta == 0  

                            beta_prime = 1i * pi * N * (N - 1); 

                        else                   

                            A = exp(-1i * 2 * pi * delta) - 1;
                            B = exp(-1i * 2 * pi * delta / N) - 1;                   
                            A_prime = -1i * 2 * pi * exp(-1i * 2 * pi * delta);
                            B_prime = -1i * 2 * pi / N * exp(-1i * 2 * pi * delta / N);
                            beta_prime = (A_prime * B - A * B_prime) / (B ^ 2);
                        end

                        alpha_prime = beta_prime / N;
                        linear_idx = (l_idx - 1) * NT + k_idx;
                        phi_prime_col(linear_idx) = x_pilot * alpha_prime;
                        
                    end
                end
                Phi_Prime_doppler(:, col_idx) = phi_prime_col;
                col_idx = col_idx + 1;
            end
        end

        
        %% test model:信号重构误差验证（利用信道的真实参数）（存疑）

        h_DD = zeros(L_virt,K_virt);  %构建DD域的真实信道向量（MtNv*1）

        for i=1:taps
            g_i=chan_coef(i);
            l_i=delay_taps(i);
            % delay_taps ∈ （0，lmax） ，-最近的l_virtual
            [~, l_idx] = min(abs(l_virtual - l_i));
            k_i=doppler_taps(i);
            % doppler_taps ∈ （-kmax，kmax），-最近的k_virtual 
            [~, k_idx] = min(abs(k_virtual - k_i)); 
            h_DD(l_idx,k_idx)=g_i*exp(-1i*2*pi*k_i*l_i);
        end

        h_DD_vec = reshape(h_DD.', [], 1);
        % 1. 格上项：Phi_grid（Mt*Nv × grid_total） × h_real_hat（grid_total × 1）
        % y_grid = Phi_grid * h_real_hat;
        y_grid = Phi_T * h_DD_vec;

        % % 2. 多普勒离格修正项：Phi_k_prime × diag(kappa) × h_real_hat（kappa为路径对应标量，按网格索引匹配）
        % % 注：diag(kappa)将离格分量向量转为对角矩阵，确保每个网格点的修正仅作用于对应信道系数
        % y_k_corr = Phi_k_prime * (diag(kappa) * h_DD_vec);
        %
        %
        % % 4. 合成最终重构信号y_T（忽略噪声）
        % y_T_DD = y_grid + y_k_corr;

        %% 计算阶段区域的DD域理论补偿输出
        % Y:经信道后得到的DD域信号，截取MTNT区域的矩阵
        % 从DD域信号矩阵Y中提取MTNT区域的子矩阵
        Y_T_grid = Y(l_start:l_end, k_start:k_end);
        Y_T_grid_1 = h_hat(l_start:l_end, k_start:k_end);
        Y_T_vec = reshape(Y_T_grid, [], 1);
        errdd_1 = sum(abs(Y_T_vec-y_grid))
        errdd_2 = sum(sum(abs(Y_T_grid - Y_T_grid_1 )))

    end
end









