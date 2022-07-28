
% chose one of the best bo training results
% different tp_set and ode_parameter_set
% warm up
% no return map validation
% control parameter is 1 dim

if ispc
    addpath '..\Functions'
else
    addpath '../Functions'
end
addpath 'Opt'

Lorenz96_m = 20;
Lorenz96_F = 2;
driven_f = 2;


dim_out = 2;
dim_couple = 3;
dim_global_drive = 1;

dim_drive = dim_couple + dim_global_drive;
dim_in = dim_out + dim_drive;


%
load('opt_Lorenz96_parallel_m20_11_20210918T001302_865.mat')
hyperpara_set = opt_result;
n_opt = n;
n = 1200;
eig_rho = hyperpara_set(1);
W_in_a = hyperpara_set(2);
a = hyperpara_set(3);
beta = 10^hyperpara_set(4);
k = round( hyperpara_set(5)/n_opt*n );
drive_W = hyperpara_set(6);
train_noise = 10^hyperpara_set(7);
%

reservoir_tstep = 0.025;
ratio_tstep = 2;



train_r_step_cut = round( 2500 / reservoir_tstep );
train_r_step_length = round( 3700 /reservoir_tstep );
validate_r_step_length = round( 100 /reservoir_tstep );
len_washout = round( 200 /reservoir_tstep );

bo = 5;


para_train_set = [1.8 2.2 2.6 3.0];
tp_train_length = length(para_train_set);


train_norm_set = zeros(tp_train_length,dim_out + dim_couple,2);
tmax_timeseries_train = (train_r_step_cut + train_r_step_length + validate_r_step_length + 20) * reservoir_tstep; % time, for timeseries
rng('shuffle');
tic;

rmse_min = 10000;
for bo_i = 1:bo
    fprintf('preparing training data...\n');
    
    train_data_length = train_r_step_length + validate_r_step_length + 10;
    train_data = zeros(tp_train_length, train_data_length,dim_in); % data that goes into reservior_training
    for tp_i = 1:tp_train_length
        driven_a = para_train_set(tp_i);

        ts_train = NaN;
        while  sum(sum(isnan(ts_train))) %%
            x0 = 50*randn(Lorenz96_m,1);
            [t,ts_train] = ode4(@(t,x) eq_Lorenz96_driven_sin(t,x,Lorenz96_F,driven_a,driven_f),...
                0:reservoir_tstep/ratio_tstep:tmax_timeseries_train,x0);
        end
        ts_drive = drive_W * driven_a * sin(driven_f * t');
        
        ts_train = [ts_train, ts_drive];
        
        ts_train = ts_train(:,[3,4,1,2,5,end]);
        ts_train = ts_train(1:ratio_tstep:end,:);
        %
        for dim_i = 1:dim_out+dim_couple % normalize
            %train_norm_set(tp_i,dim_i,1) = mean(ts_train(:,dim_i));
            %train_norm_set(tp_i,dim_i,2) = std(ts_train(:,dim_i));
            %ts_train(:,dim_i) = (ts_train(:,dim_i) - train_norm_set(tp_i,dim_i,1)) ...
            %    / train_norm_set(tp_i,dim_i,2);
            ts_train(:,dim_i) = (ts_train(:,dim_i) - 1) / 1.5;
        end
        %
        ts_train = ts_train(train_r_step_cut+1:end,:); % cut       
        
        train_data(tp_i,:,:) = ts_train(1:train_data_length,:);
    end
    
    % train
    fprintf('training...\n');
    flag_r_train = [n k eig_rho W_in_a a beta train_r_step_cut train_r_step_length validate_r_step_length...
        reservoir_tstep dim_out dim_drive len_washout];
    [rmse,W_in_temp,res_net_temp,P_temp,t_validate_temp,x_real_temp,x_validate_temp] = ...
        func_STP_drive_22_train_noise(train_data,tp_train_length,flag_r_train,2,1,1,0,train_noise);
    fprintf('attempt rmse = %f\n',rmse)
    
    if rmse < rmse_min
        W_in = W_in_temp;
        res_net = res_net_temp;
        P = P_temp;
        t_validate = t_validate_temp;
        x_real = x_real_temp;
        x_validate = x_validate_temp;        
        rmse_min = rmse;
    end
    
    fprintf('%f is done\n',bo_i/bo)
    toc;
end

fprintf('best rmse = %f\n',rmse_min)

% plotting validating result
plot_dim = 1; % change the ylabel
for tp_i = 1:tp_train_length
    figure('Name','Reservoir Predict');
    subplot(2,1,1)
    hold on
    plot(t_validate,x_real(tp_i,:,plot_dim));
    plot(t_validate,x_validate(tp_i,:,plot_dim),'--');
    xlabel('time');
    ylabel('x');
    title(['tp = ' num2str( para_train_set(tp_i),6 )]);
    set(gcf,'color','white')
    box on
    hold off
    subplot(2,1,2)
    hold on
    plot(t_validate,abs(x_validate(tp_i,:,plot_dim)-x_real(tp_i,:,plot_dim))/...
        ( max(x_real(tp_i,:,plot_dim)) - min(x_real(tp_i,:,plot_dim)) ) )
    line([t_validate(1) t_validate(end)],[0.05 0.05])
    xlabel('time');
    ylabel('relative error');
    box on
    hold off
end


% plotting training data
figure('Name','Training Data','Position',[50 50 480 390]);
for tp_i = 1:tp_train_length
    
    subplot(2,2,tp_i)
    plot(train_data(tp_i,:,1),train_data(tp_i,:,2));
    axis([-10,15,-10,15])
    xlabel('x1');
    ylabel('x2');
    title(['training data at' newline 'tp = ' num2str( para_train_set(tp_i),8 )]);
    set(gcf,'color','white')
    
end
