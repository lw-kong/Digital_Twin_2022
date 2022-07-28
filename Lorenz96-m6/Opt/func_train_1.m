function rmse = func_train_1(hyperpara_set,N)

%% config

eig_rho = hyperpara_set(1);
W_in_a = hyperpara_set(2);
a = hyperpara_set(3);
beta = 10^(-10);
k = hyperpara_set(4);
drive_W = hyperpara_set(5);
train_noise = 10^hyperpara_set(6);

Lorenz96_m = 6;
Lorenz96_F = 2;
driven_f = 2;

dim_ode = Lorenz96_m;
dim_drive = 1;

dim_in = dim_ode + dim_drive;
dim_out = dim_ode;





reservoir_tstep = 0.025;
ratio_tstep = 2;

train_r_step_cut = round( 2500 / reservoir_tstep );
train_r_step_length = round( 1600 /reservoir_tstep );
validate_r_step_length = round( 12 /reservoir_tstep );
len_washout = round( 100 /reservoir_tstep );


para_train_set = [2.2 2.8 3.4];
tp_train_length = length(para_train_set);



tmax_timeseries_train = (train_r_step_cut + train_r_step_length + validate_r_step_length + 20) * reservoir_tstep; % time, for timeseries
%% preparing training data

train_data_length = train_r_step_length + validate_r_step_length + 10;
train_data = zeros(tp_train_length, train_data_length,dim_in); % data that goes into reservior_training
for tp_i = 1:tp_train_length
    driven_a = para_train_set(tp_i);  %% system sensitive
    
    ts_train = NaN;
    while  sum(sum(isnan(ts_train))) %%
        x0 = 50*randn(Lorenz96_m,1);
        [t,ts_train] = ode4(@(t,x) eq_Lorenz96_driven_sin(t,x,Lorenz96_F,driven_a,driven_f),...
            0:reservoir_tstep/ratio_tstep:tmax_timeseries_train,x0);
    end
    ts_drive = drive_W * driven_a * sin(driven_f * t');
    
    ts_train = [ts_train, ts_drive];   
    ts_train = ts_train(1:ratio_tstep:end,:);
    for dim_i = 1:dim_ode % normalize
        ts_train(:,dim_i) = (ts_train(:,dim_i) - 1) / 1.5;
    end
    ts_train = ts_train(train_r_step_cut+1:end,:); % cut
    
    train_data(tp_i,:,:) = ts_train(1:train_data_length,:);
end

%% train
flag_r_train = [N k eig_rho W_in_a a beta train_r_step_cut train_r_step_length validate_r_step_length...
    reservoir_tstep dim_ode dim_drive len_washout];
[rmse,~,~,~,~,~,~] = ...
    func_STP_drive_22_train_noise(train_data,tp_train_length,flag_r_train,2,1,1,0,train_noise);

