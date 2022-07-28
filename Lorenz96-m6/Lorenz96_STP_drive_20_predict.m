addpath('\klw\Research\Functions');
addpath('\klw\Research\Reservoir\PredictBehaviorUnderDiffPara\STP_after\DigitalTwin');


warmup_r_step_cut = round( 2500 /reservoir_tstep );
warmup_r_step_length = round( 20 / reservoir_tstep );

predict_r_step_cut = round( 200 /reservoir_tstep );
predict_r_step_length = round( 100 / reservoir_tstep );


driven_a_predict = 2.8;
driven_a_warmup = min(para_train_set);

tmax_timeseries_warmup = (warmup_r_step_cut + warmup_r_step_length + 5 ) * reservoir_tstep;
tmax_timeseries_predict = (warmup_r_step_cut + warmup_r_step_length +...
    predict_r_step_cut + predict_r_step_length + 5 ) * reservoir_tstep;
rng('shuffle');
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% warm up
ts_warmup = NaN;
while sum(sum(isnan(ts_warmup)))
    x0 = 50*randn(Lorenz96_m,1);
    [t,ts_warmup] = ode4(@(t,x) eq_Lorenz96_driven_sin(t,x,Lorenz96_F,driven_a_warmup,driven_f),...
        0:reservoir_tstep/ratio_tstep:tmax_timeseries_predict,x0);
end
ts_drive = drive_W * driven_a_predict * sin(driven_f * t');
% 'ts_drive' is the driving signal in both the warming up phase and the
% prediction phase. It is not the real driving signal for the warming up
% time series, but that does not affect the performace.


t = t(1:ratio_tstep:end);
ts_warmup = ts_warmup(1:ratio_tstep:end,:);
ts_drive = ts_drive(1:ratio_tstep:end,:);

t = t(warmup_r_step_cut+1 : end);
ts_warmup = ts_warmup(warmup_r_step_cut+1 : end,:);
ts_drive = ts_drive(warmup_r_step_cut+1 : end,:);

ts_warmup = [ts_warmup, ts_drive];
%{
for dim_i = 1:dim_ode % normalize
    ts_warmup(:,dim_i) = (ts_warmup(:,dim_i) - train_norm_set(1,dim_i,1)) ...
        / train_norm_set(1,dim_i,2);
end
%}
for dim_i = 1:dim_ode % normalize
    ts_warmup(:,dim_i) = (ts_warmup(:,dim_i) - 1) / 1.5;
end

%% real
ts_predict_real = NaN;
while  sum(sum(isnan(ts_predict_real)))
    x0 = 50*randn(Lorenz96_m,1);
    [t,ts_predict_real] = ode4(@(t,x) eq_Lorenz96_driven_sin(t,x,Lorenz96_F,driven_a_predict,driven_f),...
        0:reservoir_tstep/ratio_tstep:tmax_timeseries_predict,x0);
end
ts_predict_real = ts_predict_real(1:ratio_tstep:end,:);
ts_predict_real = ts_predict_real(warmup_r_step_cut+1 : end,:);




%% predict
flag_r = [n a warmup_r_step_length predict_r_step_cut predict_r_step_length dim_ode dim_drive reservoir_tstep];
predict_r = func_STP_drive_20_predict(ts_warmup,W_in,res_net,P,flag_r);
%[predict_r,lya_timeset,lya_spectra] = func_STP_drive_20_predict_Lya(ts_warmup,W_in,res_net,P,flag_r);
% back-normalize
%for dim_i = 1:dim_ode
%    predict_r(:,dim_i) = predict_r(:,dim_i) * train_norm_set(dim_i,2) + train_norm_set(dim_i,1);
%end
predict_r = predict_r * 1.5 + 1;
toc;

%% plot
label_font_size = 12;
ticks_font_size = 12;

plot_dim = 1;

%{
figure()
plot( reservoir_tstep * (0:1:length(predict_r)-1) ,predict_r(:,plot_dim))
title(['driven a = ' num2str(driven_a_predict,8)])
xlabel('t','FontSize',label_font_size)
ylabel('x','FontSize',label_font_size)
set(gca,'FontSize',ticks_font_size)
set(gcf,'color','white')
%}

%
figure()
subplot(2,1,1)
plot(ts_predict_real(1:end,1),ts_predict_real(1:end,2))
axis([-6,8,-6,8])
xlabel('x','FontSize',label_font_size)
ylabel('y','FontSize',label_font_size)
title(['real attractor' newline 'driven a = ' num2str(driven_a_predict,8)])
set(gcf,'color','white')

subplot(2,1,2)
plot(predict_r(1:end,1),predict_r(1:end,2))
%axis([-6,8,-6,8])
xlabel('x','FontSize',label_font_size)
ylabel('y','FontSize',label_font_size)
title(['prediction of reservoir' newline 'driven a = ' num2str(driven_a_predict,8)])
set(gcf,'color','white')
%

%{
figure()
plot(t,ts_drive)
ylabel('drive')
xlabel('t')
set(gcf,'color','white')
%}