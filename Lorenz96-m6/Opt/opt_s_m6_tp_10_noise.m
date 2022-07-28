if ispc
    addpath '..'
    addpath '..\..\Functions'
else
    addpath '..'
    addpath '../../Functions'
    parpool('local',15)
end

iter_max = 160;
n = 1000;
take_num = 10;
repeat_num = 15;


% 1~2: eig_rho, W_in_a
% 3~4: a, k 
% 5~6: drive_w, noise_a
lb = [0  0       0  1      0  -3];
ub = [2  1       1  n      2  -1];



rng((now*1000-floor(now*1000))*100000)
tic
options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'PlotFcn','surrogateoptplot');
filename = ['opt_Lorenz96_m6_noise_' datestr(now,30) '_' num2str(randi(999)) '.mat'];

func = @(x) (func_repeat_train_1(x,n,repeat_num,take_num));
[opt_result,opt_fval,opt_exitflag,opt_output,opt_trials] = surrogateopt(func,lb,ub,options);
toc

save(filename)
if ~ispc
    exit;
end
