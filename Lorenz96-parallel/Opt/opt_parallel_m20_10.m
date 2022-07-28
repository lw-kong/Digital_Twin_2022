if ispc
    addpath '..'
    addpath '..\..\Functions'
else
    addpath '..'
    addpath '../../Functions'
    parpool('local',15)
end

iter_max = 200;
n = 800;
take_num = 10;
repeat_num = 15;
Lorenz96_m = 20;

% 1~2: eig_rho W_in_a
% 3~5: a beta k 
% 6~7: drive_w noise
lb = [0  0     0  -15 1    0  -4];
ub = [2  2     1  -3  n    3  -1];



rng((now*1000-floor(now*1000))*100000)
tic
options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'PlotFcn','surrogateoptplot');
filename = ['opt_Lorenz96_parallel_m' num2str(Lorenz96_m) ...
    '_10_' datestr(now,30) '_' num2str(randi(999)) '.mat'];

func = @(x) (func_repeat_train_1(x,n,repeat_num,take_num,Lorenz96_m));
[opt_result,opt_fval,opt_exitflag,opt_output,opt_trials] = surrogateopt(func,lb,ub,options);
toc

save(filename)
if ~ispc
    exit;
end
