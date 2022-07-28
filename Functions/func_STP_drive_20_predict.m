function predict = func_STP_drive_20_predict(x_warmup,W_in,res_net,P,flag)
% included warmup
% flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
n = flag(1); % number of nodes in res_net
a = flag(2);
warmup_length = flag(3);
predict_cut = flag(4);
predict_length = flag(5);

dim_ode = flag(6);
dim_drive = flag(7);

dim_in = dim_ode + dim_drive;

r = zeros(n,1); % hidden layer state n * 1
u = zeros(dim_in,1);
%% warm up
%x_warmup = x_warmup(1:warmup_length,:);
for t_i = 1:(warmup_length-1)
    u(:) = x_warmup(t_i,:);
    r = (1-a) * r + a * tanh(res_net*r+W_in*u);
end


%% predicting
% disp('  predicting...')
predict = zeros(predict_cut + predict_length,dim_ode);
u(:) = x_warmup(warmup_length,:);

for t_i=1:predict_cut + predict_length
    u(dim_ode+1:end) = x_warmup( warmup_length-1+t_i ,dim_ode+1:end); % driving signal
    
    r = (1-a) * r + a * tanh( res_net*r+W_in*u );    
    r_out = r;
    r_out(2:2:end) = r_out(2:2:end).^2; %even number -> squared
    
    predict(t_i,:) = P*r_out;
    u(1:dim_ode) = predict(t_i,:);
end

predict = predict(predict_cut+1 : end,:);


end