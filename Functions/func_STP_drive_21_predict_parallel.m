function predict = func_STP_drive_21_predict_parallel(n_parallel,W_n,x_warmup,W_in,W_r,W_out,flag)
% included warmup
% flag_r = [n dim a warmup_r_step_length predict_r_step_cut predict_r_step_length];
n = flag(1); % number of nodes in res_net
a = flag(2);
warmup_length = flag(3);
predict_cut = flag(4);
predict_length = flag(5);

dim_out = flag(6);
dim_couple = flag(7);
dim_global_drive = flag(8);

dim_in = dim_out + dim_couple + dim_global_drive;


r = randn(n,n_parallel); % hidden layer state n * 1
u = zeros(dim_in,1);
%% warm up
%x_warmup = x_warmup(1:warmup_length,:);
for p_i = 1:n_parallel
    for t_i = 1:(warmup_length-1)
        u(:) = x_warmup(t_i,:);
        r(:,p_i) = (1-a) * r(:,p_i) + a * tanh(W_r*r(:,p_i)+W_in*u);
    end
end

%% predicting
% disp('  predicting...')
predict = zeros(predict_cut + predict_length,dim_out);
u_out = zeros(n_parallel*dim_out + dim_global_drive,1);
u_out(1:n_parallel*dim_out) = repmat( x_warmup(warmup_length,1:dim_out),[n_parallel,1]);

for t_i = 1:predict_cut + predict_length
    u_out(end-dim_global_drive+1:end) = x_warmup( warmup_length-1+t_i ,end-dim_global_drive+1:end); % driving signal
    
    u_in = W_n * u_out;
    for p_i = 1:n_parallel
        r(:,p_i) = (1-a) * r(:,p_i) + a * tanh( W_r*r(:,p_i)+...
            W_in*u_in( (p_i-1)*dim_in+1:p_i*dim_in ) );
        
        r_out = r(:,p_i);
        r_out(2:2:end) = r_out(2:2:end).^2; %even number -> squared
        predict(t_i,(p_i-1)*dim_out+1:p_i*dim_out) = W_out*r_out;
    end    
    u_out(1:n_parallel*dim_out) = predict(t_i,:);
end

predict = predict(predict_cut+1 : end,:);


end