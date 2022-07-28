function [validation_performance,W_in,W_r,W_out,t_validate,x_real,x_validate] = ...
    func_STP_drive_22_train_noise(udata,tp_length,flag,...
    W_in_type,res_net_type,validation_type,echoing_plot_bool,train_noise)

% 22 updates
% train_noise


% use multiple trials of training to train one single result Wout
% Tp is affecting globally. Each node receives the same all control parameter
% W_in_type
%           1 : each node receives all dim of the input, a dense W_in
%           2 : each node receives one dim of the input
% res_net_type
%           1 : symmeric, normally distributed, with mean 0 and variance 1
%           2 : asymmeric, uniformly distributed between 0 and 1
% validation type
%           1 : max rmse among the tp_i
%           2 : success length
%           3 : prod of all tp_i rmse 
%           4 : average of all tp_i rmse

% udata = zeros( trials, steps, dim + tp_dim )

%fprintf('in train %f\n',rand)
% flag_r_train = [n k eig_rho W_in_a a beta...
%                 0 train_r_step_length validate_r_step_length reservoir_tstep dim
%                 success_threshold];
n = flag(1); % number of nodes in res_net
k = flag(2); % mean degree of res_net
eig_rho = flag(3);
W_in_a = flag(4);
a = flag(5);
beta = flag(6);

train_length = flag(8);
validate_length = flag(9);

tstep = flag(10);
dim_ode = flag(11);
dim_drive = flag(12);

dim_in = dim_ode + dim_drive;
dim_out = dim_ode;

len_washout = flag(13);

if validation_type == 2
    success_threshold = flag(14);
else
    success_threshold = 0;
end

validate_start = train_length+2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define W_in
if W_in_type == 1
    % driving dimensions and ode input dimensions are treated equally
    W_in = W_in_a*(2*rand(n,dim_in)-1);
elseif W_in_type == 2
    % each node is inputed with with one dimenson of real data
    % and all the tuning parameters
    W_in = zeros(n,dim_in);
    n_win = n - mod(n,dim_ode); % round down
    index = randperm(n_win); index = reshape(index,n_win/dim_ode,dim_ode);
    for d_i=1:dim_ode
        W_in(index(:,d_i),d_i)=W_in_a*(2*rand(n_win/dim_ode,1)-1);
    end
    W_in(:,dim_ode+1:end) = W_in_a*(2*rand(n,dim_drive)-1);
elseif W_in_type == 3
    % each node is inputed with with one dimenson of real data
    % driving dimensions and ode input dimensions are treated equally
    W_in = zeros(n,dim_in);
    n_win = n - mod(n,dim_in); % round down
    index = randperm(n_win); index = reshape(index,n_win/dim_in,dim_in);
    for d_i=1:dim_in
        W_in(index(:,d_i),d_i)=W_in_a*(2*rand(n_win/dim_in,1)-1);
    end
else
    fprintf('W_in type error\n');
    return
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define reservoir_network
if res_net_type == 1
    W_r=sprandsym(n,k/n); % symmeric, normally distributed, with mean 0 and variance 1.
elseif res_net_type == 2
    k = round(k);
    index1=repmat(1:n,1,k)'; % asymmeric, uniformly distributed between 0 and 1
    index2=randperm(n*k)';
    index2(:,2)=repmat(1:n,1,k)';
    index2=sortrows(index2,1);
    index1(:,2)=index2(:,2);
    W_r=sparse(index1(:,1),index1(:,2),rand(size(index1,1),1),n,n); 
else
    fprintf('res_net type error\n');
    return
end
%res_net, adjacency matrix
%rescale eig
eig_D=eigs(W_r,1); %only use the biggest one. Warning about the others is harmless
W_r=(eig_rho/(abs(eig_D))).*W_r;
W_r=full(W_r);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% training
%disp('  training...')
r_reg = zeros(n,tp_length*(train_length-len_washout));
y_reg = zeros(dim_out,tp_length*(train_length-len_washout));
r_end = zeros(n,tp_length);
for tp_i = 1:tp_length
    train_x = zeros(train_length,dim_in);
    train_y = zeros(train_length,dim_in);
    train_x(:,:) = udata(tp_i,1:train_length,:);
    train_y(:,:) = udata(tp_i,2:train_length+1,:);
    train_x = train_x';
    train_y = train_y';
    
    
    r_all = [];
    r_all(:,1) = zeros(n,1);%2*rand(n,1)-1;%
    for ti = 1:train_length
        r_all(:,ti+1) = (1-a)*r_all(:,ti) + ...
            a*tanh( W_r*r_all(:,ti) + W_in*(train_x(:,ti)+train_noise*randn(dim_in,1)) );
    end
    r_out = r_all(:,len_washout+2:end); % n * (train_length - 11)
    r_out(2:2:end,:) = r_out(2:2:end,:).^2;
    r_end(:,tp_i) = r_all(:,end); % n * 1
    
    r_reg(:, (tp_i-1)*(train_length-len_washout) +1 : ...
        tp_i*(train_length-len_washout) ) = r_out;
    y_reg(:, (tp_i-1)*(train_length-len_washout) +1 : ...
        tp_i*(train_length-len_washout) ) = train_y(1:dim_out,len_washout+1:end); %no tp
end
W_out = y_reg *r_reg'*(r_reg*r_reg'+beta*eye(n))^(-1);

% echoing plot
if echoing_plot_bool == 1
%
    figure()
    imagesc(r_all)
    xlabel('steps')
    ylabel('n_i')
    title('echoing of the last tp')
    set(gcf,'color','white')

    figure()
    plot(y_reg(1,:))
    hold on
    P_r_reg = W_out*r_reg; % dim * T
    plot( P_r_reg(1,:) )
    xlabel('steps')
    title('echoing and real')
    set(gcf,'color','white')
    hold off
    
    train_error(:,:) = y_reg - P_r_reg;
    train_rmse_ts = sqrt( mean( abs(train_error).^2 ,1) );
    train_rmse = mean(train_rmse_ts);
    
    figure()
    plot(train_rmse_ts)
    xlabel('steps')
    ylabel('training errors')
    set(gcf,'color','white')
    
    fprintf('training error = %f\n',train_rmse)
end
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% validate resnet model
%disp('validating...')
rmse_set = zeros(1,tp_length);
success_length_set = zeros(1,tp_length);

validate_predict_y_set = zeros(tp_length,validate_length,dim_out);
validate_real_y_set = zeros(tp_length,validate_length,dim_out);
for tp_i = 1:tp_length
    validate_real_y_set(tp_i,:,:) = udata(tp_i,validate_start:(validate_start+validate_length-1),1:dim_out);
    
    r = r_end(:,tp_i);
    u = zeros(dim_in,1);
    u(1:dim_in) = udata(tp_i,train_length+1,1:dim_in);
    for t_i = 1:validate_length
        u(dim_out+1:end) = udata(tp_i,train_length+t_i,dim_out+1:end);
        r = (1-a) * r + a * tanh(W_r*r+W_in*u);
        r_out = r;
        r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
        predict_y = W_out * r_out;
        validate_predict_y_set(tp_i,t_i,:) = predict_y;
        u(1:dim_out) = predict_y;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    error = zeros(validate_length,dim_out);
    error(:,:) = validate_predict_y_set(tp_i,:,:) - validate_real_y_set(tp_i,:,:);
    rmse_ts = sqrt( mean( abs(error).^2 ,2) );
    
    success_length_set(tp_i) = validate_length * tstep;
    for t_i = 1:validate_length
        if rmse_ts(t_i) > success_threshold
            success_length_set(tp_i) = t_i * tstep;
            break;
        end
    end
    % if there is NaN in prediction, success length = 0 and rmse = 10
    if sum(isnan(validate_predict_y_set(:))) > 0
        success_length_set(tp_i) = 0;
        rmse_ts = 10;
    end
        
    rmse_set(tp_i) = mean(rmse_ts);
end




if validation_type == 1
    validation_performance =  max(rmse_set);
elseif validation_type == 2
    success_length = min(success_length_set);
    %fprintf('attempt success_length = %f \n',success_length);
    validation_performance = success_length;
elseif validation_type == 3
    for tp_i = 1:tp_length
        rmse_set(tp_i) = max(rmse_set(tp_i),10^-3);
    end
    validation_performance = prod(rmse_set);
elseif validation_type == 4
    validation_performance =  mean(rmse_set);
else
    fprintf('validation type error');
    return
end


t_validate = tstep:tstep:tstep*validate_length;
x_validate = validate_predict_y_set;
x_real = validate_real_y_set;

end


