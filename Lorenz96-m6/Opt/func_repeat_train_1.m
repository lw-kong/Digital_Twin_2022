function mean_rmse = func_repeat_train_1(hyperpara_set,N,repeat_num,take_num)
tic

rmse_set = zeros(repeat_num,1);
parfor repeat_i = 1:repeat_num
    rng(repeat_i*20000 + (now*1000-floor(now*1000))*100000)
    rmse_set(repeat_i) = func_train_1(hyperpara_set,N);    
end

rmse_set = sort(rmse_set);
rmse_set = rmse_set(1:take_num);

mean_rmse = mean(rmse_set);
fprintf('\nmean rmse is %f\n',mean_rmse)
toc
end

