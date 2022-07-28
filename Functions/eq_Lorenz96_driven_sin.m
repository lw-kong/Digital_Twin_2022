function dxdt = eq_Lorenz96_driven_sin(t,x,F,driven_a,driven_f)

m = length(x);

x_p1 = x;
x_m1 = x;
x_m2 = x;

x_p1(1:end-1) = x(2:end);
x_p1(end) = x(1);
x_m1(2:end) = x(1:end-1);
x_m1(1) = x(end);
x_m2(3:end) = x(1:end-2);
x_m2(1:2) = x(end-1:end);

dxdt = (x_p1-x_m2).*x_m1 - x + F ...
    + driven_a * sin(driven_f * t) * ones(m,1);

end

