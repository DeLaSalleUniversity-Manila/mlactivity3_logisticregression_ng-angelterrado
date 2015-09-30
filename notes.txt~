x = load('ml3x.dat');
y = load('ml3y.dat');
m = length(x);
x = [ones(m,1), x];

pos = find(y==1); 
neg = find(y==0);

plot(x(pos,2),x(pos,3),'+');
hold on
plot(x(neg,2),x(neg,3),'o');
xlabel('exam1 scores');
ylabel('exam2 scores');
legend('admitted','not admitted');

theta = [0; 0; 0];
g = inline('1.0 ./ (1.0 + exp(-z))'); 

MAX_ITERATION = 7;
J_vals = zeros(MAX_ITERATION,1);

for i = 1:MAX_ITERATION
    z = x * theta;
    h = g(z);
    
    grad = (1/m) .* x' * (h-y);
    H = (1/m) .* x' * diag(h) * diag(1-h) * x;
    
    J_vals(i) =(1/m) * sum(-y.*log(h) - (1-y).*log(1-h));
    
    theta = theta - H\grad;
end

1 - g([1, 20, 80]*theta)

plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y);
xlabel('exam1 scores');
ylabel('exam2 scores');
legend('admitted', 'not admitted', 'decision boundary');
hold off

plot(0:MAX_ITERATION-1, J_vals, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); 
ylabel('J');
J_vals
