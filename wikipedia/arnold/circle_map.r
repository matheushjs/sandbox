

f = function(theta, omega, K) theta + omega + (K / 2 / pi) * sin(2 * pi * theta);

x = runif(min=0, max=1, n=1);
result = x;
for(i in 1:10000){
	x = f(x, omega=0.5, K=1.3);
	result = c(result, x);
}

par(mfrow=c(1, 2));
plot(result %% 1);
plot( (result / (1 : length(result)))[-c(1:10)] , type="l");
