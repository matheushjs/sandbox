require(tseriesChaos)
require(deSolve);
require(rgl);
require(colorspace);

# params = list(gamma, beta, theta, n, tau)
dP  = function(t, P, params){
	if(t < 0){
		lag = 0.2;
	} else {
		lag = lagvalue(t - params$tau);
	}

	theta = params$theta ** params$n;
	d = params$beta * theta * lag / (theta + lag**params$n) - params$gamma * P;
	return(list(d, P));
}

rotX = function(data, theta){
	theta = (theta / 360) * 2 * pi;
	mat = rbind(
				c(1, 0, 0),
				c(0, cos(theta), -sin(theta)),
				c(0, sin(theta), cos(theta))
			);
	data %*% mat;
}

rotY = function(data, theta){
	theta = (theta / 360) * 2 * pi;
	mat = rbind(
				c(cos(theta), 0, sin(theta)),
				c(0, 1, 0),
				c(-sin(theta), 0, cos(theta))
			);
	data %*% mat;
}

rotZ = function(data, theta){
	theta = (theta / 360) * 2 * pi;
	mat = rbind(
				c(cos(theta), -sin(theta), 0),
				c(sin(theta), cos(theta), 0),
				c(0, 0, 1)
			);
	data %*% mat;
}

chaotic = function(){
	params = list(gamma=0.1, beta=0.2, theta=2, n=10, tau=15.5);
	P0 = 0.1;
	BY = 0.1;
	times = seq(-params$tau, 1000, by=BY);

	out = dede(P0, times, dP, parms=params);

	data2 = embedd(out[,2], m=2, d=params$tau / BY);
	data3 = embedd(out[,2], m=3, d=params$tau / BY);
	data3 = data3[-c(1:1000),]

	plot(out / params$theta, which=1, ylim=c(0, 1.5), lwd=2);
	#plot(data3[,1:2], type="p", col=sequential_hcl("Teal", n=round(nrow(data3)*1.5)));

	data3 = rotZ(rotY(rotX(data3, 15), 15), 0);

	z = rotZ(rotY(rotX(data3, 90), 90), 90)[,3];
	colN = 1000;
	palette = sequential_hcl("Teal", n=colN*1.3)
	palette = palette[1:colN];
	cols = NULL;
	for(zi in z)
		cols = c(cols, palette[round(colN * (zi - min(z)) / (max(z) - min(z)) )]);

	plot3d(data3, type="l", axes=F, xlab="", ylab="", zlab="", col=cols, lwd=2);

	snapshot3d(paste("mackey-glass", params$tau, ".png", sep=""));
}

simple = function(){
	params = list(gamma=0.1, beta=0.2, theta=2, n=10, tau=12);
	P0 = 0.1;
	BY = 0.1;
	times = seq(-params$tau, 800, by=BY);

	out = dede(P0, times, dP, parms=params);
	plot(out[,1], out[,2] / params$theta, type="l", lwd=2);

	data2 = embedd(out[,2], m=2, d=params$tau / BY);
	data3 = embedd(out[,2], m=3, d=params$tau / BY);
	data3 = data3[-c(1:1000),]

	#plot(out / params$theta, which=1, ylim=c(0, 1.5), lwd=2);
	#plot(data3[,1:2], type="p", col=sequential_hcl("Teal", n=round(nrow(data3)*1.5)));

	data3 = rotZ(rotY(rotX(data3, 15), 15), 0);

	z = rotZ(rotY(rotX(data3, 90), 90), 90)[,3];
	colN = 1000;
	palette = sequential_hcl("Teal", n=colN*1.3)
	palette = palette[1:colN];
	cols = NULL;
	for(zi in z)
		cols = c(cols, palette[round(colN * (zi - min(z)) / (max(z) - min(z)) )]);

	plot3d(data3, type="l", axes=F, xlab="", ylab="", zlab="", col=cols, lwd=3);

	# snapshot3d("mackey-glass12.png");
}

timeseries = function(tau=6){
	graphics.off();

	params = list(gamma=0.1, beta=0.2, theta=2, n=10, tau=20);
	P0 = 0.1;
	BY = 0.1;
	times = seq(-params$tau, 800, by=BY);

	out = dede(P0, times, dP, parms=params);

	dev.new(width=8, height=4);
	plot(out[,1], out[,2] / params$theta, type="l", ylab=expression(P / theta), xlab="time (days)", lwd=2);
	savePlot("series20.png");

	params = list(gamma=0.1, beta=0.2, theta=2, n=10, tau=6);
	P0 = 0.1;
	BY = 0.1;
	times = seq(-params$tau, 800, by=BY);

	out = dede(P0, times, dP, parms=params);

	dev.new(width=8, height=4);
	plot(out[,1], out[,2] / params$theta, type="l", ylab=expression(P / theta), xlab="time (days)", lwd=2);
	savePlot("series6.png");
}

chaotic();
#simple();
#timeseries();
