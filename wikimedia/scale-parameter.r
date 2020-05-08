require(RColorBrewer);

col = brewer.pal(n=12, "Set3");
scale = 1;

for(scale in seq(1, 1.5, length=49)){
	x = seq(-20, 20, length=30000)/scale;
	y = (0.5*dnorm(x, -1, 0.5) + 0.5*dnorm(x, 1, 0.5)) / scale;
	mean = sum(x * y * diff(x[1:2]));
	plot(x, y, type="l", col=col[4], lwd=5, xlim=c(-10, 10), ylim=c(0, 0.45));
	print(mean);

	polygon(
		c(mean-0.5, mean, mean+0.5),
		c(0, 0.01, 0),
		col=3);

	text(-10, 0.43, paste("scale = ", round(scale, digits=2), "x", sep=""), pos=4);
	polygon(
		c(-5.7, -3+4*(scale-1), -3+4*(scale-1), -5.7) + 0.1,
		c(0.44, 0.44, 0.43, 0.43) - 0.0035,
		col=col[5], border=F);
}

for(scale in seq(1.5, 0.7, length=49)){
	x = seq(-20, 20, length=30000)/scale;
	y = (0.5*dnorm(x, -1, 0.5) + 0.5*dnorm(x, 1, 0.5)) / scale;
	mean = sum(x * y * diff(x[1:2]));
	plot(x, y, type="l", col=col[4], lwd=5, xlim=c(-10, 10), ylim=c(0, 0.45));
	print(mean);

	polygon(
		c(mean-0.5, mean, mean+0.5),
		c(0, 0.01, 0),
		col=3);

	text(-10, 0.43, paste("scale = ", round(scale, digits=2), "x", sep=""), pos=4);
	polygon(
		c(-5.7, -3+4*(scale-1), -3+4*(scale-1), -5.7) + 0.1,
		c(0.44, 0.44, 0.43, 0.43) - 0.0035,
		col=col[5], border=F);
}
