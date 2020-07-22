require(RColorBrewer);
require(stringr);

#img = image_graph(540, 540, res = 96);
dev.new(width="540px", height="540px", unit="px");

col = 1:12; #brewer.pal(n=12, "Set3");
scale = 1;
count = 0;

for(scale in seq(0.7, 1.5, length=40)){
	x = seq(-20, 20, length=3000);
	y = (0.5*dnorm(x/scale/2, -1, 0.5) + 0.5*dnorm(x/scale/2, 1, 0.5)) / scale/2;
	mean = sum(x * y * diff(x[1:2]));
	plot(x, y, type="l", col=col[2], lwd=5, xlab="x", ylab="density", xlim=c(-10, 10), ylim=c(0, 0.45));

	polygon(
		c(mean-0.5, mean, mean+0.5),
		c(0, 0.01, 0),
		col=3);

	polygon(
		c(mean-0.5, mean, mean+0.5) - 10 + 0.9 - mean,
		c(0, 0.01, 0) + 0.39 + 0.01,
		col=3);
	text(-8.8, 0.393 + 0.01, "mean", pos=4);

	text(-10, 0.43, paste("scale = ", round(scale, digits=2), "x", sep=""), pos=4);
	polygon(
		c(-5.7, -3+4*(scale-1), -3+4*(scale-1), -5.7),
		c(0.44, 0.44, 0.43, 0.43) - 0.0035,
		col=col[4], border=F);

	savePlot(paste("frame", str_replace_all(format(count, width=3), " ", "0"), ".png", sep=""));

	count = count + 1;
	print(count);
}

for(scale in seq(1.5, 0.7, length=40)){
	x = seq(-20, 20, length=3000);
	y = (0.5*dnorm(x/scale/2, -1, 0.5) + 0.5*dnorm(x/scale/2, 1, 0.5)) / scale / 2;
	mean = sum(x * y * diff(x[1:2]));
	plot(x, y, type="l", col=col[2], lwd=5, xlab="x", ylab="density", xlim=c(-10, 10), ylim=c(0, 0.45));

	polygon(
		c(mean-0.5, mean, mean+0.5),
		c(0, 0.01, 0),
		col=3);

	polygon(
		c(mean-0.5, mean, mean+0.5) - 10 + 0.9 - mean,
		c(0, 0.01, 0) + 0.39 + 0.01,
		col=3);
	text(-8.8, 0.393 + 0.01, "mean", pos=4);

	text(-10, 0.43, paste("scale = ", round(scale, digits=2), "x", sep=""), pos=4);
	polygon(
		c(-5.7, -3+4*(scale-1), -3+4*(scale-1), -5.7),
		c(0.44, 0.44, 0.43, 0.43) - 0.0035,
		col=col[4], border=F);

	savePlot(paste("frame", str_replace_all(format(count, width=3), " ", "0"), ".png", sep=""));

	count = count + 1;
	print(count);
}

#dev.off();
#animation <- image_animate(img, fps = 100, optimize = FALSE);
#image_write_gif(animation, "out.gif");
