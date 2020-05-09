require(RColorBrewer);
require(stringr);

#img = image_graph(540, 540, res = 96);
dev.new(width="540px", height="540px", unit="px");

col = brewer.pal(n=12, "Set3");
scale = 1;
count = 0;

for(scale in seq(1, 1.5, length=40)){
	x = seq(0, 120, length=5000);
	y = dgamma(x/scale, shape=4, scale=6) / scale;
	mean = sum(x * y * diff(x[1:2]));
	plot(x, y, type="l", col=col[4], lwd=5, xlab="x", ylab="density", xlim=c(0, 60), ylim=c(0, 0.05));

	polygon(
		c(mean-0.8, mean, mean+0.8),
		c(0, 0.002, 0),
		col=3);

	polygon(
		c(mean-0.8, mean, mean+0.8) + 0.9 - mean + 35,
		c(0, 0.002, 0) + 0.039,
		col=3);
	text(1.4 + 35, 0.0397, "mean", pos=4);

	text(-0.8 + 35, 0.043, paste("scale = ", round(scale, digits=2), "x", sep=""), pos=4);
	polygon(
		c(0, 10+8*(scale-1), 10+8*(scale-1), 0) + 13 + 35,
		c(0.043, 0.043, 0.042, 0.042) + 0.0007,
		col=col[5], border=F);

	savePlot(paste("frame", str_replace_all(format(count, width=3), " ", "0"), ".png", sep=""));

	count = count + 1;
	print(count);
}

for(scale in seq(1.5, 0.7, length=80)){
	x = seq(0, 120, length=5000);
	y = dgamma(x/scale, shape=4, scale=6) / scale;
	mean = sum(x * y * diff(x[1:2]));
	plot(x, y, type="l", col=col[4], lwd=5, xlab="x", ylab="density", xlim=c(0, 60), ylim=c(0, 0.05));

	polygon(
		c(mean-0.8, mean, mean+0.8),
		c(0, 0.002, 0),
		col=3);

	polygon(
		c(mean-0.8, mean, mean+0.8) + 0.9 - mean + 35,
		c(0, 0.002, 0) + 0.039,
		col=3);
	text(1.4 + 35, 0.0397, "mean", pos=4);

	text(-0.8 + 35, 0.043, paste("scale = ", round(scale, digits=2), "x", sep=""), pos=4);
	polygon(
		c(0, 10+8*(scale-1), 10+8*(scale-1), 0) + 13 + 35,
		c(0.043, 0.043, 0.042, 0.042) + 0.0007,
		col=col[5], border=F);

	savePlot(paste("frame", str_replace_all(format(count, width=3), " ", "0"), ".png", sep=""));

	count = count + 1;
	print(count);
}

#dev.off();
#animation <- image_animate(img, fps = 100, optimize = FALSE);
#image_write_gif(animation, "out.gif");
