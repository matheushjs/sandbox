
graphics.off();
dev.new(height=6, width=12);

alpha = seq(0.000001, 10, length=10000)
y = -digamma(alpha)/gamma(alpha)
plot(alpha, y, xlab=expression(alpha), ylab="", main=expression(frac(paste(partialdiff, L), paste(partialdiff, alpha))), col=1, axes=F);
axis(1);

abline(h=-0.3, col=4);
#idx = sort.list(abs(y + 0.3))[1:5]; # 3622 3621 1792 3623 1793
idx = 1792;
px = alpha[idx];
py = y[idx]
points(px, py, col=2, pch=19, cex=2)
legend("topright", "Moving zero", col=4, lwd=1);

savePlot("critpoint.png");

abline(h=0, col=4);
abline(h=1.05, col=4);

if(FALSE){
	toggle = TRUE;
	for(alpha in seq(0.0001, 4, length=600)){
	#for(alpha in seq(0, 4, length=1000)){
		x = seq(0.0001, 10, length=1000);

		if(toggle){
			plot(x, dgamma(x, shape=alpha, scale=exp(-0.3)));
		} else {
			lines(x, dgamma(x, shape=alpha, scale=exp(-0.3)));
		}

		abline(v=1);

		legend("topright", paste("alpha = ", alpha), bg="white");
		toggle = TRUE;
	}
}
