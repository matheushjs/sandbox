
graphics.off();
dev.new(width=8, height=4);

x = seq(0, 100, length=10000)
plot(x, dgamma(x, shape=8000, scale=0.01), type="l", col=1, lwd=3)
lines(x, dgamma(x, shape=50, scale=0.5), type="l", col=2, lwd=3, lty=2)

legend("topleft", c("original (8000, 0.5)", "initial params (50, 0.5)"), col=1:2, lwd=3, lty=1:2);

savePlot("initparam.png");
