require(colorspace)
cols = qualitative_hcl(4, "Set 2", alpha=0.5)

graphics.off();
dev.new(width=8, height=4);

x = seq(0, 100, length=10000)
plot(x, dgamma(x, shape=50, scale=0.5), type="l", col=cols[1], lwd=3, lty=1, ylim=c(0, 0.6));
lines(x, dgamma(x, shape=1, scale=1), type="l", col=cols[2], lwd=3, lty=1, ylim=c(0, 0.6));
lines(x, dgamma(x, shape=2, scale=2), type="l", col=cols[3], lwd=3, lty=1, ylim=c(0, 0.6));

data = rgamma(n=1000, shape=6257, scale=0.01)
h1 = hist(data - min(data) * (1 - 1/100), plot=F)
lines(h1, freq=F);
h1 = hist(rgamma(n=1000, shape=1, scale=1.5), plot=F)
lines(h1, freq=F);
data = rweibull(n=1000, shape=70, scale=80)
h1 = hist(data - min(data) * (1 - 1/100), plot=F)
lines(h1, freq=F);

# legend("topleft", c("original (8000, 0.5)", "initial params (50, 0.5)"), col=1:2, lwd=3, lty=1:2);

savePlot("initparam2.png");
