require(elfDistr);

graphics.off();
dev.new(height=6, width=12);

gamma = seq(0.1, 10, length=10000);
y = dkwcwg(3, 0.8, 1, gamma, 3, 0.5, log=T)
y = diff(y) / diff(beta)[1];
plot(gamma[seq_along(y)], y, xlab=expression(gamma), ylab="");
abline(h=0, col=4);

legend("topright", "Moving zero", col=4, lwd=1);
savePlot("critpoint-kwcwg.png");
