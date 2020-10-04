dtsquare = function(x, p, m) df(x * (m - p + 1) / (p*m), df1=p, df2=m-p+1);
x = seq(0, 15, length=1000);

plot(x, dtsquare(x, 2, 5), type="l", col=1, lwd=3)
lines(x, dtsquare(x, 3, 5), type="l", col=2, lwd=3)
lines(x, dtsquare(x, 4, 5), type="l", col=3, lwd=3)
lines(x, dtsquare(x, 4, 50), type="l", col=4, lwd=3)
lines(x, dtsquare(x, 4, 5000), type="l", col=5, lwd=3)
legend("topright", c(expression(paste(p==2, phantom(0), m==5)), expression(paste(p==3,phantom(0), m==5)), expression(paste(p==4,phantom(0), m==5)), expression(paste(p==4,phantom(0), m==50)), expression(paste(p==4,phantom(0), m==5000))), col=1:5, lwd=3)
savePlot("hotelling-pdf.png")

ptsquare = function(x, p, m) pf(x * (m - p + 1) / (p*m), df1=p, df2=m-p+1);
plot(x, ptsquare(x, 2, 5), type="l", col=1, lwd=3, ylim=c(0, 1))
lines(x, ptsquare(x, 3, 5), type="l", col=2, lwd=3)
lines(x, ptsquare(x, 4, 5), type="l", col=3, lwd=3)
lines(x, ptsquare(x, 4, 50), type="l", col=4, lwd=3)
lines(x, ptsquare(x, 4, 5000), type="l", col=5, lwd=3)
legend("bottomright", c(expression(paste(p==2, phantom(0), m==5)), expression(paste(p==3,phantom(0), m==5)), expression(paste(p==4,phantom(0), m==5)), expression(paste(p==4,phantom(0), m==50)), expression(paste(p==4,phantom(0), m==5000))), col=1:5, lwd=3)
savePlot("hotelling-cdf.png")
