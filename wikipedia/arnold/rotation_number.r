require(colorspace);

WIDTH = 8*500
HEIGHT = 8*500

graphics.off();
dev.new(width=8, height=8);

pixels = readBin("pixels.mat", "double", n=WIDTH*HEIGHT, size=4);
pixels = matrix(pixels, ncol=HEIGHT, nrow=WIDTH);

pixels = pixels[seq(1, HEIGHT, by=4), seq(1, WIDTH, by=4)];
HEIGHT = nrow(pixels);
WIDTH = ncol(pixels);

pixels[pixels > 1.643345] = 0;

palette = function(n) sequential_hcl(palette="Viridis", n=n);
par(fg = NA, col="black");
filled.contour(1:WIDTH, 1:HEIGHT, pixels,
			   color.palette=palette,
			   levels = c(seq(0, 1.1, length=200)),
			   #nlevels = 200,
			   axes = F,
			   key.axes = axis(4, seq(0, 5, by=0.5))
			)

title(ylab=expression(K %in% group("[", list(0, 2 * pi), "]") ), line=1)
title(xlab=expression(Omega %in% group("[", list(0, 1), "]")), line=1)
