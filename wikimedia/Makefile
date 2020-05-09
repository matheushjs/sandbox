
all:
	convert -delay 5 frame*.png -loop 0 out.gif
	gifsicle -i out.gif -O3 --optimize --colors 16 -o out-opt.gif
