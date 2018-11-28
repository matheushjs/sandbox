
make all;

rm -vf war.out raw.out raw_war.out;

for i in 1 1 1 1 1; do
	./raw &> /dev/null; # Warmup
done;

for i in $(seq 100); do
	perf stat -e cycles,instructions,cache-misses ./raw 2>&1 \
		| grep -e "cycles" -e "instructions" -e "cache-misses" -e "time elapsed" \
		| sed -e "s/^ \+//g" \
		| sed -e "s/ \+/ /g" \
		| sed -e "s/,//g" \
		| cut -f1,2 -d' ' \
		| sed -e "s/ /,/g";
done > raw.out;

for i in 1 1 1 1 1; do
	./war &> /dev/null; # Warmup
done;

for i in $(seq 100); do
	perf stat -e cycles,instructions,cache-misses ./war 2>&1 \
		| grep -e "cycles" -e "instructions" -e "cache-misses" -e "time elapsed" \
		| sed -e "s/^ \+//g" \
		| sed -e "s/ \+/ /g" \
		| sed -e "s/,//g" \
		| cut -f1,2 -d' ' \
		| sed -e "s/ /,/g";
done > war.out;

for i in 1 1 1 1 1; do
	./raw_war &> /dev/null; # Warmup
done;

for i in $(seq 100); do
	perf stat -e cycles,instructions,cache-misses ./raw_war 2>&1 \
		| grep -e "cycles" -e "instructions" -e "cache-misses" -e "time elapsed" \
		| sed -e "s/^ \+//g" \
		| sed -e "s/ \+/ /g" \
		| sed -e "s/,//g" \
		| cut -f1,2 -d' ' \
		| sed -e "s/ /,/g";
done > raw_war.out;
