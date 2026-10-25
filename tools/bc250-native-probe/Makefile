CFLAGS ?= -O2 -g -Wall -Wextra -std=gnu11

bc250-native-probe: probe.c shaders.h
	$(CC) $(CFLAGS) -o $@ probe.c -lm

kfd_probe: kfd_probe.c
	$(CC) $(CFLAGS) -o $@ kfd_probe.c

clean:
	rm -f bc250-native-probe kfd_probe

.PHONY: clean
