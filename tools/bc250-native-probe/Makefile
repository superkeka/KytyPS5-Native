CFLAGS ?= -O2 -g -Wall -Wextra -std=gnu11

bc250-native-probe: probe.c shaders.h
	$(CC) $(CFLAGS) -o $@ probe.c

clean:
	rm -f bc250-native-probe

.PHONY: clean
