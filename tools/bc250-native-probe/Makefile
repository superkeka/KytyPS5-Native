CFLAGS ?= -O2 -g -Wall -Wextra -std=gnu11

all: bc250-native-probe mem_probe

bc250-native-probe: probe.c shaders.h
	$(CC) $(CFLAGS) -o $@ probe.c -lm

mem_probe: mem_probe.c shaders.h
	$(CC) $(CFLAGS) -o $@ mem_probe.c

clean:
	rm -f bc250-native-probe mem_probe

.PHONY: all clean
