CFLAGS ?= -O2 -g -Wall -Wextra -std=gnu11

bc250-native-probe: probe.c shaders.h
	$(CC) $(CFLAGS) -o $@ probe.c -lm

kfd_probe: kfd_probe.c
	$(CC) $(CFLAGS) -o $@ kfd_probe.c

eop_irq_probe: eop_irq_probe.c
	$(CC) $(CFLAGS) -o $@ eop_irq_probe.c

scanout_probe: scanout_probe.c
	$(CC) $(CFLAGS) -o $@ scanout_probe.c -lxcb -lxcb-dri3 -lxcb-present

clean:
	rm -f bc250-native-probe kfd_probe eop_irq_probe scanout_probe

.PHONY: clean
