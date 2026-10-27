#!/bin/bash
# Builds amdgpu with bc250-amdgpu.patch for the running SteamOS kernel, as a user, on the board.
# Usage: build-amdgpu-bc250.sh WORKDIR
#   WORKDIR/src.orig/drivers/gpu/drm/amd: the driver from the SteamOS kernel source package of the
#   running kernel (linux-neptune, the same version as uname -r). The patched copy goes to WORKDIR/src,
#   the module to WORKDIR/amdgpu.ko.zst, where install-amdgpu-bc250.sh (run with sudo) takes it.
set -eu
W=$(realpath "$1")
HERE=$(dirname "$(realpath "$0")")
KVER=$(uname -r)
[ -d "$W/src.orig/drivers/gpu/drm/amd" ] || { echo "no $W/src.orig/drivers/gpu/drm/amd"; exit 1; }
rm -rf "$W/src"
mkdir -p "$W/src"
cp -r "$W/src.orig/drivers" "$W/src/"
(cd "$W/src" && patch -p1 < "$HERE/bc250-amdgpu.patch")
make -C "/usr/lib/modules/$KVER/build" M="$W/src/drivers/gpu/drm/amd/amdgpu" -j"$(nproc)" modules > "$W/build.log" 2>&1 ||
	{ tail -20 "$W/build.log"; exit 1; }
strip --strip-debug -o "$W/amdgpu.ko" "$W/src/drivers/gpu/drm/amd/amdgpu/amdgpu.ko"
zstd -q -f -19 "$W/amdgpu.ko" -o "$W/amdgpu.ko.zst"
echo "built $W/amdgpu.ko.zst for $KVER; install: sudo $HERE/install-amdgpu-bc250.sh"
