#!/usr/bin/env python3
# Generates src/graphics/native_gpu/shaderIsaGfx1013.inc: every (encoding, opcode) pair llvm-mc decodes
# for -mcpu=gfx1013, the GPU of both the PS5 and the BC-250, with one mnemonic each for log lines.
#
# The PS5 compiler targets gfx1013, so a shader can only hold opcodes from this table. The shader scanner
# (src/graphics/native_gpu/shaderScan.cpp) flags any other opcode: on the BC-250 such an opcode is undefined
# (VOP2 0x02 v_dot2c_f32_f16 shows that "undefined" can mean "silently writes 0"), or the words are not code.
#
# Method: every opcode of every gfx10 encoding is emitted with several operand variants (zero fields,
# SGPR/VGPR/inline sources, vdst=exec for v_cmpx e64, DS data/vdst combinations, MIMG dmask/unorm/r128/dim/
# a16/OP[7]), each followed by a sentinel of three `s_nop 7`. llvm-mc disassembles the whole list once per
# -mcpu; an opcode is valid when at least one variant decodes to an instruction whose first word is the
# candidate's first word. llvm-mc is needed only here, never at run time, and the output holds LLVM's
# opcode names only (no Sony content).
#
# Usage: python3 -I tools/bc250-shader-isa/gen_gfx1013_isa.py [--llvm-mc PATH] [--out FILE] [--diff FILE]
# Defaults: llvm-mc from $LLVM_MC, then Homebrew's LLVM, then PATH; --out is the checked-in .inc; the diff
# report against gfx1011, gfx1012 and gfx1030 goes to stdout unless --diff names a file.

import argparse
import collections
import os
import re
import shutil
import struct
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_OUT = os.path.join(ROOT, 'src', 'graphics', 'native_gpu', 'shaderIsaGfx1013.inc')

# The encoding order is fixed by ScanEnc in src/graphics/native_gpu/shaderScan.h.
ENCODINGS = ['SOP1', 'SOP2', 'SOPK', 'SOPC', 'SOPP', 'SMEM', 'VOP1', 'VOP2', 'VOPC', 'VOP3', 'VOP3P', 'VINTRP',
             'DS', 'MUBUF', 'MTBUF', 'MIMG', 'FLAT', 'SCRATCH', 'GLOBAL', 'EXP']

V0 = 0x100
LIT = 0x3f800000


def le(ws):
	return b''.join(struct.pack('<I', w) for w in ws)


def candidates():
	cands = []

	def add(enc, op, ws):
		cands.append((enc, op, ws))

	for op in range(62):  # VOP2; 0x3e/0x3f are the VOPC and VOP1 encodings, never VOP2
		w = (op << 25) | (2 << 17) | (1 << 9) | V0
		add('VOP2', op, [w])
		add('VOP2', op, [w & ~0x1ff | 0xff, LIT])
		add('VOP2', op, [(op << 25) | (2 << 17) | (1 << 9) | 0x80])
	for op in range(256):  # VOP1
		w = 0x7e000000 | (op << 9) | (2 << 17) | V0
		add('VOP1', op, [w])
		add('VOP1', op, [(w & ~0x1ff) | 0x80])
	for op in range(256):  # VOPC
		add('VOPC', op, [0x7c000000 | (op << 17) | (1 << 9) | V0])
	for op in range(1024):  # VOP3, VOP3B and the VOP3 forms of VOP1/VOP2/VOPC
		add('VOP3', op, [0xd4000000 | (op << 16) | 2, V0 | ((V0 | 1) << 9) | ((V0 | 2) << 18)])
		add('VOP3', op, [0xd4000000 | (op << 16) | 2, 0x80 | ((V0 | 1) << 9) | ((V0 | 2) << 18)])
		add('VOP3', op, [0xd4000000 | (op << 16) | (0x6a << 8) | 2, V0 | ((V0 | 1) << 9) | ((V0 | 2) << 18)])
	for op in range(128):  # VOP3P
		add('VOP3P', op, [0xcc000000 | (op << 16) | (1 << 14) | 2, V0 | ((V0 | 1) << 9) | ((V0 | 2) << 18) | (3 << 27)])
	for op in range(256):  # SOP1
		add('SOP1', op, [0xbe800000 | (4 << 16) | (op << 8) | 2])
		add('SOP1', op, [0xbe800000 | (4 << 16) | (op << 8) | 0x80])
	for op in range(0x60):  # SOP2
		add('SOP2', op, [0x80000000 | (op << 23) | (4 << 16) | (2 << 8) | 0])
	for op in range(0x1d):  # SOPK
		w = 0xb0000000 | (op << 23) | (4 << 16) | 0x0001
		add('SOPK', op, [w])
		add('SOPK', op, [w, 0])
		add('SOPK', op, [0xb0000000 | (op << 23) | (4 << 16) | 0xf801])
	for op in range(128):  # SOPC
		add('SOPC', op, [0xbf000000 | (op << 16) | (2 << 8) | 0])
	for op in range(128):  # SOPP
		add('SOPP', op, [0xbf800000 | (op << 16) | 0])
		add('SOPP', op, [0xbf800000 | (op << 16) | 1])
	for op in range(256):  # SMEM
		add('SMEM', op, [0xf4000000 | (op << 18) | (1 << 6) | 0, 0xfa000000])
		add('SMEM', op, [0xf4000000 | (op << 18) | (2 << 6) | 0, 0xfa000000 | 0])
	for op in range(256):  # DS
		for gds in (0, 1):
			for w1 in ((2 << 24) | (1 << 16) | (1 << 8), (2 << 24) | (1 << 8), (2 << 24), (1 << 8), (1 << 16) | (1 << 8), 0,
			           (2 << 24) | (1 << 16)):
				add('DS', op, [0xd8000000 | (op << 18) | (gds << 17), w1])
	for op in range(256):  # MUBUF: OP in 24:18, OP[7] in bit 25
		hi = op >> 7
		lo = op & 0x7f
		add('MUBUF', op, [0xe0000000 | (hi << 25) | (lo << 18) | (1 << 13), 0x80000000 | (0 << 16) | (4 << 8) | 0])
	for op in range(16):  # MTBUF: OP in 18:16, OP[3] in word 1 bit 21
		add('MTBUF', op, [0xe8000000 | ((op & 7) << 16) | (0x16 << 19) | (1 << 13), 0x80000000 | ((op >> 3) << 21) | (4 << 8) | 0])
	for op in range(256):  # MIMG: OP in 24:18, OP[7] in bit 0
		hi = op >> 7
		lo = op & 0x7f
		for dm in (1, 0xf):
			for r128 in (0, 1):
				for unorm in (0, 1):
					for dim in (0, 1, 6):
						for a16 in (0, 1):
							add('MIMG', op, [0xf0000000 | (lo << 18) | (r128 << 15) | (unorm << 12) | (dm << 8) | (dim << 3) | hi,
							                 (a16 << 30) | (4 << 8) | 0])
	for seg in range(3):  # FLAT, SCRATCH, GLOBAL by SEG in 15:14
		name = ['FLAT', 'SCRATCH', 'GLOBAL'][seg]
		for op in range(128):
			add(name, op, [0xdc000000 | (op << 18) | (seg << 14), (0x7d << 16) | (4 << 8) | 0])
			add(name, op, [0xdc000000 | (op << 18) | (seg << 14), (0x00 << 16) | (4 << 8) | 0])
	for op in range(4):  # VINTRP
		add('VINTRP', op, [0xc8000000 | (2 << 18) | (op << 16) | 0])
	add('EXP', 0, [0xf800080f, 0x03020100])

	# Opcodes that only decode with no destination, no sources, an SGPR destination and the like.
	for op in range(256):
		add('SOP1', op, [0xbe800000 | (op << 8)])
		add('SOP1', op, [0xbe800000 | (4 << 16) | (op << 8)])
		add('SOP1', op, [0xbe800000 | (op << 8) | 4])
		add('VOP1', op, [0x7e000000 | (op << 9)])
		add('VOP1', op, [0x7e000000 | (op << 9) | (4 << 17) | V0])
		add('VOPC', op, [0x7c000000 | (op << 17) | (1 << 9) | 0x80])
	for op in range(0x1d):
		add('SOPK', op, [0xb0000000 | (op << 23) | 0x0001])
		add('SOPK', op, [0xb0000000 | (op << 23) | 0xf801, 0])
	for op in range(128):
		add('SOPP', op, [0xbf800000 | (op << 16) | 0x0003])
	for op in range(0x60):
		add('SOP2', op, [0x80000000 | (op << 23) | (4 << 16) | (2 << 8) | 0x80])
		add('SOP2', op, [0x80000000 | (op << 23) | (4 << 16) | (0x80 << 8) | 2])
	for op in range(128):
		add('SOPC', op, [0xbf000000 | (op << 16) | (0x80 << 8) | 2])
		add('SOPC', op, [0xbf000000 | (op << 16) | (0 << 8) | 0])
	for op in range(1024):
		add('VOP3', op, [0xd4000000 | (op << 16) | 2, V0 | ((V0 | 1) << 9)])
		add('VOP3', op, [0xd4000000 | (op << 16) | 2, V0])
		add('VOP3', op, [0xd4000000 | (op << 16) | 2, 4 | ((V0 | 1) << 9)])
		add('VOP3', op, [0xd4000000 | (op << 16) | 2, V0 | (4 << 9)])
		add('VOP3', op, [0xd4000000 | (op << 16) | 0x6a, V0 | ((V0 | 1) << 9) | ((V0 | 2) << 18)])
	for op in range(256):
		add('SMEM', op, [0xf4000000 | (op << 18), 0])
		add('SMEM', op, [0xf4000000 | (op << 18) | (4 << 6), 0])
	for op in range(256):
		hi = op >> 7
		lo = op & 0x7f
		add('MUBUF', op, [0xe0000000 | (hi << 25) | (lo << 18), 0])
		add('MUBUF', op, [0xe0000000 | (hi << 25) | (lo << 18) | (1 << 14) | (1 << 13), 0x80000000 | (4 << 8)])
	for seg in range(3):
		name = ['FLAT', 'SCRATCH', 'GLOBAL'][seg]
		for op in range(128):
			add(name, op, [0xdc000000 | (op << 18) | (seg << 14), (0x7d << 16) | (4 << 8) | (4 << 24)])
			add(name, op, [0xdc000000 | (op << 18) | (seg << 14) | (1 << 16), (0x7d << 16) | (4 << 8) | (4 << 24)])
	for op in range(1024):
		add('VOP3', op, [0xd4000000 | (op << 16) | 0x00, V0 | ((V0 | 1) << 9)])
		add('VOP3', op, [0xd4000000 | (op << 16) | 0x00, 0])
		add('VOP3', op, [0xd4000000 | (op << 16) | 0x02, 0])
		add('VOP3', op, [0xd4000000 | (op << 16) | 0x02, 4])
	for op in range(1024):
		add('VOP3', op, [0xd4000000 | (op << 16) | 0x7e, V0 | ((V0 | 1) << 9)])  # v_cmpx e64: vdst = exec
	return cands


def find_llvm_mc(explicit):
	for c in (explicit, os.environ.get('LLVM_MC'), '/opt/homebrew/opt/llvm/bin/llvm-mc', shutil.which('llvm-mc')):
		if c and os.path.isfile(c) and os.access(c, os.X_OK):
			return c
	sys.exit('gen_gfx1013_isa: llvm-mc not found (pass --llvm-mc or set LLVM_MC)')


def llvm_version(llvm_mc):
	out = subprocess.run([llvm_mc, '--version'], capture_output=True, text=True, check=True).stdout
	m = re.search(r'LLVM version (\d+)\.(\d+)\.(\d+)', out)
	if not m:
		sys.exit('gen_gfx1013_isa: cannot read the llvm-mc version')
	return tuple(int(x) for x in m.groups())


def decode(llvm_mc, cpu, cands):
	"""(enc, op) -> set of mnemonics for every candidate llvm-mc decodes as the instruction it is."""
	sent = ','.join(['0x07,0x00,0x80,0xbf'] * 3)  # s_nop 7, three times
	lines = []
	for _, _, ws in cands:
		lines.append(','.join('0x%02x' % b for b in le(ws)))
		lines.append(sent)
	p = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=' + cpu, '--disassemble', '--show-encoding'],
	                   input='\n'.join(lines) + '\n', capture_output=True, text=True)
	outs = []
	for line in p.stdout.splitlines():
		if ';' not in line:
			continue
		a, b = line.split(';', 1)
		m = re.search(r'\[(.*?)\]', b)
		outs.append((a.strip(), [int(x, 16) for x in m.group(1).split(',')] if m else []))
	# Group the decoded lines by the sentinels. A candidate llvm-mc rejects entirely leaves no line, so its
	# sentinel run merges with the previous one: a run of 3k s_nop 7 closes k groups.
	groups = []
	cur = []
	run = 0

	def flush():
		groups.append(cur)
		for _ in range(max(1, (run + 1) // 3) - 1):
			groups.append([])

	for o, e in outs:
		if o == 's_nop 7':
			run += 1
		else:
			if run:
				flush()
				cur = []
				run = 0
			cur.append((o, e))
	if run:
		flush()
	if len(groups) != len(cands):
		sys.exit('gen_gfx1013_isa: %s: %d groups for %d candidates (sentinel grouping broke)' % (cpu, len(groups), len(cands)))
	res = {}
	for (enc, op, ws), g in zip(cands, groups):
		# The first decoded instruction must be the candidate itself, not a shorter form that left words over.
		if not g or g[0][1][:4] != list(le(ws[:1])):
			continue
		res.setdefault((enc, op), set()).add(g[0][0].split()[0])
	return res


def write_inc(path, table, version, cmd):
	by_enc = collections.defaultdict(list)
	for (enc, op), names in table.items():
		by_enc[enc].append((op, sorted(names, key=lambda n: (len(n), n))[0]))
	out = []
	out.append('// Generated by tools/bc250-shader-isa/gen_gfx1013_isa.py from llvm-mc %d.%d.%d (-mcpu=gfx1013). Do not edit.' % version)
	out.append('// Regenerate: %s' % cmd)
	out.append('//')
	out.append('// Every (encoding, opcode) pair the gfx1013 disassembler accepts, sorted by opcode, with one LLVM mnemonic for')
	out.append('// log lines. Opcode numbering as in shaderScan.cpp: VOP3 10 bits, MUBUF/MIMG with OP[7], MTBUF with OP[3],')
	out.append('// FLAT/SCRATCH/GLOBAL split by SEG. %d pairs.' % len(table))
	out.append('')
	for enc in ENCODINGS:
		ops = sorted(by_enc.get(enc, []))
		out.append('inline constexpr IsaOpName kIsa%s[] = {' % enc)
		for op, name in ops:
			out.append('\t{0x%03x, "%s"},' % (op, name))
		out.append('};')
	out.append('')
	out.append('inline constexpr std::span<const IsaOpName> kIsaByEnc[] = {')
	for enc in ENCODINGS:
		out.append('\tkIsa%s,' % enc)
	out.append('};')
	with open(path, 'w') as f:
		f.write('\n'.join(out) + '\n')


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--llvm-mc')
	ap.add_argument('--out', default=DEFAULT_OUT)
	ap.add_argument('--diff')
	args = ap.parse_args()
	llvm_mc = find_llvm_mc(args.llvm_mc)
	version = llvm_version(llvm_mc)
	if version[0] < 17:
		sys.exit('gen_gfx1013_isa: llvm-mc %d.%d.%d is older than 17' % version)
	cands = candidates()
	cpus = ['gfx1013', 'gfx1011', 'gfx1012', 'gfx1030']
	res = {c: decode(llvm_mc, c, cands) for c in cpus}
	base = res['gfx1013']
	for enc in ENCODINGS:
		if not any(k[0] == enc for k in base):
			sys.exit('gen_gfx1013_isa: no %s opcode decoded; llvm-mc output changed?' % enc)
	write_inc(args.out, base, version, 'python3 -I tools/bc250-shader-isa/gen_gfx1013_isa.py')
	rep = []
	for c in cpus[1:]:
		rep.append('== in %s, not in gfx1013' % c)
		rep += ['   %s 0x%03x %s' % (k[0], k[1], sorted(res[c][k])) for k in sorted(set(res[c]) - set(base))]
		rep.append('== in gfx1013, not in %s' % c)
		rep += ['   %s 0x%03x %s' % (k[0], k[1], sorted(base[k])) for k in sorted(set(base) - set(res[c]))]
	cnt = collections.Counter(k[0] for k in base)
	rep.append('gfx1013: %d pairs; %s' % (len(base), ', '.join('%s %d' % (e, cnt[e]) for e in ENCODINGS)))
	text = '\n'.join(rep) + '\n'
	if args.diff:
		with open(args.diff, 'w') as f:
			f.write(text)
	else:
		sys.stdout.write(text)


if __name__ == '__main__':
	main()
