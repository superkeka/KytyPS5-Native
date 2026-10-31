#!/usr/bin/env python3
# Patches a copy of the stock amdgpu (drivers/gpu/drm/amd/amdgpu) for a one-boot VCN experiment on the BC-250,
# following ps5-linux (github.com/ps5-linux/ps5-linux-patches, linux.patch): with amdgpu.bc250_vcn=1 the VCN 2.0.3
# block is added, its firmware is ps5_vcn.bin (navi10_vcn.bin that does not wait for the SMU, which has no VCN
# messages on cyan_skillfish either) loaded directly into the VCPU cache window while GC stays loaded by the PSP,
# its rings have no doorbells, and it is powered and reset by UVD register writes as on the PS5. bc250_vcn=0 is the
# stock driver. vcnpatch.py AMDGPU_DIR
import os
import sys

d = sys.argv[1]


def patch(name, subs):
    path = os.path.join(d, name)
    s = open(path).read()
    for old, new, count in subs:
        n = s.count(old)
        if n != count:
            sys.exit("%s: %r found %d times, expected %d" % (name, old[:70], n, count))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", name)


PSP_EQ = "adev->firmware.load_type == AMDGPU_FW_LOAD_PSP"
PSP_NE = "adev->firmware.load_type != AMDGPU_FW_LOAD_PSP"

patch("amdgpu_vcn.h", [(
    "#define AMDGPU_VCN_STACK_SIZE\t\t(128*1024)",
    "/* BC-250 VCN experiment (bc250_vcn): VCN firmware loaded directly while the rest goes through the PSP. */\n"
    "extern int amdgpu_bc250_vcn;\n"
    "#define AMDGPU_VCN_PSP(adev) ((adev)->firmware.load_type == AMDGPU_FW_LOAD_PSP && !amdgpu_bc250_vcn)\n\n"
    "#define AMDGPU_VCN_STACK_SIZE\t\t(128*1024)", 1)])

patch("amdgpu_vcn.c", [
    ("static void amdgpu_vcn_idle_work_handler(struct work_struct *work);\n",
     "static void amdgpu_vcn_idle_work_handler(struct work_struct *work);\n\n"
     "int amdgpu_bc250_vcn;\n"
     "module_param_named(bc250_vcn, amdgpu_bc250_vcn, int, 0444);\n"
     "MODULE_PARM_DESC(bc250_vcn, \"BC-250: VCN 2.0.3 with ps5_vcn.bin loaded directly, as ps5-linux (1 = on, 0 = off)\");\n", 1),
    ("\tamdgpu_ucode_ip_version_decode(adev, UVD_HWIP, ucode_prefix, sizeof(ucode_prefix));\n",
     "\tamdgpu_ucode_ip_version_decode(adev, UVD_HWIP, ucode_prefix, sizeof(ucode_prefix));\n"
     "\tif (amdgpu_bc250_vcn)\n"
     "\t\tstrscpy(ucode_prefix, \"ps5_vcn\", sizeof(ucode_prefix));\n", 1),
    (PSP_EQ, "AMDGPU_VCN_PSP(adev)", 2),
    (PSP_NE, "!AMDGPU_VCN_PSP(adev)", 2),
])

patch("amdgpu_discovery.c", [(
    "\t\tcase IP_VERSION(2, 0, 3):\n\t\t\tbreak;\n\t\tcase IP_VERSION(2, 5, 0):\n\t\t\tamdgpu_device_ip_block_add(adev, &vcn_v2_5_ip_block);",
    "\t\tcase IP_VERSION(2, 0, 3):\n"
    "\t\t\tif (amdgpu_bc250_vcn) {\n"
    "\t\t\t\tadev->vcn.num_vcn_inst = 1;\n"
    "\t\t\t\tamdgpu_device_ip_block_add(adev, &vcn_v2_0_ip_block);\n"
    "\t\t\t}\n"
    "\t\t\tbreak;\n"
    "\t\tcase IP_VERSION(2, 5, 0):\n\t\t\tamdgpu_device_ip_block_add(adev, &vcn_v2_5_ip_block);", 1)])

ps5_reset = '''
/* BC-250 (bc250_vcn), as ps5-linux on the PS5: the VCPU and its rings reset by UVD register writes. */
static void vcn_v2_0_bc250_reset(struct amdgpu_device *adev)
{
	WREG32_SOC15(UVD, 0, mmUVD_SOFT_RESET, 0x3ffff);
	udelay(20);
	WREG32_SOC15(UVD, 0, mmUVD_SOFT_RESET2, 1);
	udelay(10);
	WREG32_SOC15(UVD, 0, mmUVD_SOFT_RESET2, 0);
	WREG32_SOC15(UVD, 0, mmUVD_SOFT_RESET, UVD_SOFT_RESET__VCPU_SOFT_RESET_MASK);
	udelay(20);
	WREG32_SOC15(UVD, 0, mmUVD_SCRATCH2, 0);
	WREG32_SOC15(UVD, 0, mmUVD_STATUS, 0);
}

static int vcn_v2_0_start(struct amdgpu_vcn_inst *vinst)
'''

patch("vcn_v2_0.c", [
    (PSP_EQ, "AMDGPU_VCN_PSP(adev)", 3),
    ("ring->use_doorbell = true;", "ring->use_doorbell = !amdgpu_bc250_vcn;", 2),
    ("\tif (amdgpu_sriov_vf(adev))\n\t\tvcn_v2_0_start_sriov(adev);\n\n\tr = amdgpu_ring_test_helper(ring);",
     "\tif (amdgpu_sriov_vf(adev))\n\t\tvcn_v2_0_start_sriov(adev);\n\n"
     "\tif (amdgpu_bc250_vcn) {\n"
     "\t\tdev_info(adev->dev, \"bc250 vcn: UVD_STATUS 0x%08x UVD_POWER_STATUS 0x%08x before ungating\\n\",\n"
     "\t\t\t RREG32_SOC15(UVD, 0, mmUVD_STATUS), RREG32_SOC15(UVD, 0, mmUVD_POWER_STATUS));\n"
     "\t\tr = vcn_v2_0_set_pg_state(adev->vcn.inst, AMD_PG_STATE_UNGATE);\n"
     "\t\tif (r)\n\t\t\treturn r;\n"
     "\t}\n\n\tr = amdgpu_ring_test_helper(ring);", 1),
    ("\nstatic int vcn_v2_0_start(struct amdgpu_vcn_inst *vinst)\n", ps5_reset, 1),
    ("\tvcn_v2_0_disable_static_power_gating(vinst);\n\n\t/* set uvd status busy */",
     "\tif (amdgpu_bc250_vcn) {\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_POWER_STATUS, RREG32_SOC15(UVD, 0, mmUVD_POWER_STATUS) & 2);\n"
     "\t\tWREG32_P(SOC15_REG_OFFSET(UVD, 0, mmUVD_SOFT_RESET), UVD_SOFT_RESET__VCPU_SOFT_RESET_MASK,\n"
     "\t\t\t ~UVD_SOFT_RESET__VCPU_SOFT_RESET_MASK);\n"
     "\t\tvcn_v2_0_bc250_reset(adev);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_RBC_RB_CNTL, RREG32_SOC15(UVD, 0, mmUVD_RBC_RB_CNTL) |\n"
     "\t\t\t     UVD_RBC_RB_CNTL__RB_NO_FETCH_MASK | UVD_RBC_RB_CNTL__RB_RPTR_WR_EN_MASK);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_RBC_RB_RPTR, 0);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_RBC_RB_WPTR, 0);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_RB_RPTR, 0);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_RB_WPTR, 0);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_RB_RPTR2, 0);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmUVD_RB_WPTR2, 0);\n"
     "\t\tWREG32_SOC15(UVD, 0, mmJPEG_SYS_INT_STATUS, 0);\n"
     "\t\tdev_info(adev->dev, \"bc250 vcn: start, UVD_POWER_STATUS 0x%08x, VCPU cache 0x%08x_%08x\\n\",\n"
     "\t\t\t RREG32_SOC15(UVD, 0, mmUVD_POWER_STATUS),\n"
     "\t\t\t RREG32_SOC15(UVD, 0, mmUVD_LMI_VCPU_CACHE_64BIT_BAR_HIGH),\n"
     "\t\t\t RREG32_SOC15(UVD, 0, mmUVD_LMI_VCPU_CACHE_64BIT_BAR_LOW));\n"
     "\t} else {\n"
     "\t\tvcn_v2_0_disable_static_power_gating(vinst);\n"
     "\t}\n\n\t/* set uvd status busy */", 1),
    ("\t\t\tmdelay(10);\n\t\t}\n\t\tr = 0;\n\t\tif (status & 2)\n\t\t\tbreak;",
     "\t\t\tmdelay(10);\n\t\t}\n"
     "\t\tif (amdgpu_bc250_vcn && !(status & 2)) {\n"
     "\t\t\tWREG32_P(SOC15_REG_OFFSET(UVD, 0, mmUVD_SOFT_RESET), UVD_SOFT_RESET__VCPU_SOFT_RESET_MASK,\n"
     "\t\t\t\t ~UVD_SOFT_RESET__VCPU_SOFT_RESET_MASK);\n"
     "\t\t\tdev_err(adev->dev, \"bc250 vcn: firmware did not become ready (UVD_STATUS 0x%08x, VCPU cache 0x%08x_%08x)\\n\",\n"
     "\t\t\t\tstatus, RREG32_SOC15(UVD, 0, mmUVD_LMI_VCPU_CACHE_64BIT_BAR_HIGH),\n"
     "\t\t\t\tRREG32_SOC15(UVD, 0, mmUVD_LMI_VCPU_CACHE_64BIT_BAR_LOW));\n"
     "\t\t\treturn -ETIMEDOUT;\n"
     "\t\t}\n"
     "\t\tr = 0;\n\t\tif (status & 2)\n\t\t\tbreak;", 1),
    ("\tr = SOC15_WAIT_ON_RREG(VCN, 0, mmUVD_STATUS, UVD_STATUS__IDLE, 0x7);\n\tif (r)\n\t\treturn r;",
     "\tr = SOC15_WAIT_ON_RREG(VCN, 0, mmUVD_STATUS, UVD_STATUS__IDLE, 0x7);\n"
     "\tif (r && amdgpu_bc250_vcn)\n\t\tgoto bc250_reset;\n\tif (r)\n\t\treturn r;", 1),
    ("\t\tUVD_LMI_STATUS__WRITE_CLEAN_RAW_MASK;\n\tr = SOC15_WAIT_ON_RREG(VCN, 0, mmUVD_LMI_STATUS, tmp, tmp);\n\tif (r)\n\t\treturn r;\n\n\t/* stall UMC channel */",
     "\t\tUVD_LMI_STATUS__WRITE_CLEAN_RAW_MASK;\n\tr = SOC15_WAIT_ON_RREG(VCN, 0, mmUVD_LMI_STATUS, tmp, tmp);\n"
     "\tif (r && amdgpu_bc250_vcn)\n\t\tgoto bc250_reset;\n\tif (r)\n\t\treturn r;\n\n\t/* stall UMC channel */", 1),
    ("\t/* clear status */\n\tWREG32_SOC15(VCN, 0, mmUVD_STATUS, 0);\n\n\tvcn_v2_0_enable_clock_gating(vinst);\n\tvcn_v2_0_enable_static_power_gating(vinst);",
     "\t/* clear status */\n\tWREG32_SOC15(VCN, 0, mmUVD_STATUS, 0);\n\n"
     "\tgoto clock_gating;\n\n"
     "bc250_reset:\n"
     "\tdev_warn(adev->dev, \"bc250 vcn: did not become idle, resetting\\n\");\n"
     "\tWREG32_P(SOC15_REG_OFFSET(UVD, 0, mmUVD_VCPU_CNTL), 0, ~UVD_VCPU_CNTL__CLK_EN_MASK);\n"
     "\tvcn_v2_0_bc250_reset(adev);\n\n"
     "clock_gating:\n"
     "\tvcn_v2_0_enable_clock_gating(vinst);\n\tvcn_v2_0_enable_static_power_gating(vinst);", 1),
])

# Out-of-tree build: the trace header is found next to its .c file (as tools/bc250-amdgpu's patch does).
with open(os.path.join(d, "Makefile"), "a") as f:
    f.write("\n# Out-of-tree build: the trace header is found next to its .c file.\nCFLAGS_amdgpu_trace_points.o := -I$(src)\n")
patch("amdgpu_trace.h", [("#define TRACE_INCLUDE_PATH ../../drivers/gpu/drm/amd/amdgpu", "#define TRACE_INCLUDE_PATH .", 1)])
