/*
 * QEMU Apple paravirtualized graphics device, MMIO (arm64) variant
 *
 * Copyright © 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * macOS guests drive a paravirtualized GPU, the device of Apple's
 * ParavirtualizedGraphics.framework, which implements 3d graphics passthrough
 * to the host as well as a proprietary guest communication channel. This
 * device model drives reims-vgpu, a library that implements the host side of
 * that device, from within QEMU as an MMIO-based system device for macOS on
 * arm64 VMs. Its second MMIO region is the IOSurface device, which maps the
 * guest's IOSurfaces.
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/error-report.h"
#include "qemu/main-loop.h"
#include "hw/sysbus.h"
#include "hw/irq.h"
#include "migration/vmstate.h"
#include "system/runstate.h"
#include "apple-gfx.h"
#include "trace.h"

OBJECT_DECLARE_SIMPLE_TYPE(AppleGFXMMIOState, APPLE_GFX_MMIO)

struct AppleGFXMMIOState {
    SysBusDevice parent_obj;

    AppleGFXState common;

    qemu_irq irq_gfx;
    qemu_irq irq_iosfc;
    MemoryRegion iomem_iosfc;
    RvgIOSurfaceDevice *pgiosfc;
    /* log2 of the guest's page size; arm64 guests use 16 KiB pages */
    uint32_t guest_page_shift;
    bool process_isolation;

    /*
     * Migration (vphone-linux fast-resume). reims serializes the graphics
     * device's state (FIFO and display registers, device info) into a small
     * blob; the guest's GPU objects live in guest RAM, which QEMU migrates.
     * The IOSurface device has no suspend API of its own, so its setup
     * registers are snooped from the guest's writes and replayed onto the
     * freshly created device on resume, which rebuilds its ring and table
     * from the (restored) guest memory.
     */
    uint8_t *mig_blob;
    uint32_t mig_blob_len;
    uint32_t iosfc_ring_base;
    uint32_t iosfc_ring_size;
    uint32_t iosfc_descriptor_base;
    uint32_t iosfc_desc_enable;
    uint32_t iosfc_ring_wr;
    uint32_t iosfc_ring_rd;
    bool mig_suspended;
};

/* The IOSurface device's MMIO register offsets (reims iosurface/regs.rs). */
#define APPLE_GFX_IOSFC_RING_BASE       0x1000
#define APPLE_GFX_IOSFC_RING_SIZE       0x1008
#define APPLE_GFX_IOSFC_DESCRIPTOR_BASE 0x1010
#define APPLE_GFX_IOSFC_RING_WR_COUNTER 0x1018
#define APPLE_GFX_IOSFC_RING_RD_COUNTER 0x1020
#define APPLE_GFX_IOSFC_DESC_ENABLE     0x1030

static void apple_gfx_mmio_vm_state_change(void *opaque, bool running,
                                           RunState state);

/*
 * The IOSurface device's registers are 64 bits wide, and QEMU reads and
 * writes them as 32: a value read is truncated, then zero-extended, and a
 * value written is truncated. The vCPU calls the library itself, as
 * apple-gfx.c's accesses do: an access waits only on the IOSurface device's
 * thread, whose one callback schedules a BH.
 */
static uint64_t iosfc_read(void *opaque, hwaddr offset, unsigned size)
{
    AppleGFXMMIOState *s = opaque;
    uint64_t value = (uint32_t)rvg_iosurface_device_mmio_read(s->pgiosfc,
                                                              offset);

    trace_apple_gfx_mmio_iosfc_read(offset, value);
    return value;
}

static void iosfc_write(void *opaque, hwaddr offset, uint64_t val,
                        unsigned size)
{
    AppleGFXMMIOState *s = opaque;

    /*
     * Snoop the setup registers so a resume can replay them onto a fresh
     * IOSurface device (it has no suspend state of its own). The read
     * counter is the device's, read back at save time.
     */
    switch (offset) {
    case APPLE_GFX_IOSFC_RING_BASE:       s->iosfc_ring_base = (uint32_t)val; break;
    case APPLE_GFX_IOSFC_RING_SIZE:       s->iosfc_ring_size = (uint32_t)val; break;
    case APPLE_GFX_IOSFC_DESCRIPTOR_BASE: s->iosfc_descriptor_base = (uint32_t)val; break;
    case APPLE_GFX_IOSFC_RING_WR_COUNTER: s->iosfc_ring_wr = (uint32_t)val; break;
    case APPLE_GFX_IOSFC_DESC_ENABLE:     s->iosfc_desc_enable = (uint32_t)val; break;
    default: break;
    }

    rvg_iosurface_device_mmio_write(s->pgiosfc, offset, (uint32_t)val);
    trace_apple_gfx_mmio_iosfc_write(offset, val);
}

static const MemoryRegionOps apple_iosfc_ops = {
    .read = iosfc_read,
    .write = iosfc_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
};

static void raise_irq_bh(void *opaque)
{
    qemu_irq *irq = opaque;

    qemu_irq_pulse(*irq);
}

/* On the IOSurface device's own thread. */
static void apple_gfx_mmio_iosfc_raise_interrupt_cb(void *ctx, uint32_t vector)
{
    AppleGFXMMIOState *s = ctx;

    trace_apple_gfx_iosfc_raise_irq(vector);
    aio_bh_schedule_oneshot(qemu_get_aio_context(),
                            raise_irq_bh, &s->irq_iosfc);
}

static bool apple_gfx_mmio_create_iosurface_device(AppleGFXMMIOState *s,
                                                   Error **errp)
{
    /*
     * The library copies the ranges and duplicates the files' descriptors.
     * vphone-linux: the IOSurface device must get the files (the guest-RAM
     * memfd behind each range), as the graphics device does, or on Linux it
     * cannot map the guest's IOSurface descriptor table and ring (the display
     * stays black). Same ranges as apple_gfx_common_realize built, so the
     * memory-backend-memfd check there has already passed.
     */
    g_autoptr(GArray) memory_files = NULL;
    g_autoptr(GArray) memory_ranges = apple_gfx_new_memory_ranges(&memory_files);
    const RvgMemoryRangeFile *files = NULL;
    RvgMemoryMap memory_map = {
        .ranges = (const RvgMemoryRange *)memory_ranges->data,
        .range_count = memory_ranges->len,
    };
    RvgIOSurfaceCallbacks callbacks = {
        .struct_size = sizeof(callbacks),
        .ctx = s,
        .raise_interrupt = apple_gfx_mmio_iosfc_raise_interrupt_cb,
    };
    RvgStatus status;

    if (!apple_gfx_memory_files(memory_ranges, memory_files, &files, errp)) {
        return false;
    }

    RvgIOSurfaceDeviceConfig config = {
        .struct_size = sizeof(config),
        .memory_map = &memory_map,
        .callbacks = &callbacks,
        .guest_page_shift = s->guest_page_shift,
        .memory_files = files,
    };

    status = rvg_iosurface_device_create(&config, &s->pgiosfc);
    if (status != RVG_OK) {
        error_setg(errp, "apple-gfx-mmio: rvg_iosurface_device_create: %s",
                   rvg_status_message(status));
        return false;
    }
    return true;
}

/*
 * On a library thread (a FIFO's or the device's reporting thread), or inside
 * a call that may call back: an MMIO access, on the vCPU's thread, or
 * apple_gfx_mmio_reset's.
 */
static void apple_gfx_mmio_raise_interrupt_cb(void *ctx, uint32_t vector)
{
    AppleGFXMMIOState *s = ctx;

    trace_apple_gfx_raise_irq(vector);
    aio_bh_schedule_oneshot(qemu_get_aio_context(),
                            raise_irq_bh, &s->irq_gfx);
}

static void apple_gfx_mmio_realize(DeviceState *dev, Error **errp)
{
    AppleGFXMMIOState *s = APPLE_GFX_MMIO(dev);
    RvgHostCallbacks callbacks = {
        .struct_size = sizeof(callbacks),
        .ctx = s,
        .raise_interrupt = apple_gfx_mmio_raise_interrupt_cb,
    };
    RvgDeviceConfig config = {
        .struct_size = sizeof(config),
        /* The IOSurface device made below maps the guest's IOSurfaces. */
        .use_iosurface_mapper = true,
        .process_isolation = s->process_isolation,
        .callbacks = &callbacks,
        .guest_page_shift = s->guest_page_shift,
    };

    /* This variant drives reims suspend/resume via its VMStateDescription. */
    s->common.migratable = true;

    if (!apple_gfx_common_realize(&s->common, dev, &config, errp)) {
        return;
    }

    apple_gfx_mmio_create_iosurface_device(s, errp);

    /*
     * Suspend reims on the stop for a save/migrate, before RAM is captured, so
     * finish_suspend's writeback into guest memory is part of the saved state.
     */
    qemu_add_vm_change_state_handler(apple_gfx_mmio_vm_state_change, s);
}

static void apple_gfx_mmio_init(Object *obj)
{
    AppleGFXMMIOState *s = APPLE_GFX_MMIO(obj);

    apple_gfx_common_init(obj, &s->common, TYPE_APPLE_GFX_MMIO);

    sysbus_init_mmio(SYS_BUS_DEVICE(s), &s->common.iomem_gfx);
    memory_region_init_io(&s->iomem_iosfc, obj, &apple_iosfc_ops, s,
                          TYPE_APPLE_GFX_MMIO, 0x10000);
    sysbus_init_mmio(SYS_BUS_DEVICE(s), &s->iomem_iosfc);
    sysbus_init_irq(SYS_BUS_DEVICE(s), &s->irq_gfx);
    sysbus_init_irq(SYS_BUS_DEVICE(s), &s->irq_iosfc);
}

static void apple_gfx_mmio_finalize(Object *obj)
{
    AppleGFXMMIOState *s = APPLE_GFX_MMIO(obj);

    apple_gfx_common_finalize(&s->common);
}

static void apple_gfx_mmio_reset(Object *obj, ResetType type)
{
    AppleGFXMMIOState *s = APPLE_GFX_MMIO(obj);

    rvg_device_reset(s->common.pgdev);
}

/*
 * Migration (vphone-linux fast-resume). On save, reims serializes the
 * graphics device into a blob and we read the IOSurface ring's positions; the
 * guest's GPU objects ride in guest RAM. On load, the IOSurface device (freshly
 * created in realize) is replayed its setup registers so it rebuilds its ring
 * and table from the restored guest memory, then the graphics device resumes
 * from the blob.
 */
/*
 * Suspend reims: serialize the graphics device into the blob and read the
 * IOSurface ring's positions. finish_suspend also writes the device's resume
 * state (FIFO fault bits, task roots) back into guest memory, so this must run
 * BEFORE QEMU captures RAM -- the vm-state-change handler does it on
 * FINISH_MIGRATE/SAVE_VM, ahead of the RAM save; pre_save is only a fallback.
 * Idempotent: the handler and pre_save both call it, the second a no-op.
 */
static void apple_gfx_mmio_do_suspend(AppleGFXMMIOState *s)
{
    RvgBytes blob = { NULL, 0 };

    if (s->mig_suspended) {
        return;
    }
    rvg_device_will_suspend(s->common.pgdev);
    if (rvg_device_finish_suspend(s->common.pgdev, &blob) != RVG_OK) {
        error_report("apple-gfx-mmio: rvg_device_finish_suspend failed");
        return;
    }
    g_free(s->mig_blob);
    s->mig_blob = g_memdup2(blob.data, blob.length);
    s->mig_blob_len = blob.length;
    rvg_bytes_free(blob);

    s->iosfc_ring_wr = (uint32_t)rvg_iosurface_device_mmio_read(
        s->pgiosfc, APPLE_GFX_IOSFC_RING_WR_COUNTER);
    s->iosfc_ring_rd = (uint32_t)rvg_iosurface_device_mmio_read(
        s->pgiosfc, APPLE_GFX_IOSFC_RING_RD_COUNTER);
    s->mig_suspended = true;
}

/* Undo a suspend, on the destination after a load or on a source that continues. */
static void apple_gfx_mmio_do_resume(AppleGFXMMIOState *s)
{
    RvgResumeError err = RVG_RESUME_ERROR_NONE;

    if (!s->mig_suspended) {
        return;
    }
    /*
     * Rebuild the IOSurface ring and table before resuming the graphics
     * device, whose will_resume..did_resume window forbids new guest mappings.
     * Order matters (reims iosurface/regs.rs): a ring base unmaps the ring
     * until the size is written, and the read counter is honoured only while
     * no ring is mapped. The surfaces themselves are rebuilt from the table.
     */
    iosfc_write(s, APPLE_GFX_IOSFC_DESC_ENABLE, s->iosfc_desc_enable, 4);
    iosfc_write(s, APPLE_GFX_IOSFC_DESCRIPTOR_BASE, s->iosfc_descriptor_base, 4);
    rvg_iosurface_device_rebuild_surfaces(s->pgiosfc);
    iosfc_write(s, APPLE_GFX_IOSFC_RING_BASE, s->iosfc_ring_base, 4);
    iosfc_write(s, APPLE_GFX_IOSFC_RING_RD_COUNTER, s->iosfc_ring_rd, 4);
    iosfc_write(s, APPLE_GFX_IOSFC_RING_SIZE, s->iosfc_ring_size, 4);
    iosfc_write(s, APPLE_GFX_IOSFC_RING_WR_COUNTER, s->iosfc_ring_wr, 4);

    if (rvg_device_will_resume(s->common.pgdev, s->mig_blob, s->mig_blob_len,
                               &err) != RVG_OK) {
        error_report("apple-gfx-mmio: rvg_device_will_resume failed: %u",
                     (unsigned)err);
        return;
    }
    rvg_device_did_resume(s->common.pgdev);

    g_free(s->mig_blob);
    s->mig_blob = NULL;
    s->mig_blob_len = 0;
    s->mig_suspended = false;
}

static void apple_gfx_mmio_vm_state_change(void *opaque, bool running,
                                           RunState state)
{
    AppleGFXMMIOState *s = opaque;

    if (!running &&
        (state == RUN_STATE_FINISH_MIGRATE || state == RUN_STATE_SAVE_VM)) {
        /* Before RAM is captured: finish_suspend's writeback is then migrated. */
        apple_gfx_mmio_do_suspend(s);
    } else if (running && s->mig_suspended) {
        /* The source kept running (migration did not hand off); undo it. */
        apple_gfx_mmio_do_resume(s);
    }
}

static int apple_gfx_mmio_pre_save(void *opaque)
{
    AppleGFXMMIOState *s = opaque;

    apple_gfx_mmio_do_suspend(s); /* fallback; normally the handler already did */
    return s->mig_blob ? 0 : -1;
}

static int apple_gfx_mmio_post_load(void *opaque, int version_id)
{
    AppleGFXMMIOState *s = opaque;

    /* The destination loaded a suspended device; resume it into the restored state. */
    s->mig_suspended = true;
    apple_gfx_mmio_do_resume(s);
    return 0;
}

static const VMStateDescription vmstate_apple_gfx_mmio = {
    .name = "apple-gfx-mmio",
    .version_id = 1,
    .minimum_version_id = 1,
    .pre_save = apple_gfx_mmio_pre_save,
    .post_load = apple_gfx_mmio_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(mig_blob_len, AppleGFXMMIOState),
        VMSTATE_VBUFFER_ALLOC_UINT32(mig_blob, AppleGFXMMIOState, 1, NULL,
                                     mig_blob_len),
        VMSTATE_UINT32(iosfc_ring_base, AppleGFXMMIOState),
        VMSTATE_UINT32(iosfc_ring_size, AppleGFXMMIOState),
        VMSTATE_UINT32(iosfc_descriptor_base, AppleGFXMMIOState),
        VMSTATE_UINT32(iosfc_desc_enable, AppleGFXMMIOState),
        VMSTATE_UINT32(iosfc_ring_wr, AppleGFXMMIOState),
        VMSTATE_UINT32(iosfc_ring_rd, AppleGFXMMIOState),
        VMSTATE_END_OF_LIST()
    }
};

static const Property apple_gfx_mmio_properties[] = {
    DEFINE_PROP_ARRAY("display-modes", AppleGFXMMIOState,
                      common.num_display_modes, common.display_modes,
                      qdev_prop_apple_gfx_display_mode, AppleGFXDisplayMode),
    /* The guest's display ports, 1 to RVG_MAX_DISPLAY_PORTS. */
    DEFINE_PROP_UINT32("displays", AppleGFXMMIOState, common.num_displays, 1),
    /*
     * In window mode, the mode the guest runs first, scaled by the window:
     * "<width>x<height>" (default 1920x1080), at 1x or followed by "@<ppi>",
     * the density the guest is told for it, or "host" for the host screen's
     * own modes at its density.
     */
    DEFINE_PROP_STRING("resolution", AppleGFXMMIOState, common.resolution),
    /*
     * The library's default is its own build's page size; on an x86_64 host
     * that would be 4 KiB, while every guest of this arm64 device uses 16.
     */
    /*
     * Each user task in its own host process, as on macOS. On Linux that needs
     * every guest RAM range mapped in whole guest pages; turn it off to run
     * the tasks in QEMU's process.
     */
    DEFINE_PROP_BOOL("process-isolation", AppleGFXMMIOState,
                     process_isolation, true),
    DEFINE_PROP_UINT32("guest-page-shift", AppleGFXMMIOState,
                       guest_page_shift, 14),
};

static void apple_gfx_mmio_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);

    rc->phases.hold = apple_gfx_mmio_reset;
    dc->hotpluggable = false;
    dc->realize = apple_gfx_mmio_realize;
    dc->vmsd = &vmstate_apple_gfx_mmio;

    device_class_set_props(dc, apple_gfx_mmio_properties);
}

static const TypeInfo apple_gfx_mmio_types[] = {
    {
        .name          = TYPE_APPLE_GFX_MMIO,
        .parent        = TYPE_SYS_BUS_DEVICE,
        .instance_size = sizeof(AppleGFXMMIOState),
        .class_init    = apple_gfx_mmio_class_init,
        .instance_init = apple_gfx_mmio_init,
        .instance_finalize = apple_gfx_mmio_finalize,
    }
};
DEFINE_TYPES(apple_gfx_mmio_types)
