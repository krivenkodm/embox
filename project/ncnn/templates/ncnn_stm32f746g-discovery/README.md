# STM32F746G-DISCO NCNN bring-up template

This is the first, hardware-oriented stage of the NCNN port. It deliberately
contains the STM32F746G-DISCO platform, UART shell, external SDRAM heap, and a
small C++ smoke test, but does not link NCNN or model data yet.

Keeping the first image small separates board bring-up failures from NCNN
memory and linker-layout failures.

## Build

The Ubuntu host needs the ARM bare-metal C++ headers and libraries:

```sh
sudo apt install gcc-arm-none-eabi libstdc++-arm-none-eabi-dev \
  libstdc++-arm-none-eabi-newlib
```

When switching from another template, build from a freshly loaded
configuration. `EMBOX_CROSS_COMPILE` is explicit because the Embox
`libstdcxx_toolchain` header-discovery helper reads it during `confload`:

```sh
make confclean
EMBOX_CROSS_COMPILE=arm-none-eabi- \
  make confload-project/ncnn/ncnn_stm32f746g-discovery
EMBOX_CROSS_COMPILE=arm-none-eabi- make -j1
arm-none-eabi-size -A build/base/bin/embox
```

The flashable in-chip image is `build/base/bin/embox.bin`; the ELF file used
by GDB is `build/base/bin/embox`.

## Flash and run

Start OpenOCD in a separate terminal:

```sh
sudo openocd -f board/stm32f746g-disco.cfg
```

Then load the in-chip image from the repository root in another terminal:

```sh
GDB=gdb-multiarch ./scripts/gdb_load_stm32.sh
```

On Ubuntu, `openocd` and `gdb-multiarch` can be installed with APT. Attach the
board's ST-LINK USB device to the UTM virtual machine before starting OpenOCD.
QSPI loading will be added when the NCNN code and model sections are
introduced.

Connect the ST-LINK virtual COM port at 115200 8N1. A successful stage-1
bring-up reaches the Embox shell and runs `stl_demo_sort1` at startup. For
example, on Ubuntu this is commonly `/dev/ttyACM0`:

```sh
picocom -b 115200 /dev/ttyACM0
```

## Planned stages

1. UART shell and C++ runtime (this template).
2. NCNN library with no model and an allocation smoke test.
3. MobileNetV3-Small only; place large read-only sections in QSPI.
4. Measure peak SDRAM use and inference time.
5. Add image input and preprocessing only after the memory budget is proven.

YOLOv8n is intentionally excluded: the current QEMU image contains about
12.7 MiB of YOLO weights in addition to NCNN code and other data.
