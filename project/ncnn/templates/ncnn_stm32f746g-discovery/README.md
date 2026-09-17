# STM32F746G-DISCO NCNN template

This template brings NCNN inference to the STM32F746G-DISCO with a UART shell,
the Cortex-M7 hard-float ABI, and an 8 MiB external SDRAM heap. It includes
three NCNN hardware smoke tests:

- `ncnn_alloc_smoke` allocates and verifies a 768 KiB `ncnn::Mat` in SDRAM.
- `ncnn_inference_smoke` loads a binary NCNN graph, runs a dense 3-to-2 layer
  with ReLU, and checks the expected `[2.500, 0.000]` output in SDRAM.
- `ncnn_conv_smoke` verifies a two-channel `Convolution -> ReLU -> Pooling`
  graph in unpacked FP32, including every intermediate tensor and its SDRAM
  allocation. The 5x5 input uses a ramp and a checkerboard, two 3x3 filters with
  biases, a separate ReLU, and valid 2x2 max pooling with stride 1. Intermediate
  blobs are retained for inspection, so this is not a peak-memory benchmark.

All three NCNN tests run automatically before the `tish` shell starts and remain
available as shell commands. An additional `ncnn_conv_smoke qspi` mode loads
the same graph and weights directly from external Flash; provision its separate
image as described below, then invoke it from the shell.
`ncnn_mobilenet_smoke` runs the full MobileNetV3-Small graph on a fixed 96x96
input using FP32 weights in QSPI and checks all 1000 output logits. Its separate
image preparation and programming instructions are in
[the MobileNet guide](../../ncnn_mobilenet_smoke/README.md).

## Build

Use an Arm GNU Toolchain installation that contains the bare-metal C++ headers,
`libstdc++.a`, and `libsupc++.a`. When switching from another template, load a
fresh configuration and pass the selected toolchain prefix as the make
`CROSS_COMPILE` variable.

On macOS, put GNU coreutils and GNU cpio before the system utilities in
`PATH`. For Apple Silicon Homebrew and the Arm GNU Toolchain 14.3.Rel1 package:

```sh
export PATH="/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/opt/cpio/bin:/opt/homebrew/opt/make/libexec/gnubin:$PATH"
gmake confclean
gmake confload-project/ncnn/ncnn_stm32f746g-discovery \
  CROSS_COMPILE=/Applications/ArmGNUToolchain/14.3.rel1/arm-none-eabi/bin/arm-none-eabi-
gmake -j4 \
  CROSS_COMPILE=/Applications/ArmGNUToolchain/14.3.rel1/arm-none-eabi/bin/arm-none-eabi-
```

On Ubuntu, install `gcc-arm-none-eabi`, `libstdc++-arm-none-eabi-dev`, and
`libstdc++-arm-none-eabi-newlib`, then use `make` and
`CROSS_COMPILE=arm-none-eabi-`.

The flashable image is `build/base/bin/embox.bin`. The ELF file for GDB is
`build/base/bin/embox`.

## Flash and run

With the board connected through the ST-LINK USB connector:

```sh
openocd -f board/stm32f746g-disco.cfg \
  -c "reset_config srst_only srst_nogate connect_assert_srst" \
  -c "program build/base/bin/embox.bin 0x08000000 verify reset exit"
```

Connect to the ST-LINK virtual COM port at 115200 8N1. For example, on macOS:

```sh
picocom -b 115200 --flow n --parity n --databits 8 --stopbits 1 \
  /dev/cu.usbmodem103
```

The exact device suffix may change after reconnecting the board. Press RESET
with `picocom` open to capture all startup output. A successful NCNN inference
ends with:

```text
ncnn_inference_smoke: output=[2.500, 0.000]
ncnn_inference_smoke: PASS dense inference in external SDRAM
ncnn_conv_smoke: output[0]=[0.000, 10.500, 46.500, 52.500]
ncnn_conv_smoke: output[1]=[57.500, 39.500, 3.500, 3.500]
ncnn_conv_smoke: PASS convolution -> ReLU -> pooling in external SDRAM
```

When automating UART input on the tested macOS/ST-LINK V2J31M21 setup,
pace characters (10 ms per character was verified). Sending whole command
lines in a burst stalled host-to-board delivery; resetting the ST-LINK USB
connection restored it. Three consecutive paced runs passed without resetting
the STM32.

## Model in QSPI

The tested board reports JEDEC ID `ef4018` (Winbond W25Q128FV/JV), whereas
ST's BSP flash initialization assumes a Micron N25Q128A. The project-local
reader reuses the BSP GPIO setup and HAL controller API, checks the JEDEC ID,
and maps standard SPI read `03h` at 27 MHz through the QUADSPI peripheral.
It accepts `ef4018` and `20ba18` (Micron); only the Winbond part was tested.
This initial mode uses one data line. It does not change the flash's QE/status
registers, erase/program memory, or require the generic QSPI init module.
The chip must be in standard SPI mode (power-on state and the state used by the
OpenOCD commands below); another part/mode returns an initialization failure.

The 512-byte smoke image occupies the start of the final 64 KiB sector:

- graph: `0x90ff0000`, 224 bytes;
- FP32 tag, weights and biases: `0x90ff0100`, 156 bytes;
- remaining image bytes: `0xff` padding.

The firmware compares both blobs against the compiled fixture before passing
QSPI pointers to NCNN's pointer-based loaders, which do not take buffer lengths.
An absent, corrupted or different model is rejected before parsing. This is a
fixed-fixture hardware test, not a loader for arbitrary models. The compiled
fixture remains available for the baseline test; QSPI mode passes external
addresses to NCNN without staging a copy. Internal layer allocations and input,
intermediate and output tensors use the SDRAM heap. Tensor checks still cover
all 44 expected values. No large NCNN code sections have been moved yet.

Build the image on a little-endian host from the same shared fixture header:

```sh
c++ -std=c++11 -Wall -Wextra -Werror \
  project/ncnn/tools/export_conv_qspi.cpp -o /tmp/export_conv_qspi
/tmp/export_conv_qspi /tmp/ncnn-conv-qspi.bin
```

Before the first write, back up **all 16 MiB** and verify the backup. Choose a
new backup filename and keep it: the final sector may contain factory/user data
on another board. These commands expect OpenOCD's board configuration to expose
`stm32f7x.qspi` as 256 sectors of 64 KiB (confirm its probe output first).
Paths shown below are examples; adapt them without overwriting an old backup.

```sh
openocd -f board/stm32f746g-disco.cfg \
  -c "reset_config srst_only srst_nogate connect_assert_srst" \
  -c "init; reset init; flash probe stm32f7x.qspi; flash read_bank stm32f7x.qspi qspi-before-ncnn.bin 0 0x1000000; reset run; shutdown"
openocd -f board/stm32f746g-disco.cfg \
  -c "reset_config srst_only srst_nogate connect_assert_srst" \
  -c "init; reset init; flash verify_bank stm32f7x.qspi qspi-before-ncnn.bin 0; reset run; shutdown"
```

Then erase **only sector 255**, program and verify the separate model image:

```sh
openocd -f board/stm32f746g-disco.cfg \
  -c "reset_config srst_only srst_nogate connect_assert_srst" \
  -c "init; reset init; flash erase_sector stm32f7x.qspi 255 255; flash write_bank stm32f7x.qspi /tmp/ncnn-conv-qspi.bin 0xff0000; flash verify_bank stm32f7x.qspi /tmp/ncnn-conv-qspi.bin 0xff0000; reset run; shutdown"
```

Internal firmware and model are separate files: never concatenate the QSPI
address gap into `embox.bin`. Flash the internal firmware as above, then run:

```text
embox> ncnn_conv_smoke qspi
ncnn_qspi: JEDEC=ef4018
ncnn_conv_smoke: QSPI fixture verified param=0x90ff0000 model=0x90ff0100
...
ncnn_conv_smoke: PASS model loaded from QSPI
```

To restore only the modified sector, extract the last 64 KiB from the original
full backup, erase sector 255, write that sector image at offset `0xff0000`,
and verify it with `flash verify_bank`. Other sectors need no erasure.

## Verified memory use

Clean build verified on hardware with Arm GNU Toolchain 14.3.1
(profiling plus the single/quad QSPI comparison):

- internal Flash: 908,984 B / 1 MiB (86.69%)
- internal SRAM: 141,888 B / 320 KiB (43.30%)
- external SDRAM heap: 8 MiB at `0x60000000`; measured MobileNet peak
  1,256,256 B of allocated pages, with 7,115,904 B free at peak and full release
- QSPI images: 10,156,800 B for MobileNet at `0x90500000`, plus the
  512 B convolution fixture at `0x90ff0000` (separate from the ELF)

## Next stages

1. Use `ncnn_mobilenet_smoke quad` for the verified Winbond quad-data mode:
   about 0.752 s CRC and 3.242 s inference versus 3.010 s and 8.190 s single-line
   (medians of three paired runs at 27 MHz). See the MobileNet README for limits.
   Evaluate real-image preprocessing and FP16/INT8 tradeoffs from this FP32 baseline.
2. Move large read-only NCNN sections only if the internal Flash budget requires it.
3. Add real image input and preprocessing after the memory budget is proven.

YOLOv8n is intentionally deferred because its weights alone are much larger
than the STM32F746's internal Flash.
