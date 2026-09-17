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
available as shell commands.

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

## Verified memory use

Clean build verified on hardware with Arm GNU Toolchain 14.3.1:

- internal Flash: 890,432 B / 1 MiB (84.92%)
- internal SRAM: 141,888 B / 320 KiB (43.30%)
- external SDRAM heap: 8 MiB at `0x60000000`
- QSPI: unused

## Next stages

1. Place large read-only model and NCNN sections in QSPI.
2. Run MobileNetV3-Small on a fixed 96x96 test input.
3. Measure inference time and peak SDRAM use before evaluating FP16 or INT8.
4. Add image input and preprocessing after the memory budget is proven.

YOLOv8n is intentionally deferred because its weights alone are much larger
than the STM32F746's internal Flash.
