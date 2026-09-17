# STM32F746 NCNN LCD dashboard

The STM32 NCNN template now initializes the existing ST board BSP/LTDC at boot
and shows `DISPLAY READY`. It uses a project-local static adapter and CPU-only
RGB565 drawing, without changing the common Embox video driver or vendor BSP.
Keep BSP/HAL outside every App `BuildDepends` closure so command initialization
cannot restore live peripheral state or `SystemCoreClock`.

From `embox>`:

```text
ncnn_lcd_smoke colors
ncnn_lcd_smoke ready
ncnn_mobilenet_smoke quad photo
```

The first command displays eight solid color bars; `ready` restores the startup
screen. Photo mode shows the decoded JPEG and `RUNNING...`, then the actual
top-1 ImageNet label, inference time and `PASS` only after the existing 1000
output and heap-release checks succeed. Failures display `FAILED` and leave
details in UART. Photo inference is still manual; powering up alone shows the
readiness screen. No camera or automatic inference is added in this stage.

The on-screen photo preserves its original aspect ratio using nearest-neighbor
scaling. This is separate from the unchanged, CRC-verified NCNN preprocessing:
224x224 bilinear square resize, RGB planar FP32, scaling by 1/255, and graph
mean/std normalization. Display scaling never feeds back into inference.

## Memory and cache

| Region | Address | Bytes |
| --- | --- | ---: |
| 480x272 RGB565 frame | `0x60000000..0x6003fbff` | 261120 |
| Guard filled with A5 | `0x6003fc00..0x6003ffff` | 1024 |
| Fixed external heap, including allocator control | `0x60040000..0x607fffff` | 8126464 |

The first 256 KiB is an aligned MPU normal, non-cacheable region; the remainder
retains caching for NCNN. CPU writes to the frame are followed by a data barrier.
The frame is never allocated from or freed into the NCNN heap. All four NCNN
smoke tests reject tensors below the new heap boundary. The adapter checks the
heap allocator's placement before touching the reserved memory.

The existing ST BSP supplies panel pins, 9.6 MHz pixel clock and timing. Layer 0
uses RGB565; layer 1 is disabled. This static dashboard polls latched LTDC
transfer/underrun flags and uses no display IRQ handler. Frame CRC and the entire
guard must remain intact across inference, with HCLK still 216 MHz. A failed
check fails the command. A single frame is updated in place, so transient
tearing during redraw is possible; this is not a double-buffered animation UI.
Cache and bandwidth considerations follow ST's
[AN4861](https://www.st.com/resource/en/application_note/an4861-lcdtft-display-controller-ltdc-on-stm32-mcus-stmicroelectronics.pdf).

The profiler reports **heap pages only**, excluding the separate 262144-byte
display reservation and allocator control. Add the display reservation to heap
peak when budgeting SDRAM. Neither QSPI model nor status registers are written.

## Verified on 2026-09-17

Clean Arm GNU 14.3.1 build and internal Flash programming (`Verified OK`) passed.
Flash is 1010888 B (96.41%, 37688 B remaining); internal SRAM is 141920 B
(43.31%). The image adds 37008 B of Flash and 32 B of linker-reported SRAM
versus the photo-only firmware. Reserved SDRAM is separate from linker SRAM.

Four startup tests, both LCD commands, both synthetic QSPI modes, QSPI
convolution, dense inference and allocation passed. Three photo runs with the
display continuously active checked all 1000 outputs, maximum reported error
0.000028, top-1 `Egyptian cat` (class 285):

| Measurement | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Input preparation including display, ms | 236 | 236 | 236 |
| Inference, ms | 15349 | 15347 | 15356 |
| Total through heap restoration, ms | 16564 | 16562 | 16570 |
| Peak heap pages, bytes | 6067008 | 6067008 | 6067008 |
| Minimum free heap pages, bytes | 2043520 | 2043520 | 2043520 |
| Baseline / final allocated pages, bytes | 0 / 0 | 0 / 0 | 0 / 0 |

Usable heap page capacity is 8110528 B; allocator control occupies 15936 B.
Heap peak plus the full display reservation is 6329152 B (about 6.04 MiB),
excluding allocator control. No failed or other-heap page allocations occurred.
The total timer ends before final result drawing and its LCD CRC checks; UART
command wall time was about 17.06 s including paced input and final diagnostics.
Inference median is 15.349 s versus the earlier photo-only 14.815 s. This is a
comparison of two builds, not a controlled same-build display-on/off benchmark.

The complete 261120-byte frame read from SDRAM after run 3 matched native
rendering pixel for pixel; all 1024 guard bytes remained A5. The final frame
CRC32 was `0a8f2f7d` and displayed time was 15.356 s. Independent register reads
confirmed RGB565, layer enabled, correct frame address/stride/height, the MPU
uncached region, enabled CPU caches, zero LTDC errors and advancing scan position.
These are electrical/controller and framebuffer checks, not a camera photograph
of the physical panel. Early LCD printf text can be truncated before the UART
console is selected; shell-command diagnostics were captured completely.

## Native renderer test

Use `decoded.rgb` from the [photo reference export](../ncnn_mobilenet_smoke/README.md).
From the Embox root on a little-endian host:

```sh
cc -std=c11 -Wall -Wextra -Werror -g -fsanitize=address,undefined -Isrc \
  project/ncnn/tools/check_lcd_render.c \
  project/ncnn/ncnn_lcd_smoke/lcd_render.c src/drivers/video/fonts.c \
  -o /tmp/check_lcd_render
/tmp/check_lcd_render /tmp/mobilenet-photo/decoded.rgb /tmp/lcd-host.rgb565 15356
```

This verifies all pixels of eight reference color bars and buffer guards around
ready/photo/result/error/long-label rendering under ASan/UBSan. The raw output
can be compared against an LCD memory dump with the same displayed time. Adjust
the final argument to the actual inference time when repeating on the board.
