# MobileNetV3-Small on STM32F746G-DISCO

`ncnn_mobilenet_smoke` runs the repository's existing 138-layer NCNN model on a
fixed synthetic RGB input, 96x96x3, and compares all 1000 output logits with a
native scalar-NCNN reference. The input value is
`((3*x + 5*y + 17*channel) % 256) / 255`; mean/std normalization is already in
the graph. This verifies inference and storage, not classification accuracy
on a dataset or an optimized memory/performance budget. An optional `photo`
mode below decodes a real JPEG on the MCU and verifies the 224x224 pipeline.

The source arrays are the existing `models/mobilenetv3_small/assets` headers.
Their 5,095,776-byte weight stream contains FP16 storage. NCNN's generic FP32
loader expands those weights, which would exceed the 8 MiB SDRAM heap if the
whole model were retained there. The host converter uses each NCNN layer's
`load_model` implementation to emit the same weights as FP32. All 1000 logits
from the original and converted streams must match exactly on the host.
This changes storage precision, not the information in the trained weights.

The converted weights are 10,147,440 bytes. NCNN references 10,147,224 bytes
of FP32 data directly in memory-mapped QSPI; the remaining 216 bytes are weight
tags. Runtime objects and activations use SDRAM. Packing, FP16 arithmetic/storage,
BF16 and INT8 inference are disabled; light mode releases intermediate blobs.
The final output and input are checked to lie fully within the SDRAM heap.

## Image layout

The separate `mobilenet-qspi.bin` image is 10,156,800 bytes, programmed at QSPI
offset `0x500000` (CPU address `0x90500000`). It uses sectors **80 through 234**
of the OpenOCD bank's 256 sectors of 64 KiB. The first 5 MiB and the convolution
fixture at `0x90ff0000` remain untouched. The reserved sectors were all erased
on the tested board; inspect and back up another board before using them.

| Region | Image offset | Size |
| --- | ---: | ---: |
| Fixed header | 0 | 64 B |
| Binary graph | 64 | 9,296 B |
| FP32 weights | 9,360 | 10,147,440 B |

The header contains 16 little-endian uint32 words: magic `0x33564e4d`, version,
image length, graph offset/length, weight offset/length, graph CRC32, weight
CRC32, input size, input/output blob indexes, reference CRC32, original weight
CRC32, weight-tag bytes and a reserved zero. The firmware requires an exact
header match and checks both payload CRCs before parsing. `BoundedReader`
limits reads and references to each blob, and exact consumption is checked.
Only this fixed, checked model is supported; this is not arbitrary-model loading.
NCNN's graph parser is not relied on to recover from malformed parameters.

## Reproduce the image and reference

Use Python 3, CMake, Ninja and a native C++ compiler. The source archive is the
same pinned NCNN `20250916` used by Embox (MD5
`da74ad0af613a8dd4d9a4a02a234ecef`). Build the portable backend on the host so
its operations correspond to the MCU backend. The following commands are for
macOS, from the Embox root; on Linux set `CMAKE_SYSTEM_NAME` to `Linux`.
Choose fresh temporary directories if these names already contain other work.

```sh
mkdir -p /tmp/mobilenet-host-src /tmp/mobilenet-export
LC_ALL=C tar -xzf download/ncnn-20250916.tar.gz -C /tmp/mobilenet-host-src
patch -d /tmp/mobilenet-host-src/ncnn-20250916 -p1 \
  -i "$PWD/third-party/lib/ncnn/patches/0003-cortex-m-generic-backend.patch"
cat > /tmp/mobilenet-host.cmake <<'CMAKE'
set(CMAKE_SYSTEM_NAME Darwin)
set(CMAKE_SYSTEM_PROCESSOR cortex-m7)
CMAKE
cmake -S /tmp/mobilenet-host-src/ncnn-20250916 -B /tmp/mobilenet-host-build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/tmp/mobilenet-host.cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-ffp-contract=off \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF \
  -DNCNN_RUNTIME_CPU=OFF -DNCNN_C_API=OFF
cmake --build /tmp/mobilenet-host-build -j4
c++ -std=c++11 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  -I/tmp/mobilenet-host-src/ncnn-20250916/src -I/tmp/mobilenet-host-build/src \
  project/ncnn/tools/prepare_mobilenet_qspi.cpp \
  /tmp/mobilenet-host-build/src/libncnn.a -o /tmp/prepare_mobilenet_qspi
python3 project/ncnn/tools/export_mobilenet_qspi.py \
  --converter /tmp/prepare_mobilenet_qspi --output-dir /tmp/mobilenet-export
cmp /tmp/mobilenet-export/manifest.h project/ncnn/ncnn_mobilenet_smoke/manifest.h
cmp /tmp/mobilenet-export/reference.h project/ncnn/ncnn_mobilenet_smoke/reference.h
```

The exporter requires the pinned source-array SHA-256 values and exact expected
sizes, compares the original and converted inference outputs, then writes the
image, generated headers and JSON manifest. Keep the committed reference when
porting to another host compiler; small FP32 differences may require inspecting
the generated reference comparison. The board comparison allows
`abs(actual-reference) <= 0.002 + 0.0002 * abs(reference)` for each logit and
rejects nonfinite results.

Verified image SHA-256:
`035a785e94530830171ecb0839a79178b9681aed790bd7d80f62ac9ee76998c8`.

## Program and run

First build and flash the internal firmware using the STM32 template README.
Create and verify a fresh full 16 MiB QSPI backup using its backup commands;
keep that backup in persistent storage. Confirm sectors 80..234 can be used.
Erase only those sectors and program the separate image:

```sh
openocd -f board/stm32f746g-disco.cfg \
  -c "reset_config srst_only srst_nogate connect_assert_srst" \
  -c "init; reset init; flash erase_sector stm32f7x.qspi 80 234; flash write_bank stm32f7x.qspi /tmp/mobilenet-export/mobilenet-qspi.bin 0x500000; flash verify_bank stm32f7x.qspi /tmp/mobilenet-export/mobilenet-qspi.bin 0x500000; reset run; shutdown"
```

Programming and verification can take several minutes through ST-LINK. The ELF
linker reports no QSPI usage because this model is a separate image. Never
append the QSPI address gap to the internal firmware binary.

Run `ncnn_mobilenet_smoke` from the UART shell. It is intentionally manual so a
fresh board without the external image still boots the existing tests. Missing
headers or mismatched payload checksums return a failure before NCNN parsing.
After the reference comparison the command prints the top five raw logits
(not softmax probabilities), `extract` time and a final PASS after heap restore.
Extraction time excludes CRC validation, model loading and input preparation.

To restore the previous contents, extract bytes `0x500000..0xeaffff` from the
full backup into a sector image, erase only sectors 80..234, write that image
at bank offset `0x500000`, and verify it. Other sectors require no erasure.

## Verified on the board

Arm GNU Toolchain 14.3.1, NCNN 20250916 generic backend, Cortex-M7 at 216 MHz,
QSPI standard read at 27 MHz, one thread. Three consecutive runs after reset
passed all 1000 logit comparisons, with maximum absolute error about 0.000017.
Their `extract` times were 8179, 8181 and 8182 ms. Full command wall time was
about 11.63 s including CRC checks and model setup (plus paced UART input).
Top-five class indexes were 905, 858, 789, 854 and 421; top logit was 9.468515.
These are results on the synthetic fixture, not recognition-quality metrics.

The four original startup tests passed. After the three MobileNet runs, the
convolution QSPI, dense inference and 768 KiB allocation tests also passed.
An absent MobileNet image was rejected before parsing. Full 16 MiB verification
against the original image plus the new payload confirmed other QSPI contents
were preserved. The exporter reproduced the image and generated headers exactly.

The original uninstrumented firmware occupied 907,064 B / 1 MiB (86.50%);
internal SRAM was 141,888 B / 320 KiB (43.30%). See the measured profile below
for the instrumented firmware and SDRAM peak.

## Memory and timing profile

The same `ncnn_mobilenet_smoke` command now prints `timing_ms` and `SDRAM bytes`
records before PASS. It still validates all 1000 logits on every run.

The STM32 template links project-local `--wrap=page_alloc` and
`--wrap=page_alloc_zero` hooks. They sample the external page allocator's used
bytes after every allocation while this command is active. This includes brief
workspace peaks, graph objects, activations, heap segment metadata, alignment
and unused space in allocated segments. `capacity` excludes the fixed page
allocator control area at the start of the 8 MiB SDRAM; `page` gives the
measurement granularity. `baseline`, `peak` and `final` are absolute allocated
page bytes, and `min_free = capacity - peak`. These are not tensor payload sizes
or a largest-contiguous-free-block measurement. Existing allocations and any
other tasks using this allocator during the command are included.

Before measurement a self-test allocates one page plus two zeroed pages, checks
the observed peaks, frees both, and requires the original free count. A missing
linker hook therefore cannot silently produce a zero peak. The command fails
if an external page allocation failed or final usage differs from its baseline.
`other_heap_allocs` counts successful page-allocation observations outside the
external heap during the interval; it helps identify fallback or concurrent
system activity, but does not measure byte allocations inside existing segments.
No heap implementation, common build scripts or QSPI contents are changed.

Timing uses `CLOCK_MONOTONIC` in milliseconds; a zero means less than one timer
interval. The fields are:

| Field | Measured work |
| --- | --- |
| `qspi` | QSPI setup, including the existing JEDEC diagnostic |
| `crc` | Graph and weight CRC32 validation |
| `param` | Net construction, options and binary graph loading |
| `model` | Weight loading and pipeline setup |
| `input` | Input allocation/fill (or JPEG decode, resize and CRCs), extractor construction and input binding |
| `extract` | NCNN inference |
| `verify` | Output shape/location, 1000 reference comparisons and top-five selection |
| `cleanup` | Destruction of the output, extractor, input and Net |
| `total` | QSPI initialization through heap restoration, including intervening UART output |

`total` excludes typed command delivery, the profiler self-test and the final
profile/PASS diagnostics. It need not equal the sum of stage times because
intervening output is included and each stage is rounded down to milliseconds.
The heap peak includes the timing diagnostic after heap restoration. The hooks
and clocks add measurement overhead; compare read modes with the same profiling
firmware and report the uninstrumented reference separately.

### Profile verified on 2026-09-17

Three consecutive runs after flashing and reset passed all 1000 output values
(maximum reported absolute error 0.000017) and the allocation-hook self-test:

| Measurement | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| QSPI initialization, ms | 2 | 2 | 2 |
| CRC validation, ms | 3010 | 3010 | 3010 |
| Graph loading, ms | 91 | 91 | 91 |
| Weight/pipeline loading, ms | 2 | 2 | 2 |
| Input/extractor preparation, ms | 4 | 4 | 4 |
| Inference, ms | 8188 | 8185 | 8191 |
| Output verification, ms | 0 | 0 | 0 |
| Cleanup, ms | 29 | 29 | 29 |
| Total through heap restoration, ms | 11385 | 11382 | 11388 |
| Peak allocated SDRAM pages, bytes | 1256256 | 1256256 | 1256256 |
| Baseline / final allocated bytes | 0 / 0 | 0 / 0 | 0 / 0 |

Usable SDRAM page capacity is 8,372,160 bytes, with 64-byte pages and 16,448
bytes reserved for allocator control. Peak allocation is about 1.198 MiB;
minimum free capacity is 7,115,904 bytes (about 6.786 MiB). All runs reported
zero failed external page allocations and zero other-heap page allocations.
These figures apply to this fixed 96x96 FP32 fixture; they do not establish the
memory requirement of larger inputs or camera/display buffers.

The four startup tests and the convolution-QSPI, dense and 768 KiB allocation
tests after profiling all passed. Disassembly confirms both heap-growth paths
in `mspace_memalign` call the wrapper and that the wrappers call the real
allocator without recursion. No QSPI erase or programming was needed.

The clean profiling build occupies 908,440 B / 1 MiB internal Flash (86.64%),
1,376 B more than the baseline. Internal SRAM remains 141,888 B (43.30%).
The subsequent quad-data and real-photo experiments are recorded below.

## Compare single-line and quad-data QSPI

The command defaults to the original one-line data mode. To compare modes in
one firmware image, alternate these commands three times after a reset:

```text
ncnn_mobilenet_smoke single
ncnn_mobilenet_smoke quad
```

Both use a 27 MHz QSPI clock (216 MHz HCLK, prescaler 7), the same model,
96x96 input, profiling and all 1000 reference comparisons. `single` uses 03h;
`quad` uses Winbond's 6Bh quad-output read: one-line instruction and 24-bit
address, eight dummy clocks, then four data lines. See section 8.2.9 of the
[Winbond W25Q128FV datasheet](https://files.waveshare.com/upload/4/46/W25Q128fv.pdf)
and the [W25Q128JV datasheet](https://www.mouser.com/datasheet/2/949/Winbond_Electronics_Corporation_09_06_2024_W25Q128-3501276.pdf).
This is quad-output SPI, not four-line QPI instruction mode or 6Bh at a higher
clock. The existing convolution-QSPI command still selects single-line mode.

Quad mode is restricted to the tested Winbond JEDEC ID `ef4018` and requires
an existing QE=1 in SR2. The tested board already has SR1=00 / SR2=02. The
helper only reads these registers (05h/35h); it never issues Write Enable,
status writes, program or erase commands. If QE is clear, quad returns an
explicit error; single mode remains available. Micron `20ba18` remains allowed
only in single mode. No model image conversion or QSPI reprogramming is needed.

Before configuring either mode, the helper invalidates D-cache entries for
the entire aligned 16 MiB read-only QSPI window. This prevents previously
cached contents from masking an incorrect bus mode and gives both modes the
same cache preparation. Cache maintenance and diagnostic output are included
in the `qspi` timing field. No ELF code or mutable data occupies that window.

The mode diagnostic prints actual selected instruction, data lines and dummy
cycles. All existing CRC, bounded-loader, SDRAM placement, reference and
memory-release checks still run. A failed hardware comparison is a failure,
not a silent fallback from quad to single.

### Same-firmware comparison verified on 2026-09-17

Three alternating single/quad pairs produced these times in milliseconds:

| Stage | Single runs | Quad runs | Median speedup |
| --- | --- | --- | ---: |
| CRC | 3010 / 3010 / 3010 | 752 / 753 / 752 | 4.00x |
| Inference | 8187 / 8191 / 8190 | 3242 / 3240 / 3245 | 2.53x |
| Total through heap restoration | 11409 / 11414 / 11413 | 4205 / 4203 / 4207 | 2.71x |

QSPI setup including cache preparation was 27 ms for both modes. Graph loading
was 91 ms single and 89 ms quad; weights/pipeline 2 ms, input 4 ms and cleanup
29 ms in both. The larger setup time versus the previous firmware includes the
new full-window cache invalidation and status/mode diagnostics. Use the paired
results above for the speedup comparison.

All six runs passed all 1000 logits with the same reported maximum error
0.000017 and peak allocated SDRAM pages of 1,256,256 bytes. Baseline/final
usage remained zero; there were no failed external or observed other-heap page
allocations. Top-five indexes and reported logits were identical. SR1/SR2
remained 00/02 throughout. The default command was also verified as single mode,
and QSPI convolution, dense inference and 768 KiB allocation passed afterward.
A separate direct quad run immediately after reset also passed (3243 ms inference,
4206 ms total), as did rejection of an invalid argument and subsequent commands.

Clean build and verified internal Flash programming passed. The image occupies
908,984 B (86.69%), 544 B more than the profiling firmware; internal SRAM remains
141,888 B. A host HAL mock verified the read-only command sequence and rejection
of QE=0, Micron quad, unknown IDs, busy status, invalid line counts and HAL
receive/mapping errors. On actual hardware, only Winbond ef4018 with QE=1 was
exercised; unsupported-chip/error cases were host tests.

These speedups apply to this synthetic 96x96 FP32 fixture at 27 MHz. Quad is an
explicit opt-in; the following photo mode uses the same model and checks.


## Real JPEG input

The fixed fixture is the existing [cat2.jpg](../data_samples/photos/cat2.jpg):
295x231 pixels, 25,243 bytes, SHA-256
`a5e283094ee97c0acbff965fc4def7e896f5c1e7ce33ae3c1629aa7d20b49f75`.
It is embedded in internal Flash; the model image in QSPI is unchanged.
With the model already provisioned, flash the updated internal firmware and run:

```text
ncnn_mobilenet_smoke quad photo
```

Select a read mode explicitly for `photo`; `single photo` also works but is
slower. Commands without `photo` retain the synthetic 96x96 reference.
Photo inference remains manual. The STM32 template now also displays the
photo and result on LCD; see the [LCD guide](../ncnn_lcd_smoke/README.md) for
the updated memory map, cache handling and timings. No camera is included.

The MCU decodes JPEG into RGB using a private JPEG-only, scalar `stb_image`
implementation, then NCNN bilinearly resizes it to 224x224, converts to planar
FP32 and multiplies by 1/255. This matches the existing project classifier:
square resize, no center crop, and mean/std normalization already in the graph.
The native reference uses the same decoding/preprocessing source. JPEG bytes,
decoded RGB and the entire FP32 input tensor must pass CRC32 checks before
inference, in addition to the model CRCs and all 1000 logit comparisons.
RGB, input and output placement are checked against the SDRAM range.
The RGB buffer is freed before inference; its transient allocation still
contributes to the measured peak. Only this compiled, checked fixture is
supported, not arbitrary JPEG dimensions or externally supplied files.

The QSPI header's input-size/reference fields still describe the original
96x96 synthetic fixture and retain their exact-match validation. The same
graph accepts 224x224 input; photo mode has a separate compiled reference and
preprocessing CRCs. No QSPI erase/program operation is needed for this stage.

### Reproduce the photo reference

First complete the native NCNN build and model export described above. Then,
from the Embox root:

```sh
c++ -std=c++11 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  -I/tmp/mobilenet-host-src/ncnn-20250916/src -I/tmp/mobilenet-host-build/src \
  project/ncnn/tools/prepare_mobilenet_photo.cpp \
  project/ncnn/ncnn_mobilenet_smoke/photo_input.cpp \
  /tmp/mobilenet-host-build/src/libncnn.a -o /tmp/prepare_mobilenet_photo
python3 project/ncnn/tools/export_mobilenet_photo.py \
  --runner /tmp/prepare_mobilenet_photo --model-dir /tmp/mobilenet-export \
  --output-dir /tmp/mobilenet-photo
cmp /tmp/mobilenet-photo/photo_jpeg.h project/ncnn/ncnn_mobilenet_smoke/photo_jpeg.h
cmp /tmp/mobilenet-photo/photo_reference.h project/ncnn/ncnn_mobilenet_smoke/photo_reference.h
```

The exporter pins the JPEG and all three model-file hashes, requires all 1000
original/FP32-storage outputs to match exactly, and writes the generated
headers, decoded RGB, input/output binaries and `photo-manifest.json`.
The manifest records byte sizes, SHA-256 and CRC32 for each stage. Reference
generation and header reproduction passed on the native scalar backend.
Host ASan/UBSan checks covered valid decoding/preprocessing and rejection of
null, truncated and wrong-dimension input. The fixed dimensions are checked
before full decoding and capped at 512 by the private decoder configuration.

### Photo verified on 2026-09-17

Three consecutive quad-mode runs on STM32F746G-DISCO passed RGB/input CRCs
and all 1000 logits, with maximum reported absolute error 0.000028:

| Measurement | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Input preparation and validation, ms | 146 | 146 | 146 |
| Inference, ms | 14813 | 14817 | 14815 |
| Total through heap restoration, ms | 15927 | 15931 | 15929 |
| Peak allocated SDRAM pages, bytes | 6067008 | 6067008 | 6067008 |
| Baseline / final allocated bytes | 0 / 0 | 0 / 0 | 0 / 0 |

Minimum free page capacity was 2,305,152 bytes (about 2.20 MiB); no external
allocation failures or other-heap page allocations were observed. The top
five outputs were identical across runs:

| Class index | ImageNet label | Raw logit |
| ---: | --- | ---: |
| 285 | Egyptian cat | 9.297002 |
| 750 | quilt | 8.008289 |
| 282 | tiger cat | 7.288603 |
| 281 | tabby | 6.480601 |
| 831 | studio couch | 6.445649 |

These are raw logits, not probabilities or a verified cat-breed diagnosis.
One photo does not establish dataset accuracy. In a native comparison, resizing
this same photo to 96x96 instead produced top-1 `quilt`; 224x224 matches the
existing project classifier's input size and produced the cat category above.
Do not generalize the earlier synthetic 96x96 timing to this larger input.

The clean firmware occupies 973,880 B / 1 MiB internal Flash (92.88%), with
74,696 B remaining. Internal SRAM stays at 141,888 B (43.30%). The JPEG,
decoder, new reference, labels and command code add 64,896 B versus the quad
comparison firmware. Four startup tests, both synthetic read modes, and the
subsequent convolution-QSPI, dense and allocation tests also passed. A separate
`single photo` run passed the same checks (32,799 ms inference, 36,173 ms total,
same memory peak), as did rejection of a missing read mode or unknown input
argument. Display and camera buffers must be budgeted separately before
enabling them.


## LCD integration

`photo` now draws the already decoded RGB photo, then publishes the top label
and inference time after all output and heap checks pass. The 256 KiB display
reservation is outside the external heap, which starts at `0x60040000` and
ends at `0x60800000`. The earlier photo-only measurements above remain the
historical baseline; use the [LCD guide](../ncnn_lcd_smoke/README.md) for current
firmware sizes, 15.349 s median inference, combined SDRAM budget and checks.
Display rendering does not change the input or reference tensors.
