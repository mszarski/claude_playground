# Holoscan acquisition notes: ultrasound reference + Sensor Bridge

Source-reading notes for reusing NVIDIA's ultrasound acquisition path in the GPU sonar suite
(FLS / SAS / flank array). Nothing here was built or run: the reading host is x86_64 with no GPU,
no Holoscan SDK and a restricted network. Every claim cites `file:line` in one of the trees below.

## 0. Sources and conventions

| Tag | Repo | Revision read | Licence |
| --- | --- | --- | --- |
| `US:` | `nvidia-holoscan/holoscan-ultrasound-imaging` at `/home/user/nvidia-holoscan/holoscan-ultrasound-imaging` | **`24d405e0f2cd3e216b797baff4c9857d87e5f10d`**, commit date 2026-10-22T08:00:47-04:00, "fix-pgx-link (#7)". The pinned commit exists; all `US:` notes are about it. Package version 0.1.0 (`US:pyproject.toml:7`). | Apache-2.0 (`US:LICENSE`, `US:pyproject.toml:12`). Exceptions: `US:fpga/build/ip` is under an Altera licence, "used solely for design and implementation on an Altera product" (`US:fpga/build/LICENSE:9`, `US:README.md:93-95`); MimickNet weights are CC BY-NC 4.0 and not in the repo (`US:README.md:98-101`). |
| `HSB:` | `nvidia-holoscan/holoscan-sensor-bridge` at `/home/user/nvidia-holoscan/holoscan-sensor-bridge` (shallow clone) | **HEAD `41cecaa81ab906a55e7ec22e5e6989f7e602a4c8`**, 2026-11-09T14:54:45-08:00, "Fix incorrect SIOCSARP permission error handling (#137)". No tag at HEAD in the shallow clone; `HSB:VERSION` = `3.0.0`. | Apache-2.0 (`HSB:LICENSE`, `HSB:README.md:103`); third-party parts listed at `HSB:README.md:113-119`. |
| `HSB@2.7.0:` | same repo, tag `2.7.0` = `1df133d4c78a6921249be4b147d65554aa8efa68` (2026-07-16), fetched as an extra shallow tag and read with `git show 2.7.0:<path>`; the working tree stays at HEAD. | This is the exact SDK commit the ultrasound repo pins (`US:docker/README.md:6-8`). | Apache-2.0 |

Not read: anything outside these two repos (NVIDIA online docs, TI datasheets for TX73L64 / AFE5932,
the FPGA `.rpd` release asset, the vendored HSB FPGA core, which is intentionally absent at
`US:fpga/rtl/nv_hsb_ip/README.md:3-5`). LFS objects were not fetched.

## 1. Discrepancies with the brief (read this first)

1. **The reference imaging app is not a Holoscan operator graph.** Live PW4cm acquisition runs in a
   native C++ loop (`PolicyExecutor` -> `Run` -> `HsbBackend` -> `HololinkTransport`) that uses the
   non-operator class `hololink::operators::RoceReceiver` directly
   (`US:cpp/src/hololink_transport.cpp:15,64,483-487`); Python does setup, reconstruction and
   display (`US:examples/scanner_cpp_live.py:4-8`). A Holoscan `Application` with `RoceReceiverOp`
   exists only in the receive-only test-pattern capture (`US:python/nvus/capture.py:121-128`) and
   two commissioning tools (`US:python/tools/nvus_lvds_validation.py:434-481`,
   `US:python/tools/nvus_tx_update_packet.py:352-391`).
2. **Version split.** The reference pins HSB **2.7.0** + Holoscan **4.4.0** and needs six local
   patches to the SDK (`US:docker/README.md:6-16,29-31`). HSB HEAD is **3.0.0** on Holoscan SDK
   **5.0** / CUDA 13, which "is not backward compatible with Holoscan SDK 4.x"
   (`HSB:docs/user_guide/introduction.mdx:34-45`, `HSB:docs/user_guide/setup.mdx:3`). Operator
   construction and parameters differ between the two (section 7.2). The native header the
   reference includes, `hololink/operators/roce_transmitter/roce_transmitter.hpp`
   (`US:cpp/src/hololink_transport.cpp:16`), is created by patch `00-roce-transmitter-api.patch`
   and is not in the HEAD directory listing of `HSB:src/hololink/operators/roce_transmitter/`.
3. **The "GPU receive ring of 1024 frames" is the reference's own setting, not an HSB default.**
   It is `HololinkTransportConfig::rx_pages = 1024` (`US:cpp/include/nvus/hololink_transport.hpp:17`),
   range-checked 2..1024 (`US:cpp/src/hololink_transport.cpp:254-255`). Stock `RoceReceiverOp`
   defaults are `pages = 2`, `queue_size = 1` (`HSB@2.7.0:src/hololink/operators/roce_receiver/roce_receiver_op.cpp:70-72`;
   HEAD `HSB:src/hololink/operators/roce_receiver/roce_receiver_op.hpp:55-56`). It is configurable
   without FPGA changes (section 3.4).
4. **"Odd/even" is a name, not an element interleave.** Per the code, "odd" routes elements 0-63
   and "even" routes 64-127, contiguous blocks (`US:python/nvus/raw.py:38-48`).
5. **The mux is the TX73L64's own T/R switch select (register 0x1A), not a separate HV mux chip**
   (`US:python/nvus/tx.py:165,177,732-734`). No other mux device appears in the code.
6. **Gain ramp claim is partly wrong.** 12 dB is the maximum start attenuation and ~122.4 us is
   the longest one-segment ramp the encoder accepts at 12 dB; neither is "as coded". The shipped
   CLI uses 12 dB, start 2 us, ramp 10 us (`US:python/nvus/run.py:41-42`). It is an attenuation
   that ramps down to 0 dB, inside the TX73L64. **No AFE5932 gain/attenuator/TGC register is
   written or named anywhere: "0-24 dB attenuator" is not found** (section 6).
7. **150k samples per ping is above two limits baked in host code** (65536 samples in the C++
   sequence compiler, 65536 records = 32768 samples in the capture CLI); neither is an FPGA limit
   (section 5.4).
8. **No per-ping metadata from the FPGA reaches the application.** Event identity is
   "host_inferred" and timestamps are host `steady_clock` taken before `acquire`
   (`US:cpp/include/nvus/runtime.hpp:91-95`, `US:cpp/src/runtime.cpp:263,275-280`,
   `US:python/nvus/scanner.py:17-18`). HSB's 128-byte frame metadata does carry PTP timestamps and a
   frame counter, but the reference keeps only frame number, bytes written, PSN and drop count
   (`US:cpp/include/nvus/hsb_backend.hpp:8-13`).
9. **Jetson AGX Orin and AGX Thor cannot run the RoCE receiver path** (section 7.3). The reference
   itself lists only ThinkStation PGX or DGX Spark as hosts (`US:docs/hardware.md:23`) and says the
   image "targets PGX/GB10, not IGX" (`US:docker/README.md:32`).
10. **Transmit waveforms are ultrasound-only**: 5, 6.25 or 8.333 MHz, 1/2/6 cycles
    (`US:python/nvus/tx.py:208-213`); the native path accepts only 5 MHz x 2 cycles
    (`US:python/nvus/native_observation.py:44-45`). Probe geometry (128 elements, 0.3 mm pitch,
    1480 m/s) is hard-coded (`US:python/nvus/tx.py:225-227`, `US:cpp/src/sequence.cpp:66`).
11. Minor: docs name the probe "Sonosray L12-5 D" (`US:README.md:18`); code comments and the viewer
    say "L7-3" (`US:cpp/include/nvus/sequence.hpp:18`, `US:examples/scanner_cpp_live.py:368`).
12. Minor: HSB's data-plane doc labels the metadata fields little-endian
    (`HSB:docs/user_guide/dataplane.mdx:319-327`), but host code parses them big-endian at both
    revisions (`HSB@2.7.0:src/hololink/core/hololink.cpp:3144-3153`,
    `HSB:src/hololink/core/frame_metadata_parser.hpp:37-46`).

Confirmed as briefed: 64 receive channels, 12-bit samples, 50 MSPS, 128-element aperture built from
two 64-channel halves.

## 2. Operators and pipeline of the reference

### 2.1 Live imaging path (`./nvus run pw4cm`), no Holoscan graph

`US:python/nvus/run.py:67-72` calls `nvus._pw4cm.main`, which is `examples/scanner_cpp_live.py`
installed under that name (`US:CMakeLists.txt:81`).

| Stage | Class / function | Lang | File | In -> out |
| --- | --- | --- | --- | --- |
| Idle bring-up | `_HsbSession.open(prepared)` -> `HololinkTransportConfig` | Py | `US:python/nvus/scanner_hardware.py:182-311` | Enumerate, `control.reset()`, `Evm.init(...)`, program attenuation/DTGC + LPF, snapshot registers. Never fires. |
| Policy | `Pw4cmPolicy(angles_deg, label="pw4cm")`, `Policy::next(count, PolicyObservation)` | C++ | `US:cpp/include/nvus/policy.hpp:29-45` | -> `std::vector<TransmitEvent>` |
| Compiler | `compile_plane_waves(angles_deg, PlaneWaveConfig)` -> `PlaneWaveAction` | C++ | `US:cpp/src/sequence.cpp:98-147` | angles -> `PreparedSequence` (2 events per angle, each a 2176-byte TX update packet) |
| Executor | `PolicyExecutor(hardware, policy, rollout_length, PolicyLimits)`; `start/stop/wait/next/status` | C++ | `US:cpp/include/nvus/policy.hpp:76-86` | -> `ObservationGroup` |
| Run loop | `Run`, `Scanner::run(PreparedSequence, RunOptions)` | C++ | `US:cpp/include/nvus/runtime.hpp:114-161`, `US:cpp/src/runtime.cpp:260-299` | per event: `Hardware::acquire` -> `Buffer` + `EventRecord` |
| Backend | `HsbBackend::acquire(Event, stop, deadline)` | C++ | `US:cpp/src/hsb_backend.cpp:185-308` | send update packet, flush, (guard, trigger), receive, validate |
| Transport | `HololinkTransport` (implements `HsbTransport`), factory `make_hololink_hardware(HololinkTransportConfig, HsbExecutionConfig)` | C++ | `US:cpp/src/hololink_transport.cpp:56-662` | owns `hololink::DataChannel`, `RoceReceiver`, `RoceTransmitter` |
| Owned copy | `copy_cuda_buffer(source, bytes, device, stream)` | C++ | `US:cpp/src/cuda_buffer.cpp:111-115,146` | ring page -> freshly `cuMemAlloc`'d device buffer via `cuMemcpyDtoDAsync` |
| Python view | `observation_view(frame, board)` -> `Observation`; `Observation.gpu_rf()` | Py | `US:python/nvus/native_observation.py:51-52`, `US:python/nvus/scanner.py:45-89` | CuPy `uint8` 1-D views over the owned CUDA buffers |
| Recon | `Processor.process(observation)` -> `Image` | Py/CUDA | `US:python/nvus/scanner_processing.py:225-259` | |
| Display | `Display.show(db, scanner)` | Py | `US:python/nvus/scanner_display.py`, call at `US:examples/scanner_cpp_live.py:378` | |

Python bindings (nanobind module `nvus._scanner`) expose all of the above
(`US:cpp/src/bindings.cpp:20-279`); `Event.packet`, `PreparedSequence.events/resources/...` are
read-write from Python (`US:cpp/src/bindings.cpp:56-73`).

Data handed to the consumer per frame group: 64 buffers (32 angles x 2 mux halves), each a 1-D
`uint8` array of `samples * 96` bytes (`US:python/nvus/scanner.py:34`,
`US:python/nvus/scanner_processing.py:243-246`). There are no named tensors on this path.

### 2.2 The only Holoscan graphs in the repo

| Graph | Operators | Ports / tensor | File |
| --- | --- | --- | --- |
| Test-pattern capture | `hololink.operators.RoceReceiverOp` (HSB, C++) -> `Sink(hs.core.Operator)` (Py) | flow `("output","input")`; sink reads `op_input.receive("input")[""]` (unnamed tensor, uint8, `frame_bytes` long) and `self.metadata[...]` keys `frame_number, received_frame_number, dropped, bytes_written, page_number` | `US:python/nvus/capture.py:101-128` |
| LVDS validation tool | `RoceReceiverOp` -> `PatternValidatorOp` | `("output","input")` | `US:python/tools/nvus_lvds_validation.py:33,449-481` |
| TX update packet tool | `TxUpdatePackageSourceOp` -> transmitter | `("output","input")` | `US:python/tools/nvus_tx_update_packet.py:325,391` |

`RoceReceiverOp` kwargs as used at HSB 2.7.0: `frame_size, frame_context, ibv_name, ibv_port=1,
hololink_channel, device, trim=True, pages, queue_size` with `CountCondition` and a
`GreedyScheduler` (`US:python/nvus/capture.py:123-133`). The capture CLI uses `pages = 4096`,
`queue_size = 4` (`US:python/nvus/capture.py:19,159`).

### 2.3 Where beamforming happens

On the GPU, in Python-launched CUDA kernels, off the per-shot receive path
(`US:AGENTS.md:84-85` states the convention; the code matches):

- Decode + demodulation: `decode_fir31_frame_kernel` in `US:python/nvus/kernels/decode.cu:30-40`,
  loaded by `Beamformer` (`US:python/nvus/live_recon.py:283,351`), called as
  `set_pages` / `baseband_frame` (`US:python/nvus/live_recon.py:400,414`).
- Delay-and-sum: `das_kernel` in `US:python/nvus/kernels/das.cu:23`, via
  `Beamformer.compound_image` (`US:python/nvus/live_recon.py:431`); offline path `recon.beamform`
  with `RawKernel(DAS_KERNEL, "das_pw0")` (`US:python/nvus/recon.py:141,238`).
- Intermediate: IQ per event, `complex64`, shape `(plan.n_keep, 64)`
  (`US:python/nvus/scanner_processing.py:189`); stacked channel data `(n_tx, 128, n_keep)`
  `complex64` (`US:python/nvus/scanner_processing.py:254`); image `(528, 519)` for PW4cm
  (`US:examples/scanner_cpp_live.py:361`).

## 3. Frame and packet layout

### 3.1 Sample packing (payload, identical on the wire and in GPU memory)

| Item | Value | Source |
| --- | --- | --- |
| Record | `RECORD_BYTES = 48`; one 24-byte word per AFE (`AFE_SAMPLE_BYTES = 24`, `AFES_PER_RECORD = 2`) | `US:python/nvus/raw.py:14-16` |
| Lanes | `LANES_PER_AFE = 16`, 12 bit each, **LSB-first** in a little-endian 192-bit word | `US:python/nvus/raw.py:5-6,17,34-35` |
| Channels | `CHANNELS_PER_AFE = 32`; "Physical channel c rides lane c >> 1 with record parity c & 1" -> **2 records per sample**, 64 channels | `US:python/nvus/raw.py:6,18`; `samples_to_records = 2 * samples` `US:python/nvus/units.py:24-25` |
| Bytes per sample | 96 (all 64 channels x 12 bit) | `US:cpp/src/hsb_backend.cpp:134,137` |
| Sample format | 12-bit two's complement (`SAMPLE_MASK = 0xFFF`, sign-extend at 0x800) | `US:python/nvus/raw.py:19,28-30` |
| Sample rate | `FS = 50e6`; record rate is therefore 100 MHz | `US:python/nvus/raw.py:22`, `US:python/nvus/fpga.py:45-48` |
| Frame alignment | `FRAME_ALIGNMENT_BYTES = 192` (4 records) | `US:python/nvus/raw.py:20`, `US:python/nvus/units.py:28-31` |
| GPU decode to `(n_samples, 64)` float32 | `recon.unpack` (reshape `(-1, 2, 2, 2, 16)`, transpose, zero first 8 samples "start-of-record capture spike") | `US:python/nvus/recon.py:166-186` |
| Column -> element | `element = half_offset + afe * 32 + physical_channel`, `half_offset` 0 ("odd") or 64 ("even") | `US:python/nvus/raw.py:38-48` |

Raw rate while capturing: 96 B x 50e6 = 4.8 GB/s = 38.4 Gbit/s (derived). Link is 100G QSFP28
(`US:docs/hardware.md:29`).

Reference frame: 2974 samples (`US:docs/cli.md:24`; computed at `US:cpp/src/sequence.cpp:106-110`)
= 285,504 bytes per ping.

There is **no application header inside a frame**: the frame is `length_records` raw records, then
TLAST (`US:fpga/rtl/capture/capture_ctrl.sv:163-167`).

### 3.2 FPGA capture window (read-only understanding)

- `capture_ctrl`: `IN_DWIDTH = 384` (one 48-byte record), `OUT_DWIDTH = 512`, `BUF_DEPTH = 1024`,
  instantiated with `NO_BACKPRESSURE (1)` (`US:fpga/rtl/capture/capture_ctrl.sv:19-21,330`,
  `US:fpga/rtl/top/FPGA_top.sv:667-670`).
- Registers at `CAPTURE_CTRL = 0x5000_0000`: `CMD 0x00` (START 0x1, ABORT 0x2, CLEAR 0x4),
  `CFG 0x04` (ENABLE 0x1, TLAST_ENABLE 0x2), `DELAY_RECORDS 0x08`, `LENGTH_RECORDS 0x0C`,
  `STATUS 0x80`, `ACCEPTED 0x84`, `COUNT 0x88`, `LAST_CAPTURE 0x8C`
  (`US:python/nvus/fpga.py:15,51-64`; RTL `US:fpga/rtl/capture/capture_ctrl.sv:56-64`).
- **`LENGTH_RECORDS` is a full 32-bit register with no clamp in RTL**
  (`US:fpga/rtl/capture/capture_ctrl.sv:78,150-151,163`). Status bit 8 is a sticky FIFO overflow
  flag (`US:fpga/rtl/capture/capture_ctrl.sv:178,269-271`).
- Capture start is phase-aligned to the 50 MHz sample (`ST_WAIT_PHASE`,
  `US:fpga/rtl/capture/capture_ctrl.sv:164,289-299`; rationale `US:python/nvus/fpga.py:45-49`).
- HSB core build settings: `DATAPATH_WIDTH 512`, `SENSOR_RX_IF_INST 1`, packetizer disabled
  (`SIF_RX_PACKETIZER_EN = '{default:0}`), `HOST_IF_INST 2`, **`HOST_MTU 4096`**
  (`US:fpga/rtl/top/HOLOLINK_def.svh:64,73,88,113-114`).

### 3.3 On the wire (HSB RoCE v2)

- Data packets: Ethernet(14) + IPv4(20) + UDP(8) + BTH(12, opcode `0x2A`, P_Key `0xFFFF`,
  dest QP, 24-bit PSN incrementing per packet) + RETH(16: 64-bit virtual address, R_Key, DMA
  length) + sensor data + iCRC(4) (`HSB:docs/user_guide/dataplane.mdx:265-281`). Host constant
  `roce_overhead = 74` (`HSB@2.7.0:src/hololink/core/data_channel.cpp:265`).
- End of frame: one metadata packet, opcode `0x2B` (RDMA write with immediate), carrying 128
  bytes: flags(4), PSN(4), CRC of the window(4), PTP of first sample(12), valid bytes(8), frame
  number(4, 16-bit counter), PTP of metadata formation(12), reserved(80)
  (`HSB:docs/user_guide/dataplane.mdx:305-329,333-335`).
- Host parse order (big-endian): `flags u32, psn u32, crc u32, timestamp_s u64, timestamp_ns u32,
  bytes_written u64, ignored u16, frame_number u16, metadata_s u64, metadata_ns u32`
  (`HSB@2.7.0:src/hololink/core/hololink.cpp:3144-3153`); struct at
  `HSB@2.7.0:src/hololink/core/hololink.hpp:1097-1109`; `METADATA_SIZE = 128`
  (`HSB@2.7.0:src/hololink/core/hololink.hpp:158`).
- Immediate data: page index = `imm_data & 0xFFF`, PSN = `(imm_data >> 12) & 0xFFFFF`
  (`HSB@2.7.0:src/hololink/operators/roce_receiver/roce_receiver.cpp:945,950`).
- Queue pair: `IBV_QPT_UC`, `path_mtu = IBV_MTU_4096`, `rq_psn = 0`
  (`HSB@2.7.0:.../roce_receiver.cpp:365,424-425`); UDP port 4791 is passed by the reference
  (`US:cpp/src/hololink_transport.cpp:501-502`). The reference also opens a transmit QP,
  `tx_qp = 2` (`US:cpp/include/nvus/hololink_transport.hpp:17`,
  `US:cpp/src/hololink_transport.cpp:503-505`).
- Payload per packet: `((mtu - overhead) / 128) * 128` with `DEFAULT_MTU = 1500` unless the
  enumeration metadata carries `"mtu"` (`HSB@2.7.0:src/hololink/core/data_channel.cpp:131-145,151-156`,
  `HSB@2.7.0:src/hololink/core/networking.hpp:40,43`). No `mtu` override appears in the ultrasound
  repo, so the expected payload is 1408 bytes (derived; not confirmed on hardware).
- Stock 2.7.0 receive queue: CQ size 100, `max_recv_wr = 2048`, 256 posted receives
  (`HSB@2.7.0:.../roce_receiver.cpp:294,361,392`). Patch 30 raises all three to 4096 and stops
  the receiver replacing a queued frame before the consumer attaches
  (`US:docker/patches/30-roce-receive-queue.patch:10,19,28,36-58`).

### 3.4 In GPU memory: the receive ring

```
metadata_offset = round_up(frame_bytes, 128)
page_size       = metadata_offset + 128
allocation      = round_up(page_size * rx_pages, 4096)
```
(`US:cpp/src/hololink_transport.cpp:461-465`). Each page = raw frame, then the 128-byte metadata
block. The FPGA is told the ring through `configure_roce(frame_memory, frame_size, page_size,
pages, local_data_port)`, which writes `DP_PAGE_INC` and `DP_MAX_BUFF = pages - 1`
(`HSB@2.7.0:src/hololink/core/data_channel.cpp:227,272-274`).

| Question | Answer | Source |
| --- | --- | --- |
| Where is 1024 set? | Default member `rx_pages = 1024`; `_HsbSession(..., rx_pages=1024, ...)`; `native_hsb_config(..., rx_pages=1024)`; **hard-coded positional `1024` in the example** | `US:cpp/include/nvus/hololink_transport.hpp:17`; `US:python/nvus/scanner_hardware.py:20,189-190`; `US:examples/scanner_cpp_live.py:255-256` |
| Allowed range | 2..1024 in both languages | `US:cpp/src/hololink_transport.cpp:254-255`; `US:python/nvus/scanner_hardware.py:196` |
| HSB's own limit | 1..4096 pages (12-bit page index) | `HSB@2.7.0:src/hololink/core/data_channel.cpp:249-254`; `US:python/nvus/capture.py:17-19` |
| Configurable without FPGA change? | **Yes.** Python: pass `rx_pages` to `_HsbSession` or set `HololinkTransportConfig.rx_pages` (bound read-write, `US:cpp/src/bindings.cpp:266`). C++: set the struct field. The CLI exposes no flag; the example must be edited. | as cited |
| Queue depth | The reference passes `rx_pages` as both `pages` and `queue_size` to `RoceReceiver` | `US:cpp/src/hololink_transport.cpp:483-486` |
| Is the ring handed to the consumer? | No. Each frame is copied to an owned CUDA buffer before `acquire` returns; "never a borrowed cyclic DMA page" | `US:cpp/src/hololink_transport.cpp:565-575`; `US:cpp/include/nvus/hsb_backend.hpp:15-16` |
| Host-memory fallback | If the descriptor has a host pointer the ring is pinned host memory; otherwise GPUDirect with an optional `cuFlushGPUDirectRDMAWrites` fence | `US:cpp/src/hololink_transport.cpp:466-479,553-561` |

Caveat: "HSB retains its 12-bit page index across runs, even when the ring shrinks"
(`US:python/nvus/capture.py:17-18`). The native path resets the board on open
(`US:python/nvus/scanner_hardware.py:241`), so a smaller ring there starts clean; untested here.

Ring sizes (derived from the formula): PW4cm 285,696 B/page x 1024 = 292.6 MB. A 150,000-sample
ping is 14,400,000 B (+128): x1024 = 14.75 GB, x64 = 921.6 MB, x32 = 460.8 MB.

### 3.5 Per-frame integrity checks the reference applies

`bytes_written == frame_bytes`, `dropped == 0`, 16-bit metadata frame number equals the low 16 bits
of the extended counter, metadata PSN equals completion PSN (20 bits), frame numbers consecutive
(`US:cpp/src/hsb_backend.cpp:275-296`).

## 4. The 64-of-128 receive mux

| Aspect | Finding | Source |
| --- | --- | --- |
| Device | TX73L64 pair (`tx1`, `tx2`); T/R switch select register `TX_TRSW_SEL_REG = 0x1A`, written `broadcast` to both chips | `US:python/nvus/tx.py:12,165,732-734` |
| Values | `TX_PROBE_RX_SELECT_VALUES = {"odd": 0x00000000, "even": 0xFFFFFFFF}` | `US:python/nvus/tx.py:177` |
| Related setup | `configure_probe(rx_select, trsw_delay_cycles, global_mode)`: regs 0x16-0x19 = 0, 0x1A = select, 0x1B = `TX_TRSW_EN_AUTO_DIS_RX (0x1000)` [`| TX_TRSW_GBL_MODE (0x1)`], 0x2D/0x2E = 0, memory words 0x80/0x81 = TRSW delay | `US:python/nvus/tx.py:161-175,736-758` |
| Per ping or static | **Per ping.** Native path: operation index 1 of every event packet writes 0x1A (`US:cpp/src/sequence.cpp:87,141`), enforced by `validate_packet` (`US:cpp/src/hsb_backend.cpp:46-48`). Python path: `Evm.fire(rx_select=...)` calls `set_trsw_select` then sleeps `trsw_settle_s` (`MUX_SETTLE_S = 0.002`) | `US:python/nvus/evm.py:224-229`, `US:python/nvus/tx.py:179-181` |
| Full aperture | Two physical acquisitions per logical transmit, same delay profile, "mux-major" order: events `0..N-1` are side 0 ("odd"), `N..2N-1` side 1 ("even") | `US:cpp/src/sequence.cpp:130-143`; `US:python/nvus/native_observation.py:33-37`; `US:docs/cli.md:22-23` |
| Assembly | Reconstruction concatenates the two halves per geometry into 128 channels and requires each geometry to have both halves exactly once | `US:python/nvus/scanner_processing.py:140-144,194-202` |
| Transmit side | Always the full 128-element aperture; chip `t` channel `c` drives element `64*(c % 2) + 32*t + c//2` | `US:python/nvus/tx.py:229-233,263-276` |

Sonar consequence: a moving platform sees the two halves at different ping times. One ping gives
only 64 contiguous elements.

## 5. Programming scan sequences

There are two APIs. They share the bring-up code but differ in limits.

### 5.1 Python driver: `nvus.Evm` (slow SPI, one shot at a time)

- `EvmConfig(**overrides)`; unknown keys raise (`US:python/nvus/evm.py:51-59`). Defaults at
  `US:python/nvus/evm.py:19-46`, e.g. `sync_clk_hz=100e6`, `wakeup_s=200e-6`, `tx_delay_s=200.1e-6`,
  `tx_pulse_s=0.4e-6`, `interval_margin_s=100e-6`, `c=1480.0`, `fs=50e6`.
- `Evm.init(pulse=None, delays=None, *, depth=None, time=None, samples=None, pri=None,
  rx_select="odd", repeat_shots=1, retries=None)` (`US:python/nvus/evm.py:97-98`). Exactly one of
  depth/time/samples (`US:python/nvus/units.py:14-21`); depth is multiplied by 1.1
  (`US:python/nvus/evm.py:103-104`). `pulse` is a key of `TX_SAMPLE_PATTERN_START_WORDS`
  (`US:python/nvus/tx.py:62-68`); `delays` is 128 per-element values in BF half-cycles, loaded at
  `delay_profile_word` (default 510) (`US:python/nvus/evm.py:175-176`, `US:python/nvus/tx.py:183`).
- `Evm.arm(shots=1)`, `Evm.fire(rx_select=None)`, `Evm.rx_device(mode, arm_delay_s=None)`
  (`US:python/nvus/evm.py:218-249`).
- Timing goes to the FPGA sync sequencer via `Fpga.program_sync_capture(wakeup, tx_delay,
  tx_pulse, capture_delay, count, interval, ...)` (`US:python/nvus/fpga.py:223-264`); capture window
  via `Fpga.program_capture(length_records, delay_records, wait_s)`, limited only to 32 bits
  (`US:python/nvus/fpga.py:165-167`).
- TX helpers: `Tx.load_pattern`, `select_pattern`, `load_delay_profiles(profiles, first_word)`,
  `select_profile`, `set_auto_increment`, `verify_sample_pattern(..., burst_cycles, frequency_hz,
  invert)`, `setup_plane_wave_sweep(tx_device, angle_count, angle_max_deg, c)`
  (`US:python/nvus/tx.py:381-405,618-730,783-797`).
- Geometry helpers: `plane_wave_delays(angle_rad, c, bf_clock_hz)` (`US:python/nvus/tx.py:315`),
  `focus_profile(x, z, c)` and `linear_foci(lines, width, depth, c)` (`US:python/nvus/focus.py:10,31`).

### 5.2 Native streamed API: C++ with Python bindings

- `struct PlaneWaveConfig { double depth_mm = 40., sound_speed_m_s = 1480., interval_us = 400.; }`
  (`US:cpp/include/nvus/sequence.hpp:7-9`).
- `PreparedSequence prepare_plane_waves(const std::vector<double>& angles_deg, PlaneWaveConfig = {})`
  and `PlaneWaveAction compile_plane_waves(const std::vector<double>& angles_deg, PlaneWaveConfig = {})`
  (`US:cpp/include/nvus/sequence.hpp:22-25`). Python: `nvus._scanner.compile_plane_waves(angles_deg,
  config=PlaneWaveConfig())` (`US:cpp/src/bindings.cpp:191-192`). **This is the only sequence
  compiler; it knows only plane waves.**
- `struct Event { event_id; rx_side; samples; interval_ns; packet; }`,
  `struct PreparedSequence { id; config_id; events; programming_manifest; resources; rx_start_ticks; }`
  (`US:cpp/include/nvus/runtime.hpp:20-39`).
- Execution: `Scanner(hardware).run(prepared, RunOptions)` -> `Run.start()/next_observation()/
  submit_next()` (`US:cpp/src/bindings.cpp:167-180`); or `PolicyExecutor` (section 2.1). Note
  `PolicyExecutor` always calls `compile_plane_waves(angles)` with the default config
  (`US:cpp/src/policy_executor.cpp:43`), and accepts `rollout <= 32`, `groups <= 1024`,
  `max_duration_ms <= 120000` (`US:cpp/src/policy_executor.cpp:51-59`).
- Python dataclasses `Waveform, Transmit, Receive, Event, BoardConfig, PreparedEvent,
  PreparedSequence` (`US:python/nvus/scanner_sequence.py:7-98`) are "Immutable acquisition metadata
  for reconstruction and recording" (line 3), not a compiler. `Transmit.aperture` exists
  (line 18) but nothing programs it.

### 5.3 Parameter by parameter

| Parameter | How it is set | Limits baked in code |
| --- | --- | --- |
| Delays | 128 values, units = half-cycles of `BF_CLOCK_HZ = 200_000_000.0` (2.5 ns). Word = 14-bit integer cycles + half-cycle bit 14 | max 16383 cycles = 81.9 us; `delay_word` raises on overflow (`US:python/nvus/tx.py:190-196,236-249`); C++ requires `cycles < 16383.5` (`US:cpp/src/sequence.cpp:71`) |
| Delay memory | Regs `0x80 + word`, 2 words per profile, 16 pages selected by regs 0x02/0x03; profile select regs 0x0D..0x14 (`BF_PROF_SEL`); auto-increment reg 0x15 bit 12 | memory 0x80..0x27F (`US:python/nvus/tx.py:27-35,183-195,661-665`); native path uses `delay_start = 32`, waveform word 9 (`US:cpp/src/sequence.cpp:79`) |
| Apodization | **not found** as a programmable quantity. Reconstruction rejects anything but the full aperture (`US:python/nvus/scanner_processing.py:119-120`). Pulser power-down regs 0x2D/0x2E are only ever written 0 or all-ones (`US:python/nvus/tx.py:156-158,751-752`) | |
| Pulse shape | 3-level pattern in TX73L64 pattern memory; `burst_pattern(frequency_hz, cycles=2, *, quiet=False, invert=False)` writes words 0x89/0x8A | frequency in {5, 6.25, 8.333 MHz} (half-period 20/16/12 clocks), cycles in {1, 2, 6} (`US:python/nvus/tx.py:199-223`). Shipped presets: 10, 2.5, 5 MHz (`US:python/nvus/tx.py:48-68`). Native path: fixed `{0x89: 0x20001, 0x8a: 0x9d99}` = 5 MHz, 2 cycles (`US:cpp/src/sequence.cpp:121`) |
| PRF | Native: `interval_us` per event, host-paced by default (`US:cpp/src/runtime.cpp:286-297`) or FPGA auto-start with `fpga_auto_interval_ticks` (`US:cpp/include/nvus/hsb_backend.hpp:38-47`). Python: `pri=` / `interval_s` | native requires `interval_us >= 10.1 + 0.4 + samples/50 + 20` (`US:cpp/src/sequence.cpp:111`); Python floor comment at `US:python/nvus/evm.py:309-313` |
| Receive window | Native: derived from `depth_mm` only: `ceil(2 * depth_mm*0.001*1.1 / c * 50e6)`, rounded up to even | **`<= 65536` samples** (`US:cpp/src/sequence.cpp:106-110`); all events in a sequence must share one length (`US:cpp/src/hsb_backend.cpp:82-83`) |
| Receive start | Native: `rx_start_ticks = 80` (0.4 us at 200 MHz) after TX sync (`US:cpp/src/sequence.cpp:120`); Python: `capture_delay_s` | 32-bit tick registers |
| Number of transmits | Native: 1..2048 angles per compile, 2 events each (`US:cpp/src/sequence.cpp:100,130`) | wire sequence counter is 16-bit; wrap only with `allow_sequence_wrap` (`US:cpp/src/hsb_backend.cpp:165-171`) |
| Sample rate | fixed 50 MSPS; `_HsbSession` rejects anything but 50 MSPS, 200 MHz sync, 10.1 us TX delay | `US:python/nvus/scanner_hardware.py:199-201` |

### 5.4 What gets written where

| Device | Path | Content |
| --- | --- | --- |
| TX73L64 x2 | Idle: FPGA "TX-HS" SPI bridge at `0x7000_0000` (`CTRL 0x00, CFG 0x04, TX1_ADDR 0x08, TX1_DATA 0x0C, TX2_ADDR 0x10, TX2_DATA 0x14, STATUS 0x80, READ_DATA 0x84`) (`US:python/nvus/bus.py:45-53,380-415`) | init list `TX_GUI_INIT_WRITES`, pattern memory, delay profiles, TRSW, attenuation (`US:python/nvus/tx.py:15-25`) |
| TX73L64 x2 | Per ping (native): one RoCE packet to the FPGA. 32-byte header (magic `0x44585455`, version 1, 16-bit sequence at offset 8) + 67 operations x 32 bytes = **2176 bytes**: op0 read-disable, op1 reg 0x1A (mux), then 16 x (page A, page B, delay word 0, delay word 1), op66 reg 0x33 (waveform start word) | `US:cpp/src/sequence.cpp:20-44,77-95`; validator `US:cpp/src/hsb_backend.cpp:21-66`; FPGA limits `TX_HS_SPI_DEFAULT_MAX_OPS_PER_BANK = 128`, 2 banks (`US:fpga/rtl/tx_hs_spi/tx_hs_spi_pkg.sv:41-43`) |
| AFE5932 x2 (`rx1`, `rx2`) | HSB SPI bus 0, selected by board-control mux bits [1:0] at `0x6000_0000` (`US:python/nvus/bus.py:9-10,14,30,290-302`) | init `AFE_GUI_INIT_WRITES = (0x11,0x0002),(0x28,0x0001),(0x95,0x0241),(0x99,0x0028),(0xE0,0x0000)`; LPF reg `0x92` bits [1:0] (0 = 15 MHz, 1 = 10 MHz); test patterns from reg `0x6F` (`US:python/nvus/afe.py:9-31`; `US:python/nvus/scanner_hardware.py:292`). Nothing else. |
| FPGA sync sequencer | `SYNC_CTRL = 0x4000_0000`: `START 0x00, CFG 0x04, WAKEUP_CYCLES 0x08, TX_SYNC_DELAY 0x0C, TX_SYNC_PULSE 0x10, RX_SYNC_DELAY 0x14, RX_SYNC_PULSE 0x18, CAPTURE_START_DELAY 0x1C, EVENT_MASK 0x20, SEQUENCE_COUNT 0x24, SEQUENCE_INTERVAL 0x28, TX_UPDATE_GUARD 0x2C, TX_UPDATE_RELEASE 0x30` | `US:fpga/rtl/sync/sync_ctrl.sv:50-62`; `US:python/nvus/fpga.py:14,18-49`; native writes `US:cpp/src/hololink_transport.cpp:346-381` |
| FPGA capture | `0x5000_0000`, `LENGTH_RECORDS = frame_bytes / 48` | `US:cpp/src/hololink_transport.cpp:340-344` |
| FPGA update ingress | `TX_UPDATE_CTRL = 0x0120_0000` | `US:python/nvus/fpga.py:16`; `US:cpp/src/hololink_transport.cpp:383-387` |

### 5.5 Verdict: one wide/unfocused transmit and a ~150,000-sample window per ping

**No FPGA change is indicated by the source. The existing API does not allow it as shipped; host
code must be edited or bypassed.**

Geometry check (derived): 150,000 samples = 3.0 ms = 14,400,000 bytes = 300,000 records, a whole
number of 192-byte groups; at 1500 m/s that is 2.25 m range.

| Layer | Accepts 150,000 samples? | Source |
| --- | --- | --- |
| FPGA `capture_ctrl` length | Yes, 32-bit record count, no clamp | `US:fpga/rtl/capture/capture_ctrl.sv:78,163` |
| FPGA sync timers | Yes, 32-bit counts at 200 MHz (about 21 s) | `US:fpga/rtl/sync/sync_ctrl.sv:301-360` |
| HSB `DP_BUFFER_LENGTH` | Yes, 32-bit `frame_size` | `HSB@2.7.0:src/hololink/core/data_channel.cpp:186` |
| `HsbBackend` | Yes, needs even samples `<= UINT32_MAX / 96` | `US:cpp/src/hsb_backend.cpp:72-75` |
| `HololinkTransport::prepare` | Yes, `bytes / 48` | `US:cpp/src/hololink_transport.cpp:344` |
| Python `Evm.init(samples=...)` / `Fpga.program_capture` | Yes, only a 32-bit check | `US:python/nvus/evm.py:105-110`; `US:python/nvus/fpga.py:166-167` |
| **`compile_plane_waves`** | **No: `requested_samples <= 65536`**, and the window is settable only through `depth_mm` (max about 0.88 m at 1480 m/s with the 1.1 factor, derived) | `US:cpp/src/sequence.cpp:106-109` |
| **`nvus capture test-pattern`** | **No: `records` 4..65536 (32,768 samples)**, and `inspect_capture` rejects frames over `65536 * 48` bytes | `US:python/nvus/capture.py:147`; `US:python/nvus/rf.py:80` |
| HSB 3.0 receiver operators (if we move to HEAD) | Yes, 14.4 MB is under `kMaxFrameBytes = 64 MiB` | `HSB:src/hololink/operators/base_receiver_op.hpp:103` |
| Ring memory at `rx_pages = 1024` | 14.75 GB; must be lowered to 32-64 (section 3.4) | derived |

Ways through, in order of least change:

1. **Python `Evm` path with existing calls only**: `EvmConfig(eval_board_mode='probe', ...)`,
   `Evm.init(delays=<128 zeros or any profile>, samples=150000, pri=...)`, then `arm`/`fire` with a
   stock `RoceReceiverOp` sized for the frame (`tools/nvus_lvds_validation.py` builds such a graph,
   `US:python/tools/nvus_lvds_validation.py:449-462`; the tool was not read in full). It is slow-SPI and host-fired per shot; whether it is acceptable for sonar ping
   rates is an open question.
2. **Native path with a hand-built `PreparedSequence`**: Python can construct `Event` and
   `PreparedSequence` objects and set `samples` directly (`US:cpp/src/bindings.cpp:56-73`), copying
   the packet layout of `compile_event_packet`; `HsbBackend` validates the packet but not the 65536
   cap. `prepared_view` and `validate_frame` are PW4cm-specific and would be replaced by our
   adapter (`US:python/nvus/native_observation.py:25,44-45`; `US:examples/scanner_cpp_live.py:344-353`).
3. **Patch `sequence.cpp`** to lift the 65536 cap and expose samples (host C++ only).

Wide/unfocused transmit: a 0-degree plane wave (all delays 0, full aperture) is available today via
`compile_plane_waves([0.0])` (`US:cpp/src/sequence.cpp:60-75`). A diverging ("wide") profile has no
helper; arbitrary 128-element delays are accepted by `Tx.load_delay_profiles`
(`US:python/nvus/tx.py:661`) and by a hand-built event packet.

Things the source cannot settle (not FPGA code, but hardware behaviour):

- Sustained 38.4 Gbit/s for 3 ms through a 1024-deep FIFO with `NO_BACKPRESSURE(1)`; the reference
  only streams for about 60 us per ping. Check `CAPTURE_STATUS` bit 8 and `dropped`.
- **Transmit frequency.** `burst_pattern` cannot produce anything below 5 MHz; the pattern byte is
  `PERIOD[7:3], LEVEL[2:0]` with "state duration is PERIOD+1 clocks" (`US:python/nvus/tx.py:202-203`),
  so a two-state pattern at 200 MHz bottoms out near 3.1 MHz (derived). Sonar carriers would need
  longer pattern words or a divided TX clock, which the code explicitly refuses for the current
  waveforms ("requires pattern mode and undivided TX clocks", `US:python/nvus/tx.py:640-641`).
  Whether the TX73L64/EVM can drive a sonar transducer at all is outside these repos.

## 6. Gain ramp / TGC

### 6.1 What the code does

All receive gain control in the repo is the **TX73L64's receive-path attenuator** (the code calls it
"receive attenuation" and "DTGC"). Two mutually exclusive modes
(`US:examples/scanner_cpp_live.py:567-568`):

| Mode | Registers (per chip, `tx1` and `tx2`) | Values | Source |
| --- | --- | --- | --- |
| Fixed | `TX_LOOPBACK_FIXED_TGC_REG = 0x1F`: bits [5:0] = code, bit 6 `TX_FIXED_ATTEN_ENABLE_MASK = 0x40` must be set | code 0..48, "nominal 0.25 dB/step" -> 0..12 dB. Default `receive_attenuation_code = 23` (5.75 dB) | `US:python/nvus/tx.py:83,86,407-421`; `US:python/nvus/scanner_sequence.py:49`; `US:examples/scanner_cpp_live.py:505-506` |
| Dynamic (DTGC), one segment | `TX_DTGC_REGS = (0x05, 0x06, 0x07)`: `0x05 = (start_tgc << 6) \| start_atten`, `0x06 = (step_count << 8) \| step_period`, `0x07 = 0`; then clear bit 6 of 0x1F "only after the complete profile is present" | `start_atten = round(start_db * 4)`, `step_count = start_atten`, tick = `2e6 / bf_clock_hz` us = 10 ns at 200 MHz | `US:python/nvus/tx.py:85,99-137,492-549` |
| Source impedance model | `TX_LOOPBACK_MODE_REG = 0x54` bits [30:29]: `{800: 0x00000000, 400: 0x20000000, 200: 0x40000000, 100: 0x60000000}` | CLI uses 800 | `US:python/nvus/tx.py:84,89-96`; `US:python/nvus/run.py:41` |

Encoder limits (`receive_dtgc_profile(start_db, start_us, ramp_us, bf_clock_hz=200e6)`):

- `0.25 <= start_db <= 12.0` in 0.25 dB steps (`US:python/nvus/tx.py:105-106`).
- `TX_DTGC_START_TGC_MAX = 0x3FF` -> start delay at most 10.23 us (`US:python/nvus/tx.py:87,118-119`).
- `TX_DTGC_STEP_PERIOD_MAX = 0xFF` -> ramp at most `255 * step_count * 10 ns`; at 12 dB
  (48 steps) that is **122.4 us**; otherwise "DTGC ramp time cannot be represented by one segment"
  (`US:python/nvus/tx.py:88,117,120-121`).

What is actually run:

- `./nvus run pw4cm`: `--tx-dtgc-enable --tx-dtgc-start-db=12 --tx-dtgc-start-us=2
  --tx-dtgc-ramp-us=10 --rx-source-ohms=800` (`US:python/nvus/run.py:41-42`). Derived register
  values: `0x05 = 0x3230` (start_tgc 200, atten 48), `0x06 = 0x3015` (48 steps, period 21),
  quantized ramp 10.08 us.
- Example script defaults: DTGC off; if enabled, 12 dB, start 10 us, ramp 40 us
  (`US:examples/scanner_cpp_live.py:509-516`); `BoardConfig` mirrors these
  (`US:python/nvus/scanner_sequence.py:51-53`).

### 6.2 Verdict on the brief

- "12 dB ... on the TX73L64": the device and the 12 dB ceiling are right.
- "over about 120 us ... as coded": **not as coded.** 122.4 us is the maximum the one-segment
  encoder allows at 12 dB; shipped settings ramp over 10 us (CLI) or 40 us (example default).
- "an unused AFE5932 0-24 dB attenuator": **not found.** The AFE5932 is only initialised
  (regs 0x11, 0x28, 0x95, 0x99, 0xE0), LPF-selected (0x92) and put in test-pattern mode
  (`US:python/nvus/afe.py:9-31`). Neither the feature nor its range can be confirmed from source.

### 6.3 Where a longer ramp could hook in

A 122 us ramp covers about the first 9 cm of a 2-3 m sonar window, so it is close to irrelevant for
us. Options visible in source:

- `0x07` is always written 0 and the encoder is called "one-segment"
  (`US:python/nvus/tx.py:100,126`). Whether 0x07 encodes further segments is **not found**.
- The tick is `2 / bf_clock_hz`; `bf_clock_hz` is a parameter (`US:python/nvus/tx.py:99,113`), but
  `_HsbSession` pins the sync clock to 200 MHz (`US:python/nvus/scanner_hardware.py:199-201`).
- A generic AFE register write exists: `Afe.write(target, reg, value)`
  (`US:python/nvus/afe.py:49-50`), and the idle sequence in `_HsbSession.open` between the
  attenuation setup and `configure_native_hsb` is the natural place for extra AFE programming
  (`US:python/nvus/scanner_hardware.py:279-299`). The AFE5932 register map is not in the repo.
  Note the native verifier only compares AFE regs 0x92 and 0x95
  (`US:cpp/src/hololink_transport.cpp:260-269`).
- Otherwise: apply range gain digitally in our own processing, with the fixed attenuation code
  recorded per ping.

## 7. Sensor Bridge

### 7.1 Host setup (HSB 3.0 doc unless marked)

| Item | Requirement | Source |
| --- | --- | --- |
| Supported hosts | IGX Thor T7000 (IGX SW 2.0.1, CX7); AGX Orin (JetPack 7.2.1, on-board Ethernet, CPU only); AGX Thor (JetPack 7.2.1, MGBE, CoE); DGX Spark (DGX OS 7.6.0, CX7); x86 Linux with NVIDIA dGPU | `HSB:docs/user_guide/setup.mdx:6-29` |
| Older platforms | "Holoscan SDK 5 does not support IGX Orin (IGX OS 1.x) or JetPack 6"; use HSB 2.7 | `HSB:docs/user_guide/setup.mdx:31-33` |
| HSB 2.7.0 (what the reference uses) | DGX OS 7.2.3+, Holoscan SDK v4.4.0, NIC MTU 4096 | `HSB@2.7.0:docs/user_guide/setup.mdx:14,403,531` |
| IPs | Board `192.168.0.2` (port 0) and `192.168.0.3` (port 1); host `192.168.0.101/24` and `192.168.0.102/24` with /32 routes | `HSB:docs/user_guide/setup.mdx:145-146,195-196` |
| NIC settings (CX7 hosts) | `ethtool.ring-rx 4096`, `802-3-ethernet.mtu 4200` (4096 in 2.7.0) | `HSB:docs/user_guide/setup.mdx:147-148,404-405` |
| RDMA device name | `ls /sys/class/infiniband`, first entry = port 0; netdev from `/sys/class/infiniband/$IN0/device/net/*` | `HSB:docs/user_guide/setup.mdx:98-126,356-383` |
| DGX Spark specific | Disable CX7 hotplug: move `/etc/nvidia/cx7-hotplug-enabled` to `...-disabled` and reboot; port 0 is the leftmost QSFP | `HSB:docs/user_guide/setup.mdx:335-350` |
| Socket receivers | `net.core.rmem_max = 31326208`; isolate a core (`isolcpus=2`), `HOLOLINK_AFFINITY` selects the receiver core | `HSB:docs/user_guide/setup.mdx:252-287,738-739` |
| PTP | `phc2sys -c $EN0 -s CLOCK_REALTIME -O 0 -S 0.0001` and `ptp4l -i $EN0 -f /etc/linuxptp/hsb-ptp.conf` with `logSyncInterval -1`, `network_transport L2`, `serverOnly 1`, `BMCA noop`, `tx_timestamp_timeout 50`, `fault_reset_interval 0`, as systemd units | `HSB:docs/user_guide/setup.mdx:746-873` |
| x86 GPUDirect | Workstation-class GPU, open kernel driver, NIC and GPU as PCIe peers; otherwise build with `-DHOLOLINK_ROCE_USE_GPU_VRAM=OFF` (pinned host memory). Symptom: `Metadata psn=0 but received_psn=...` | `HSB:docs/user_guide/setup.mdx:659-719` |
| Kernel version | **not found** as an explicit number; only OS releases are named | |
| Reference repo's own steps | Follow HSB 2.7 host setup up to `docker login`, then `ip -br address`, `rdma link show`, `ping -c 3 192.168.0.2`, `./setup` | `US:docs/setup.md:51-74` |

### 7.2 Receiver classes and parameters

HEAD (HSB 3.0, Holoscan 5): operators are `BaseReceiverOp<Derived, ParamsT>` subclasses constructed
with `(std::shared_ptr<DataChannel> hololink_channel, std::function<void()> device_start = {},
std::function<void()> device_stop = {})` (`HSB:src/hololink/operators/base_receiver_op.hpp:82-91`).
Two outputs: `tensor` (`holoscan::Tensor`, "1-D uint8, device memory", a copy of the frame in a pool
slot) and `metadata` (`hololink::CaptureMetadataT`) (`...base_receiver_op.hpp:125-126`,
`HSB:src/hololink/operators/roce_receiver/roce_receiver_op.hpp:86-90`). Frame ceiling
`kMaxFrameBytes = 64 MiB` (`...base_receiver_op.hpp:103`). Python:
`graph.op(hololink.operators.RoceReceiverOp, "receiver", data_channel=..., device_start=...,
device_stop=..., frame_context=int(cu_context), frame_size=...)`
(`HSB:python/hololink/operators/__init__.py:19-33`).

| Operator | Params struct (defaults) | Transport | Source |
| --- | --- | --- | --- |
| `RoceReceiverOp` | `RoceReceiverParams { frame_context, frame_size = 0, trim = true, ibv_name = "", ibv_port = 1, pages = 2, queue_size = 1 (<= pages), metadata_offset = 0 (auto), frame_limit = 0, frame_limit_ends_run = false, watch_ingress = false }` | RoCE v2 via ibverbs, GPU or pinned host memory | `HSB:src/hololink/operators/roce_receiver/roce_receiver_op.hpp:42-69` |
| `LinuxReceiverOp` | `LinuxReceiverParams { frame_context, frame_size, trim = true, receiver_affinity = {}, pages = 1, queue_size = 1, frame_limit, frame_limit_ends_run, watch_ingress }` | UDP socket; parses the RoCE-format packets in software | `HSB:src/hololink/operators/linux_receiver/linux_receiver_op.hpp:45-70`; `HSB:docs/user_guide/dataplane.mdx:287-301` |
| `LinuxCoeReceiverOp` | `LinuxCoeReceiverParams { ..., receiver_affinity, coe_interface, pixel_width, coe_channel (0..63), frame_limit }` | raw AF_PACKET / IEEE 1722B | `HSB:src/hololink/operators/linux_coe_receiver/linux_coe_receiver_op.hpp:49-71` |
| `FusaCoeCaptureOp` | `FusaCoeCaptureParams { hololink_channel, device_start, device_stop, interface, mac_addr, timeout, stall_timeout_ms, ... }` | Thor NvFusa CoE | `HSB:src/hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp:49-73` |
| `GpuRoceTransceiverOp`, `CpuRoceTransceiverOp` | present, not read in detail | DOCA GPUNetIO / CPU RoCE | `HSB:src/hololink/operators/gpu_roce_transceiver/`, `.../cpu_roce_transceiver/`; `HSB:MIGRATION.md:17-18` |

2.7.0 (what the reference uses): `RoceReceiverOp` registers `ibv_name` (default `"roceP5p3s0f0"`),
`ibv_port` (1), `pages` (2), `queue_size`, `metadata_offset` (0) with `spec.param`, plus
`frame_size`, `frame_context`, `hololink_channel`, `device`, `trim`
(`HSB@2.7.0:src/hololink/operators/roce_receiver/roce_receiver_op.cpp:68-74`; usage
`US:python/nvus/capture.py:123-127`). Non-operator core class, same at both revisions:
`RoceReceiver(const char* ibv_name, unsigned ibv_port, CUdeviceptr cu_buffer, size_t cu_buffer_size,
size_t cu_frame_size, size_t cu_page_size, unsigned pages, size_t metadata_offset,
const char* peer_ip, unsigned queue_size, void* host_buffer = nullptr)` with `start()`,
`blocking_monitor()`, `get_next_frame(unsigned timeout_ms, RoceReceiverMetadata&)`, `close()`,
`get_qp_number()`, `get_rkey()`, `external_frame_memory()`
(`HSB:src/hololink/operators/roce_receiver/roce_receiver.hpp:59-133`).

Build switches: `HOLOLINK_BUILD_ROCE` (defaults to whether ibverbs is found),
`HOLOLINK_BUILD_SOCKET` ON, `HOLOLINK_BUILD_FUSA` OFF, `HOLOLINK_BUILD_EMULATOR` OFF,
`HOLOLINK_ROCE_USE_GPU_VRAM` (`HSB:CMakeLists.txt:38,41,48,56,77`).

### 7.3 Which receiver on which platform

| Platform | RoCE (`RoceReceiverOp`) | Linux sockets | CoE | Source |
| --- | --- | --- | --- | --- |
| DGX Spark | **Yes**, CX7 | yes | - | `HSB:docs/user_guide/setup.mdx:19-21,328-407` |
| IGX Thor T7000 | Yes, CX7 | yes | - | `HSB:docs/user_guide/setup.mdx:6-11` |
| Jetson AGX Orin | **No.** "all network I/O is performed by the CPU without network acceleration"; 10 GbE RJ45 `end0` | **Yes (only path)** | - | `HSB:docs/user_guide/setup.mdx:12-15,224-231`; `HSB:docs/user_guide/introduction.mdx:20-23` |
| Jetson AGX Thor | **No.** Built with `-DHOLOLINK_BUILD_ROCE=0 -DHOLOLINK_BUILD_FUSA=1`; interface `mgbe0_0` | yes (`linux_` examples) | Yes, FuSa CoE over MGBE | `HSB:docs/user_guide/setup.mdx:16-18,545,558-559,584-600` |
| x86 + ConnectX | Yes, "not part of the primary reference platforms"; GPUDirect conditions apply | yes | - | `HSB:docs/user_guide/setup.mdx:22-29,650-655` |

For the 38.4 Gbit/s raw ultrasound-style stream only the RoCE hosts are plausible. The reference's
C++ transport links `hololink::operators::roce_receiver` unconditionally when built with the SDK
(`US:CMakeLists.txt:59-60`), so it cannot build for AGX Thor/Orin as is.

### 7.4 Emulator

| Topic | Finding | Source |
| --- | --- | --- |
| What it is | "a minimally featured emulation of the HSB framework's IP" for host-app development without hardware; not a performance model | `HSB:docs/user_guide/emulation.mdx:23-29,46-53` |
| Language | C++ implementation + Python API (nanobind 3.0.1), module `hololink.emulation` (`import hololink.emulation as hemu`) | `HSB:docs/user_guide/emulation.mdx:69,205,536-537` |
| Classes | `HSBEmulator(config = HSB_EMULATOR_CONFIG)`, `DataPlane`, `RoCEv2DataPlane(hsb, IPAddress, data_plane_id, sensor_id)`, `COEDataPlane`, `using LinuxDataPlane = RoCEv2DataPlane`, `BaseTransmitter` (`RoCEv2Transmitter`, `COETransmitter`), `I2CPeripheral`, `IPAddress` / `IPAddress_from_string` | `HSB:src/hololink/emulation/hsb_emulator.hpp:258-278`; `.../rocev2_data_plane.hpp:32,45`; `.../linux_data_plane.hpp:27`; `HSB:docs/user_guide/emulation.mdx:560-576` |
| Sending data | `int64_t DataPlane::send(const DLTensor& tensor, FrameMetadata* = DEFAULT_FRAME_METADATA)` and `send(const uint8_t* buffer, size_t buffer_size, FrameMetadata* = nullptr)`; any DLPack array, host or CUDA | `HSB:src/hololink/emulation/data_plane.hpp:229,238`; `HSB:docs/user_guide/emulation.mdx:70-73` |
| Metadata it emits | `struct FrameMetadata { flags, psn, crc, timestamp_s_high/low, timestamp_ns, bytes_written_high/low, frame_number, metadata_s_high/low, metadata_ns, reserved[80] }` | `HSB:src/hololink/emulation/base_transmitter.hpp:35-50` |
| Sensor-agnostic | "HSB Emulator itself has no built-in knowledge of any given sensor" | `HSB:docs/user_guide/emulation.mdx:59-62` |
| Feeding custom data | `serve_linux_file <ip> <file> [-s frame_size] [-r frame_rate=60] [-l frame_limit] [-g]`; a file larger than `-s` is cycled frame by frame | `HSB:src/hololink/emulation/examples/serve_linux_file.py:53-77,98-114`; `HSB:docs/user_guide/emulation.mdx:338,372-379`; loop at `.../examples/emulator_utils.py:230-260` |
| Control-plane hooks | `HSBEmulator::register_read_callback / register_write_callback(start_address, end_address, callback, ctxt)`, `write(address, value)`, `read(address, value)` | `HSB:src/hololink/emulation/hsb_emulator.hpp:324,336,347,358` |
| Gaps | "SPI and GPIO are not fully implemented"; PTP may be a no-op; cannot be reset or reprogrammed from the host; one emulator instance per device; needs root | `HSB:docs/user_guide/emulation.mdx:48-50,75,92-102,321-323` |
| Builds | Full (`-DHOLOLINK_BUILD_EMULATOR=ON`) or emulator-only (`cmake -S src/hololink/emulation -B build`, deps `cmake zlib1g-dev`, CUDA toolkit assumed, CMake >= 3.18); MCU targets `STM32F767ZI` and `TEMPLATE` (the doc says the STM32 implementation "currently includes only the control plane communication" at line 129, yet also that both data planes are "available on the Linux and STM32 target platforms" at lines 564-565) | `HSB:docs/user_guide/emulation.mdx:125-135,180-184,189-197,224-227,250,304` |
| Tested environments | x86 Ubuntu 22.04+ and Orin Nano (emulator only); AGX Orin, AGX Thor, IGX Thor, DGX Spark (host or emulator) | `HSB:docs/user_guide/emulation.mdx:117-123` |
| Loopback without hardware | Software loopback with the Linux socket receiver on one machine: `pytest` in the container (default imx477 over "linux roce"); `pytest --emulator` for the wider sweep. Accelerated receivers (`RoceReceiverOp`, `FusaCoeCaptureOp`) need "a physical connection and network isolation": cable between two ports, then `pytest --cuda-subprocesses --hw-loopback IF1 IF2` or `scripts/nsisolate.sh IF1 192.168.0.2/24` / `nsexec.sh` / `nsjoin.sh` | `HSB:docs/user_guide/emulation.mdx:79-83,383-400,402-429,435-457` |

Separately, the ultrasound repo has its own hardware-free stand-ins: `FakeHardware` (returns
`samples * 96` bytes filled with the event id, `US:cpp/src/runtime.cpp:424-440`), selected by
`--simulate` (`US:examples/scanner_cpp_live.py:250-252`), and `SimulatedTransport`
(`US:tests/native_simulated_transport.hpp:9-63`). `./nvus run pw4cm --dry-run` uses the former
(`US:python/nvus/run.py:68-69`).

Running the reference's bring-up against the HSB emulator would need callbacks for the custom
register blocks (0x4000_0000, 0x5000_0000, 0x6000_0000, 0x7000_0000, 0x8000_0000, 0x0120_0000) and
an SPI stand-in for the AFEs (`US:python/nvus/bus.py:225-227`), which the emulator does not provide.
Its default config also advertises a different UUID and `hsb_ip_version 0x2602`
(`HSB:src/hololink/emulation/hsb_config.hpp:173,185`), while the reference enumerates by UUID
`ac192bb6-24be-c23f-d07e-2e130d9d2f0f` with `BasicEnumerationStrategy(total_sensors=2,
total_dataplanes=2, sifs_per_sensor=1)` (`US:python/nvus/fpga.py:67`, `US:python/nvus/evm.py:368-378`).

## 8. Second, low-rate source: 32 channels, 24-bit, 64 kSPS

Rate (derived): 32 x 3 B x 64,000 = 6.144 MB/s = 49.2 Mbit/s.

Reusable as is:

- The HSB data plane treats a frame as opaque bytes plus a 128-byte metadata block; nothing in the
  receivers is sample-format aware (`HSB:docs/user_guide/emulation.mdx:59-62`; receiver output is
  "1-D uint8"). A block of, say, 6,400 samples = 614,400 B is a legal frame (multiple of 128).
- Receivers: `RoceReceiverOp` on CX7 hosts, or `LinuxReceiverOp` on any NIC. At 49 Mbit/s the
  socket path is the obvious choice and also works on AGX Orin / AGX Thor (section 7.3).
- Enumeration and control: `Enumerator.find_channel(channel_ip=...)`, `DataChannel`,
  `Hololink.write_uint32 / read_uint32`, `BasicEnumerationStrategy` (as used at
  `US:python/nvus/evm.py:368-378`, `US:python/nvus/capture.py:164-170`).
- The emulator as a stand-in for the board during development (section 7.4), and its `TEMPLATE` /
  STM32 targets as a starting point if the capture board is MCU-based: `send_rocev2_packet`,
  `control_plane_reply`, `DataPlane::broadcast_bootp`, `clock_gettime`
  (`HSB:docs/user_guide/emulation.mdx:478-514`).

What a custom capture board would have to supply (it must speak HSB IP or the emulator HAL):

| Item | Detail | Source |
| --- | --- | --- |
| BOOTP enumeration | vendor block: tag, vendor id "NVDA", data plane, enum version, board id, UUID, serial, `hsb_ip_version`, `fpga_crc`, `sensor_count`, `data_plane_count`, `sifs_per_sensor` | `HSB:src/hololink/emulation/hsb_config.hpp:192-206` |
| Control plane | answer 32-bit register reads/writes ("single and block 32-bit read or write ECB commands"), including the data-plane registers the host writes: packet size, buffer length, host MAC/IP/UDP port, QP, R_Key, page LSB/MSB, page increment, max buffer | `HSB:docs/user_guide/known_limitations.mdx:1-2`; `HSB@2.7.0:src/hololink/core/data_channel.cpp:184-190,267-274` |
| Data packets | RoCE v2 framing (section 3.3): opcode 0x2A, dest QP, incrementing PSN, RETH address = base + offset, R_Key, length, iCRC | `HSB:docs/user_guide/dataplane.mdx:265-281` |
| End-of-frame metadata | opcode 0x2B with immediate (page index in the low 12 bits, PSN above); 128 bytes: flags, PSN, CRC, timestamp of first sample (s + ns), bytes written, 16-bit frame number, metadata timestamp | `HSB:docs/user_guide/dataplane.mdx:305-329`; `HSB@2.7.0:.../roce_receiver.cpp:945-950` |
| Time base | PTP-disciplined timestamps if the two sources are to be aligned; the host side is the PTP master | `HSB:docs/user_guide/setup.mdx:800-846` |

Limits and gaps:

- HSB metadata has no user fields exposed (80 reserved bytes; no API to set them was found). Ping
  index, sample counter, gain state and channel map must travel **inside the payload** or be
  inferred on the host.
- The ultrasound FPGA image has a single sensor RX interface (`SENSOR_RX_IF_INST 1`,
  `US:fpga/rtl/top/HOLOLINK_def.svh:73`), so the second source cannot ride that board without FPGA
  changes; it needs its own HSB endpoint (own IP) and its own `DataChannel` + receiver on the host.
- The reference's native transport handles exactly one channel (`use_sensor(metadata, 0)`,
  `US:python/nvus/scanner_hardware.py:234`); the second source would be a separate receiver
  instance, not an extension of `HololinkTransport`.
- Frame counter is 16-bit on the wire; the host extends it (`RoceReceiverMetadata::frame_number`,
  `HSB:src/hololink/operators/roce_receiver/roce_receiver.hpp:51`).

## 9. What we reuse / what we replace

| Piece | Decision | Notes |
| --- | --- | --- |
| HSB host setup (NIC, IPs, MTU, PTP) | **Reuse** | Follow the HSB version matching the SDK we pin (2.7.0 for the reference). |
| FPGA image `ti_ultrasound_126d629c25eb.rpd` and RTL | **Reuse unchanged** | Capture length and ring are host-configurable; no FPGA change identified. |
| `nvus.Evm` / `Bus` / `Afe` / `Tx` / `Fpga` bring-up | **Reuse** | Needed even on the native path (`_HsbSession.open`). |
| `_HsbSession`, `configure_native_hsb`, `native_hsb_config` | **Reuse, parameterise** | Pass `rx_pages` 32-64; private names (leading underscore), so pin the commit. |
| `HsbBackend`, `HololinkTransport`, `Run`, `Scanner`, `Buffer` | **Reuse** | This is the acquisition core. Requires the six SDK patches. |
| `compile_plane_waves` | **Reuse for 0-degree pings up to 65536 samples; replace or patch beyond** | Section 5.5. |
| `PolicyExecutor`, `Pw4cmPolicy` | **Replace** (or bypass with `Scanner.run`) | Hard-wired to plane-wave recompilation and PW4cm limits. |
| `prepared_view`, `observation_view`, `validate_frame` | **Replace** | PW4cm-specific (32 angles, 5 MHz x 2 cycles, 64 records per frame). |
| `Observation.gpu_rf()` lease pattern | **Reuse the pattern** | Our adapter takes the `uint8` CUDA views and writes ping records. |
| `recon.unpack` / `decode.cu` bit unpacking | **Reuse as reference for our decoder** | 12-bit, 2 records per sample, lane/parity order. |
| `Processor`, `Beamformer`, `das.cu`, `focused_recon`, `mimicknet`, `scanner_display` | **Replace** | Our own backprojection and display. |
| Receive attenuation / DTGC helpers in `Tx` | **Reuse for fixed attenuation; DTGC of little use** | Section 6.3. |
| `ObservationRecorder`, `RfRecording` | **Replace with our ping-record writer** | Useful as a format example (`US:python/nvus/scanner.py:97-120`). |
| `nvus capture test-pattern dc|ramp` | **Reuse for bring-up** | Up to 32,768 samples per frame. |
| HSB `RoceReceiverOp` / `LinuxReceiverOp` | **Reuse for the low-rate source and for any Holoscan-graph front end** | Mind the 2.7.0 vs 3.0 API split. |
| HSB emulator | **Reuse for CI of the low-rate path and of our adapter** | Not a drop-in for the ultrasound board. |

Adapter inputs available per ping: `event_id`, `rx_side` (0/1), `samples`, host `timestamp_ns`
(`steady_clock`), the 2176-byte TX packet, `interval_ns`, the sequence `id`/`config_id`, and the
`uint8` buffer of `samples * 96` bytes (`US:cpp/include/nvus/runtime.hpp:20-25,90-104`).

## 10. Open questions

1. Which SDK line do we build on: the reference's HSB 2.7.0 + Holoscan 4.4.0 + six patches, or
   HSB 3.0 / Holoscan 5 (reference not ported; native `RoceTransmitter` API absent upstream)?
2. Can the board stream 3 ms at 38.4 Gbit/s without `capture_ctrl` FIFO overflow or receiver drops?
   Not determinable from source.
3. Actual packet payload on the wire (1408 B from the 1500 default, or larger)? The FPGA allows
   `HOST_MTU 4096`; the host default is 1500 and the reference sets no override.
4. Does shrinking `rx_pages` after a 1024-page run need anything beyond the `control.reset()` already
   in `_HsbSession.open`?
5. Can the TX73L64 / AFE5932EVM front end work at sonar frequencies and transducer impedances at
   all? Pattern encoding below 5 MHz, clock division and AFE filter corners are not in the repos
   (LPF choices are 10 or 15 MHz only).
6. AFE5932 gain structure (LNA/PGA/attenuator, any TGC input): not found. What do registers 0x11,
   0x28, 0x95, 0x99, 0xE0 select? Needs the TI datasheet.
7. TX73L64 register 0x07 and multi-segment DTGC: not found.
8. PTP timestamps of the first sample exist in HSB metadata but are dropped by `HsbFrame`. Do we
   patch `HololinkTransport::receive` to keep `timestamp_s/ns` for SAS navigation alignment? (Host
   C++ only, no FPGA change.) Is the FPGA's PTP actually locked on this image?
9. Two-half aperture across two pings on a moving platform: acceptable for SAS/FLS, or do we treat
   each half as an independent 64-element ping?
10. Maximum practical ping rate: the native path sends a TX update packet and (without
    `fpga_auto_start`) a host trigger per ping; `Run` is host-paced. Jitter is unqualified
    ("hardware_launch_timing_qualified=False", `US:python/nvus/scanner_timing.py:49`).
11. The reference caps runs at 1024 frames / 120 s (`US:python/nvus/run.py:13,25`,
    `US:cpp/src/policy_executor.cpp:51-56`); a survey needs continuous operation, so the executor
    limits must go.
12. For the low-rate board: FPGA with HSB IP, or MCU using the emulator HAL? The HSB doc is
    inconsistent about whether the STM32 target has a data plane (section 7.4).
13. Licence check for redistributing the FPGA build inputs (Altera-only clause) if we ever rebuild.

## 11. First hardware-run checklist (draft)

Host and link
- [ ] Host is DGX Spark or ThinkStation PGX with CX7; OS at the version the pinned HSB expects.
- [ ] `/etc/nvidia/cx7-hotplug-enabled` disabled (Spark); reboot done.
- [ ] `hololink-$EN0` connection: `192.168.0.101/24`, route to `192.168.0.2/32`, `ring-rx 4096`,
      MTU per pinned HSB doc.
- [ ] `ip -br address` shows the port UP; `rdma link show` maps an RDMA device; `ping -c 3 192.168.0.2`.
- [ ] `phc2sys` and `ptp4l` services running; journal shows `FAULTY to MASTER on INIT_COMPLETE`.
- [ ] `nvidia-smi -L` GPU selected; `./setup` completed; `./nvus doctor` clean.

Board, receive-only (no transmit, HV supplies OFF)
- [ ] FPGA programmed with the manifest; UUID `ac192bb6-24be-c23f-d07e-2e130d9d2f0f` enumerates.
- [ ] `./nvus capture test-pattern dc --count 4` and `... ramp --count 4`: PASS, no pattern errors.
- [ ] Repeat with `--records 65536`: 4/4 frames, `dropped == 0`.
- [ ] `./nvus hw read` of `CAPTURE_STATUS` (0x50000080): overflow bit 8 clear.

Long window, still receive-only
- [ ] Our own script: `Evm.init(samples=150000)` in pattern mode with `RoceReceiverOp`
      `frame_size = 14_400_000`, `pages` 32-64. Check `bytes_written`, `dropped`, frame-number
      continuity, ramp pattern over the full 3 ms.
- [ ] Record `CAPTURE_STATUS`, `CAPTURE_ACCEPTED`, `CAPTURE_LAST_CAPTURE` after each frame.
- [ ] Capture packets once to confirm payload size and PSN behaviour.
- [ ] GPU memory: ring allocation size logged and as predicted.

Native path
- [ ] `./nvus run pw4cm --dry-run`, then `--simulate` run of our adapter against `FakeHardware`.
- [ ] `_HsbSession(..., rx_pages=64, commissioning=True)` opens; register readbacks recorded.
- [ ] Reference PW4cm for a few frames with `--frames 12 --headless` and HV at the documented
      15 V / 20-30 mA limit, to prove the unmodified chain before changing anything.
- [ ] Our sequence: single 0-degree transmit, both mux halves, long window; verify the 0x1A value
      per event against which 64 channels show signal (confirms the 0-63 / 64-127 mapping).
- [ ] Fixed attenuation code sweep (0, 23, 48) on a known target; confirm 0.25 dB/step.
- [ ] Measure achieved ping interval and its jitter from host timestamps; compare with HSB PTP
      timestamps if we keep them.
- [ ] Clean stop: `PW4cm: complete` / restoration `restored: True` before HV off.

Low-rate source
- [ ] Emulator loopback on one machine: `serve_linux_file -s <frame> -r <rate> 192.168.0.2 file`
      feeding `LinuxReceiverOp`; adapter decodes 24-bit samples and embedded counters.
- [ ] Same over a cable between two ports with `nsisolate.sh`.
