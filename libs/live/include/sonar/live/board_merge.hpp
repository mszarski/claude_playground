// One stream per capture board in, the whole-array stream out (sonar-6f9): the C++ twin of
// python/gpusonar/sources/board_merge.py, whose docstring states the rules. The engine-parity tests in
// tests/python/test_board_streams.py hold both to the same merged bytes and counts.
//
// A per-board block is a GSFB version 1 block with flag bit 1 (kFlagBoard) set and its board id as the u16 at header
// offset 44; it carries that board's channels on that board's own sample counter and capture start. The merge aligns
// each board to the reference board's counter by the boards' capture starts (whole samples), waits up to wait_s for a
// board's part of each reference block, then emits whole-array blocks (or, with a part missing, drops that span or
// fills it with zeros), which the unchanged PingAssembler takes.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

constexpr std::uint16_t kFlagOverflow = 1;   ///< block flag bit 0: ADC overflow in this block
constexpr std::uint16_t kFlagBoard = 2;      ///< block flag bit 1: a per-board block; its board id is the u16 at offset 44
constexpr double kBoardWaitSeconds = 0.05;   ///< board_merge.py WAIT_S (ours)

/// Payload of one block of the plain-UDP stream: 128 samples of 32 channels of 3 bytes (flank_stream.py
/// STREAM_PAYLOAD_BYTES). Samples per block for `n_channels` channels: block_samples_for.
constexpr std::int64_t kStreamPayloadBytes = 128 * 32 * 3;
std::int64_t block_samples_for(std::int64_t n_channels, std::int64_t sample_bytes);

/// One block's bytes (flank_stream.py encode_block): counts [n][n_channels] row-major, each within the word.
/// board < 0: a whole-array block.
std::vector<unsigned char> encode_block(const std::int32_t* counts, std::int64_t n, std::int64_t n_channels, std::int64_t first,
                                        std::int64_t tx, std::int64_t index, std::uint16_t gain, std::uint16_t flags,
                                        std::int64_t start_ns, std::int64_t sample_bytes, int board = -1);

/// floor(x + 0.5): the rounding board_merge.py's half_up and this file use.
std::int64_t half_up(double x);

/// True ns from capture start to `sample` of a counter running at sample_rate (1 + ppm 1e-6) (sonar-ubg; flank_stream.py
/// stamp_offset_ns): floor(sample / (fs (1 + ppm 1e-6)) 1e9 + 0.5).
std::int64_t stamp_offset_ns(std::int64_t sample, double sample_rate, double ppm);
/// A clock-rate error in ppm is refused beyond this (flank_stream.py MAX_CLOCK_PPM).
constexpr double kMaxClockPpm = 1000.0;
/// Throws std::invalid_argument naming `where` unless `ppm` is finite and within kMaxClockPpm.
double clock_rate(double ppm, const std::string& where);

/// A board clock's rate against the host clock from its blocks' arrival times: board_merge.py ClockEstimator, whose
/// docstring states the method (earliest arrival per bin, least-squares slope over a window), same sums in the same order.
struct ClockOptions {
  bool enabled = false;          ///< estimate at all (the program's stream_clock_estimate)
  double bin_s = 1.0, window_s = 600.0;
  bool discipline = false;       ///< the stamps use the estimate once it is good enough
  double min_span_s = 300.0, max_sigma_ppm = 0.5;
};
struct ClockEstimate {
  bool valid = false;            ///< at least 3 bins: ppm and sigma_ppm are set
  double ppm = 0.0, sigma_ppm = 0.0, span_s = 0.0;
  std::int64_t bins = 0;
  std::string json() const;      ///< {"ppm", "sigma_ppm" (null when not valid), "span_s", "bins"}
};
class ClockEstimator {
 public:
  ClockEstimator(double sample_rate, const ClockOptions& options);
  /// True when the block opens a new bin.
  bool feed(std::int64_t sample_end, std::int64_t start_ns, std::int64_t arrival_ns);
  ClockEstimate estimate() const;
  /// The estimate when disciplining and it spans min_span_s with sigma at most max_sigma_ppm, else `configured`.
  double stamp_ppm(double configured) const;
  std::int64_t restarts = 0;

 private:
  double fs_;
  ClockOptions o_;
  std::int64_t bin_ = 1, window_bins_ = 1, start_ns_ = 0, latest_ = 0;
  bool started_ = false, have_latest_ = false;
  std::map<std::int64_t, std::pair<double, double>> bins_;   ///< bin -> (x, y) of its earliest arrival, s
};

struct BoardSpec {
  int id = 0;
  std::int64_t first = 0, count = 0;
  double ppm = 0.0;   ///< its clock's configured rate error (clock_ppm; sonar-ubg)
};

struct BoardMergeOptions {
  std::vector<BoardSpec> boards;   ///< every channel on exactly one board
  int reference = -1;              ///< the board whose counter, capture start and transmits the merged stream has; -1: the first
  bool zero = false;               ///< a missing part: false drops the span (the default), true fills it with zeros
  double wait_s = kBoardWaitSeconds;
  int sample_bytes = 3;
  std::int64_t block_samples = 0;  ///< samples of an output block; 0: block_samples_for(n_channels, sample_bytes)
  std::vector<std::pair<int, double>> clock_ppm;   ///< (board id, rate error in ppm); boards not named: 0
};

/// "id:ppm,id:ppm,...": the boards' rates as the key stream_board_ppm writes them.
std::vector<std::pair<int, double>> parse_board_ppm(const std::string& text);

/// "id:first:count,id:first:count,...": a board -> channel map as the key stream_board_channels writes it.
std::vector<BoardSpec> parse_board_channels(const std::string& text);
/// The map a template record's metadata.channel_board gives: board b carries the channels marked b (contiguous).
std::vector<BoardSpec> boards_from_template(const PingRecord& record);

class BoardMerger {
 public:
  /// Throws std::invalid_argument (the messages of board_merge.py, after "boards.") for a map that is not the array's,
  /// a reference not in it, or a wait that is not a number of seconds.
  BoardMerger(const BoardMergeOptions& options, std::int64_t n_channels, double sample_rate);
  /// Take one per-board block; append the whole-array blocks it completes. False, counted and with last_error set,
  /// when the block is refused (malformed, no board id, a board not in the map, another channel count).
  bool feed(const unsigned char* data, std::size_t size, std::vector<std::vector<unsigned char>>& out);
  /// The streams have ended: decide every waiting span with what the boards gave.
  void finish(std::vector<std::vector<unsigned char>>& out);
  /// The counts of board_merge.py stats(), as one JSON object.
  std::string stats_json() const;
  std::string last_error;
  std::int64_t merged_blocks = 0, dropped_blocks = 0, zero_filled_blocks = 0, bad_blocks = 0;
  std::int64_t wait() const { return wait_; }
  int reference() const;
  double clock_ppm(int id) const;   ///< a board's configured rate error

 private:
  struct Block {
    std::int64_t n = 0;
    std::vector<std::int32_t> counts;   ///< [n][count]
    std::uint16_t flags = 0;
    bool used = false;
  };
  struct Board {
    BoardSpec spec;
    bool started = false;
    std::int64_t start_ns = 0;
    std::vector<std::int64_t> left;
    std::map<std::pair<std::int64_t, std::int64_t>, Block> blocks;   ///< (capture start, own first) -> block
    bool have_frontier = false;
    std::int64_t frontier = 0;
    std::int64_t n_blocks = 0, used_blocks = 0, late_blocks = 0, repeated_blocks = 0, restarts = 0, missing_samples = 0, bad = 0;
  };
  struct Pending {
    std::int64_t first = 0, n = 0, tx = -1, index = 0, start_ns = 0, anchor = 0;
    std::uint16_t gain = 0, flags = 0;
    std::vector<std::int32_t> counts;   ///< [n][reference count]
  };
  struct Span {
    std::int64_t f, e;
    std::pair<std::int64_t, std::int64_t> key;
  };
  bool refuse(const std::string& message, Board* board = nullptr);
  std::int64_t offset(const Board& b, std::int64_t start_ns, std::int64_t anchor) const;
  std::vector<Span> spans(const Board& b, std::int64_t anchor) const;
  bool frontier(const Board& b, std::int64_t anchor, std::int64_t& out) const;
  bool wanted(const Board& b, std::int64_t ns, std::int64_t first, std::int64_t n) const;
  std::pair<std::int64_t, std::uint16_t> cover(Board& b, std::int64_t a, std::int64_t e, std::int64_t anchor, std::int32_t* dest);
  int status(const Pending& p, Board& b, bool force);
  void decide(bool force, std::vector<std::vector<unsigned char>>& out);
  void emit(const Pending& p, std::vector<std::vector<unsigned char>>& out);
  void prune();

  std::vector<Board> boards_;
  std::size_t ref_ = 0;
  std::int64_t n_channels_ = 0, sample_bytes_ = 3, wait_ = 0, block_samples_ = 0;
  double sample_rate_ = 0.0;
  bool zero_ = false;
  std::vector<Pending> pending_;
  std::vector<std::int64_t> done_;
  std::vector<std::pair<std::int64_t, std::int64_t>> named_;
  bool have_ref_frontier_ = false, have_decided_ = false;
  std::int64_t ref_frontier_ = 0, decided_to_ = 0, anchor_ = 0;
};

struct StreamOptions;   // chain.hpp

/// The merge the stream options configure (stream_boards=1 and the stream_board_* keys, D-184; the map defaults to the
/// template's channel_board), for every receiver of the program's stream: build/bin/sonar_live (pings and --listen) and
/// the Holoscan source operator (sonar-bwu). nullptr without stream_boards. Throws std::invalid_argument for
/// stream_board_ppm without stream_boards, and for stream_clock_ppm with it (the rates go per board).
std::unique_ptr<BoardMerger> make_board_merger(const StreamOptions& stream, const PingRecord& template_record);

/// One block of the stream into `assembler` (a PingAssembler or a StreamAssembler): through `merger` when there is one
/// (a block it refuses is counted in its bad_blocks), else straight. `merged` is scratch.
template <typename Assembler>
void feed_stream(BoardMerger* merger, const unsigned char* data, std::size_t size, Assembler& assembler, std::vector<PingRecord>& ready,
                 std::vector<std::vector<unsigned char>>& merged) {
  if (merger == nullptr) {
    assembler.feed(data, size, ready);
    return;
  }
  merged.clear();
  merger->feed(data, size, merged);
  for (const auto& m : merged) assembler.feed(m.data(), m.size(), ready);
}

/// The end of the stream for the merge: its waiting spans decided with what the boards gave, into `assembler`.
template <typename Assembler>
void finish_stream(BoardMerger* merger, Assembler& assembler, std::vector<PingRecord>& ready, std::vector<std::vector<unsigned char>>& merged) {
  if (merger == nullptr) return;
  merged.clear();
  merger->finish(merged);
  for (const auto& m : merged) assembler.feed(m.data(), m.size(), ready);
}

}  // namespace sonar::live
