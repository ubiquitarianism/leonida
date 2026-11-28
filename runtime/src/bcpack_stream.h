// Shared rANS block stream used by the runtime decoder and the offline history fitter.
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <span>

namespace leonida::bcpack::detail {
using namespace std;
struct Seed {
  int16_t values[8];
  bool fitted;
};
struct BC6Mode {
  int code, transformed, base, delta[3], regions, first, count;
};
#include "bcpack_tables.inc"

using Symbols = array<uint16_t, 16>;
constexpr uint32_t home = 1u << 24;

struct Model {
  uint32_t base[11]{};
  const uint32_t *frequency{};
  int width{};
  array<uint8_t, 256> first{};
  uint64_t reciprocal[9]{};
  void init(int w, int q) {
    width = w;
    frequency = freq[w][q];
    for (int p = 1; p <= w; ++p) {
      base[p] = base[p - 1] + (frequency[p - 1] << max(0, p - 2));
      reciprocal[p] = ((uint64_t{1} << 32) + frequency[p] - 1) / frequency[p];
    }
    unsigned p = 0;
    for (unsigned window = 0; window < first.size(); ++window) {
      while (p < unsigned(width) && base[p + 1] <= (window << 8)) ++p;
      first[window] = uint8_t(p);
    }
  }
  int symbol(uint32_t state, bool fast) const {
    const uint32_t cf = state & 65535;
    if (fast) {
      unsigned p = first[cf >> 8];
      while (p < unsigned(width) && cf >= base[p + 1]) ++p;
      if (!p) return 0;
      const auto quotient = uint32_t(((cf - base[p]) * reciprocal[p]) >> 32);
      return int((1u << (p - 1)) + quotient);
    }
    int p = width;
    while (p && cf < base[p]) --p;
    return p ? (1 << (p - 1)) + (cf - base[p]) / frequency[p] : 0;
  }
  uint32_t advance(uint32_t state, unsigned sym) const {
    const int p = bit_width(sym);
    uint32_t f = frequency[p],
             c = base[p] + f * (sym - (p ? 1u << (p - 1) : 0u));
    if (!c) {
      --f;
      ++c;
    }
    return f * (state >> 16) + (state & 65535) - c;
  }
};

// VIctory's GPL-3.0 BCPack decoder uses the high slot byte to locate a bucket.
// Build this accelerator solely from Leonida's own rANS frequencies.
inline const Model& model_for(int w, int q) {
  static const auto models = [] {
    array<array<Model, 16>, 8> result{};
    for (int width = 1; width <= 8; ++width)
      for (int quality = 0; quality < 16; ++quality)
        result[width - 1][quality].init(width, quality);
    return result;
  }();
  return models[w - 1][q];
}

enum class Phase { header, length, long_length, symbols };
struct Lane {
  Phase phase{Phase::header};
  const Model* model{};
  uint32_t state{};
  int bytes{}, byte{}, zeros{}, symbol{}, slot{};
  size_t remaining{};
  bool flush{}, extended{}, buffered{}, held{}, full{}, done{};

  bool step(int width, span<const uint8_t> &input, int &budget, bool fast) {
    if (done || (held && full))
      return true;
    if (!buffered && (phase != Phase::symbols || bytes)) {
      if (!budget)
        return true;
      if (input.empty())
        return false;
      --budget;
      byte = input.front();
      input = input.subspan(1);
      buffered = true;
    }
    if (held) {
      state = model->advance(state, symbol);
      slot = symbol;
      full = true;
      held = false;
    }
    if (phase == Phase::header) {
      model = &model_for(width - (byte >> 6), byte & 15);
      flush = (byte & 16) != 0;
      extended = (byte & 32) != 0;
      zeros = 0;
      buffered = false;
      phase = Phase::length;
    } else if (phase == Phase::length || phase == Phase::long_length) {
      if (phase == Phase::long_length) {
        bytes += (byte << 8) + 257;
        phase = Phase::symbols;
      } else {
        bytes = byte + !extended;
        phase = extended ? Phase::long_length : Phase::symbols;
      }
      buffered = false;
    } else {
      if (state < home) {
        if (bytes) {
          state = (state << 8) | byte;
          buffered = false;
          --bytes;
        } else if (!flush) {
          done = remaining == 0;
          phase = Phase::header;
          return true;
        }
        if (!bytes && flush && state < home) {
          if (!state || zeros >= 8)
            return false;
          while (state < home && zeros < 8) {
            state <<= 8;
            ++zeros;
          }
        }
      }
      held = state > home && remaining;
      if (held) {
        symbol = model->symbol(state, fast);
        if (symbol >= (1 << model->width))
          return false;
        --remaining;
      }
      if (!bytes && flush && state == home) {
        done = !remaining;
        phase = Phase::header;
        state = 0;
      } else if (!bytes && !flush && !remaining && !held) {
        done = true;
      } else if (state == home || (state > home && !held)) {
        return false;
      }
    }
    return true;
  }
};

inline int unzig(int value) { return value & 1 ? -((value + 1) >> 1) : value >> 1; }
inline int interleave(int a, int b, int bits = 4) {
  int v = 0;
  for (int k = 0; k < bits; ++k)
    v |= (((a >> k) & 1) | (((b >> k) & 1) << 1)) << (2 * k);
  return v;
}

struct BC7Format {
  int subsets, color, alpha, pbits, index, partition;
};
constexpr BC7Format bc7_formats[] = {{3, 4, 0, 1, 3, 4}, {2, 6, 0, 2, 3, 6},
                                     {3, 5, 0, 0, 2, 6}, {2, 7, 0, 1, 2, 6},
                                     {1, 5, 6, 0, 3, 0}, {1, 7, 8, 0, 2, 0},
                                     {1, 7, 7, 1, 4, 0}, {2, 5, 5, 1, 2, 6}};


template<class Block>
bool visit_blocks(unsigned codec, span<const uint8_t> source, size_t size, Block block, bool fast = true) {
  const bool single = codec == 1 || codec == 4;
  const unsigned count = codec >= 6 ? 16 : 12;
  const size_t elements = size / 8;
  array<Lane, 16> lanes{};
  for (auto &lane : lanes) lane.remaining = elements;
  Symbols first{};
  size_t emitted = 0;
  while (elements) {
    bool ready = true, done = true;
    int budget = 8;
    for (unsigned l = 0; l < count; ++l) {
      auto &lane = lanes[l];
      if (!lane.step(widths[codec][l], source, budget, fast))
        return false;
      ready &= lane.full;
      done &= lane.done;
    }
    if (ready) {
      if (emitted == elements)
        return false;
      Symbols tuple{};
      for (unsigned l = 0; l < count; ++l) {
        tuple[l] = uint16_t(lanes[l].slot);
        lanes[l].full = false;
      }
      if (single || emitted % 2) {
        const size_t i = single ? emitted : emitted / 2;
        if (!block(i, single ? tuple : first, tuple)) return false;
      } else {
        first = tuple;
      }
      ++emitted;
    }
    if (done)
      break;
  }
  return emitted == elements && ranges::all_of(source, [](uint8_t v) { return v == 0; });
}
} // namespace leonida::bcpack::detail
