#pragma once

// The descriptor hash table for set 0, binding 2 (docs/d3d12_vulkan.md section 5.1 and docs/design.md):
// Shaders look up bindless resource/sampler indices by FNV-1a hashing the 8-dword descriptor key (T# or padded S#)
// and probing up to 8 slots in this table. Miss or fallback yields index 0.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

namespace leonida {

// FNV-1a constants for 32-bit dwords
inline constexpr std::uint32_t kFnvOffsetBasis = 2166136261u;
inline constexpr std::uint32_t kFnvPrime = 16777619u;
inline constexpr std::uint32_t kDescriptorProbes = 8;
// An entry's last dword is 0 while the slot is empty. This bit marks it occupied, and the low bits (kKindMask) say
// what the index is good for: one bit per array the image is bound in, sampled and storage.
inline constexpr std::uint32_t kSlotOccupiedMarker = 0x80000000u;
inline constexpr std::uint32_t kKindMask = 0x1FFFu;   // bits 0..6 sampled, 8..12 storage

// What an index is bound in, for the library whose modules read it (`descriptor_table` tagged in their index.json
// meta; tools/rdna2spirv.py reads the same bits with OpBitFieldUExtract). Bit `dimension` is that dimension's
// sampled array (dimensions 0..6, ImageDimension in vulkan_bindless.h); bit 8 + dimension its storage arrays
// (dimensions 0..4 have them). A table for a plain library leaves every kind bit zero.
inline constexpr std::uint32_t kind_sampled_bit(std::uint32_t dimension) {
    return dimension < 7 ? (1u << dimension) : 0u;
}
inline constexpr std::uint32_t kind_storage_bit(std::uint32_t dimension) {
    return dimension < 5 ? (1u << (8 + dimension)) : 0u;
}

// Key types:
// T# key is 8 dwords.
// S# key is 4 dwords followed by 4x 0xFFFFFFFF.
using DescriptorKey = std::array<std::uint32_t, 8>;

// Which hash the table, and every module it serves, probe with. Both run the same FNV-1a loop over the key's dwords;
// fnv1a_fmix32 then finalizes it with MurmurHash3's fmix32, which spreads keys whose low FNV-1a bits cluster (the T#s
// that differ only in a high nibble of dword 0 all land in one slot today). A translated index.json declares its
// modules' kind in the per-module meta (`descriptor_hash`); the runtime picks one kind for its table before the first
// insert and uses it for every insert, lookup, relocation and rehash.
enum class DescriptorHashKind : std::uint8_t { fnv1a = 0, fnv1a_fmix32 = 1 };

inline constexpr const char* descriptor_hash_kind_name(DescriptorHashKind kind) {
    return kind == DescriptorHashKind::fnv1a_fmix32 ? "fnv1a_fmix32" : "fnv1a";
}

// The bare FNV-1a loop: the hash every module has probed with until the marker says otherwise.
inline std::uint32_t hash_descriptor_key_fnv1a(const std::uint32_t key[8]) {
    std::uint32_t h = kFnvOffsetBasis;
    for (int i = 0; i < 8; ++i) {
        h = (h ^ key[i]) * kFnvPrime;
    }
    return h;
}

// MurmurHash3's fmix32 finalizer (logical shifts, arithmetic mod 2^32): what fnv1a_fmix32 applies after the FNV loop.
inline std::uint32_t finalize_descriptor_hash(std::uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

inline std::uint32_t hash_descriptor_key(const std::uint32_t key[8], DescriptorHashKind kind) {
    const std::uint32_t h = hash_descriptor_key_fnv1a(key);
    return kind == DescriptorHashKind::fnv1a_fmix32 ? finalize_descriptor_hash(h) : h;
}

inline std::uint32_t hash_descriptor_key(const DescriptorKey& key, DescriptorHashKind kind) {
    return hash_descriptor_key(key.data(), kind);
}

// The old default, so a table or a test that does not name a kind keeps probing with bare FNV-1a.
inline std::uint32_t hash_descriptor_key(const std::uint32_t key[8]) {
    return hash_descriptor_key(key, DescriptorHashKind::fnv1a);
}

inline std::uint32_t hash_descriptor_key(const DescriptorKey& key) {
    return hash_descriptor_key(key.data());
}

inline DescriptorKey make_tsharp_key(const std::uint32_t tsharp[8]) {
    DescriptorKey k{};
    std::memcpy(k.data(), tsharp, sizeof(DescriptorKey));
    return k;
}

inline DescriptorKey make_ssharp_key(const std::uint32_t ssharp[4]) {
    DescriptorKey k{};
    k[0] = ssharp[0];
    k[1] = ssharp[1];
    k[2] = ssharp[2];
    k[3] = ssharp[3];
    k[4] = 0xFFFFFFFFu;
    k[5] = 0xFFFFFFFFu;
    k[6] = 0xFFFFFFFFu;
    k[7] = 0xFFFFFFFFu;
    return k;
}

// Memory layout of each entry in the table: 10 uint32s = 40 bytes. The last word is the kind a translated shader
// reads next to the index, and its occupied bit; a module that does not read it never looks at it.
struct DescriptorEntry {
    std::uint32_t key[8] = {};
    std::uint32_t index = 0;
    std::uint32_t kind = 0; // 0 when empty, kSlotOccupiedMarker | (kKindMask bits) when occupied
};
static_assert(sizeof(DescriptorEntry) == 40);

inline constexpr bool slot_occupied(const DescriptorEntry& entry) {
    return (entry.kind & kSlotOccupiedMarker) != 0;
}
inline constexpr std::uint32_t slot_kind(const DescriptorEntry& entry) { return entry.kind & kKindMask; }

class DescriptorTable {
public:
    enum class InsertResult { added, unchanged, replaced, added_with_relocation };

    // `hash_kind` is the finalizer every insert, lookup, relocation and rehash of this table uses. It is fixed here,
    // before the first insert: one table serves all the device's shaders, which all probe with the kind the device
    // chose from their indexes (or LEONIDA_DESCRIPTOR_HASH). The default is the old bare FNV-1a.
    explicit DescriptorTable(std::uint32_t initial_capacity = 16,
                             DescriptorHashKind hash_kind = DescriptorHashKind::fnv1a)
        : hash_kind_(hash_kind) {
        // Slots use nonnegative int indices internally. Larger powers also wrap the capacity counter.
        if (initial_capacity > max_capacity) throw std::length_error("Descriptor table capacity exceeds slot range");
        std::uint32_t cap = 16;
        while (cap < initial_capacity) cap <<= 1;
        mask_ = cap - 1;
        entries_.resize(cap);
    }

    // The kind this table places and finds keys with (immutable after construction).
    DescriptorHashKind hash_kind() const { return hash_kind_; }

    // Inserts or updates a key -> (index, kind) mapping. If the key already exists in one of the 8 probe slots,
    // its entry is updated. Try one bounded relocation before growing. Load never exceeds 50%. `kind` is what a
    // shader that reads it (a `descriptor_table` tagged module) uses to tell which array the index is bound in; a
    // sampler's entry has none.
    bool insert(const std::uint32_t key[8], std::uint32_t index, std::uint32_t kind = 0) {
        insert_or_update(key, index, kind);
        return true;
    }

    // Report the change under the same lock so the GPU caller can synchronize writes to live slots.
    InsertResult insert_or_update(const std::uint32_t key[8], std::uint32_t index, std::uint32_t kind = 0) {
        std::lock_guard lock(mutex_);
        return insert_locked(key, index, kind);
    }

    bool insert(const DescriptorKey& key, std::uint32_t index, std::uint32_t kind = 0) {
        return insert(key.data(), index, kind);
    }

    // Queries the index associated with key. Returns 0 if not found.
    std::uint32_t lookup(const std::uint32_t key[8]) const {
        std::lock_guard lock(mutex_);
        const std::uint32_t h = hash_descriptor_key(key, hash_kind_);
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (slot_occupied(entry) && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                return entry.index;
            }
        }
        return 0; // Exceeded 8 probes without a match
    }

    std::uint32_t lookup(const DescriptorKey& key) const {
        return lookup(key.data());
    }

    bool contains(const std::uint32_t key[8]) const {
        std::lock_guard lock(mutex_);
        const std::uint32_t h = hash_descriptor_key(key, hash_kind_);
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (slot_occupied(entry) && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                return true;
            }
        }
        return false;
    }

    bool contains(const DescriptorKey& key) const {
        return contains(key.data());
    }

    // Erases a key from the table. Returns true if the key was found and erased, false otherwise.
    // Only the key's own slot changes (it becomes empty): no other entry moves. Every lookup, here and in the
    // shaders (rdna2spirv.py desc_probe), compares all kDescriptorProbes slots and never stops at an empty one, so
    // a hole breaks no probe chain and inserts reuse it. Keeping the other entries in place is what lets the GPU
    // copy of the table take an erase while shaders still probe it (BindlessSet::flush_replaces_table).
    bool erase(const std::uint32_t key[8]) {
        std::lock_guard lock(mutex_);
        return erase_locked(key);
    }

    bool erase(const DescriptorKey& key) {
        return erase(key.data());
    }

    void clear() {
        std::lock_guard lock(mutex_);
        std::fill(entries_.begin(), entries_.end(), DescriptorEntry{});
        occupied_count_ = 0;
        all_changed_ = true;
        changed_slots_.clear();
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return occupied_count_;
    }

    std::size_t capacity() const {
        std::lock_guard lock(mutex_);
        return entries_.size();
    }

    std::uint32_t mask() const {
        std::lock_guard lock(mutex_);
        return mask_;
    }

    std::size_t size_dwords() const {
        std::lock_guard lock(mutex_);
        return 1 + entries_.size() * 10;
    }

    std::size_t size_bytes() const {
        return size_dwords() * sizeof(std::uint32_t);
    }

    // Exports the table buffer format consumed by SPIR-V set 0, binding 2.
    // Word 0 is mask; followed by entries[mask + 1] of 10 dwords: key[8], index, kind.
    std::vector<std::uint32_t> build_gpu_buffer(bool clear_occupied = false) const {
        std::lock_guard lock(mutex_);
        std::vector<std::uint32_t> out(1 + entries_.size() * 10, 0);
        out[0] = mask_;
        for (std::size_t s = 0; s < entries_.size(); ++s) {
            const std::size_t base = 1 + s * 10;
            std::memcpy(&out[base], entries_[s].key, sizeof(entries_[s].key));
            out[base + 8] = entries_[s].index;
            out[base + 9] = clear_occupied ? slot_kind(entries_[s]) : entries_[s].kind;
        }
        return out;
    }

    // What a shader's copy of the table needs re-uploaded since the last take_changes().
    struct Changes {
        bool all = false;                  // the mask (table grew or was cleared): every slot and the header
        std::vector<std::uint32_t> slots;  // slots whose 10 dwords changed, in no particular order
    };

    Changes take_changes() {
        std::lock_guard lock(mutex_);
        Changes changes;
        changes.all = all_changed_;
        changes.slots = std::move(changed_slots_);
        changed_slots_.clear();
        all_changed_ = false;
        return changes;
    }

    // One slot's 10 dwords (key[8], index, kind) to `dest`; false when the slot is out of range.
    bool read_slot(std::uint32_t slot, std::uint32_t* dest) const { return read_slots(slot, 1, dest); }

    // `count` consecutive slots' 10 dwords each to `dest`, under one lock; false when any is out of range.
    bool read_slots(std::uint32_t first, std::uint32_t count, std::uint32_t* dest) const {
        if (count > 0 && !dest) return false;
        std::lock_guard lock(mutex_);
        if (first > entries_.size() || count > entries_.size() - first) return false;
        for (std::uint32_t i = 0; i < count; ++i) {
            const DescriptorEntry& entry = entries_[first + i];
            std::memcpy(dest + i * 10, entry.key, sizeof(entry.key));
            dest[i * 10 + 8] = entry.index;
            dest[i * 10 + 9] = entry.kind;
        }
        return true;
    }

    // Serializes directly into `dest` (must hold at least size_bytes()) without allocating a temporary vector.
    void write_to_buffer(void* dest, bool clear_occupied = false) const {
        if (!dest) return;
        std::lock_guard lock(mutex_);
        auto* out = static_cast<std::uint8_t*>(dest);
        std::memcpy(out, &mask_, sizeof(mask_));
        out += sizeof(mask_);
        for (const auto& entry : entries_) {
            std::uint32_t slot[10];
            std::memcpy(slot, entry.key, sizeof(entry.key));
            slot[8] = entry.index;
            slot[9] = clear_occupied ? slot_kind(entry) : entry.kind;
            std::memcpy(out, slot, sizeof(slot));
            out += sizeof(slot);
        }
    }

private:
    void mark(std::uint32_t slot) { changed_slots_.push_back(slot); }

    // All eight slots for the incoming key are occupied. Try moving one resident
    // to the first empty slot beyond that window, within its own eight probes.
    // A later hole is farther from every resident, so failure here is final for a single move.
    // `moved_to` receives the hole the resident moved into, so the caller can mark that slot changed too.
    static int make_probe_slot(std::vector<DescriptorEntry>& entries, std::uint32_t hash, DescriptorHashKind kind,
                               std::uint32_t* moved_to = nullptr) {
        const auto mask = static_cast<std::uint32_t>(entries.size() - 1);
        for (auto offset = kDescriptorProbes; offset < 2 * kDescriptorProbes - 1; ++offset) {
            const auto hole = (hash + offset) & mask;
            if (slot_occupied(entries[hole])) continue;
            for (auto k = kDescriptorProbes; k > 0; --k) {
                const auto slot = (hash + k - 1) & mask;
                const auto home = hash_descriptor_key(entries[slot].key, kind);
                if (((hole - home) & mask) < kDescriptorProbes) {
                    entries[hole] = entries[slot];
                    if (moved_to) *moved_to = hole;
                    return static_cast<int>(slot); // caller immediately overwrites the resident
                }
            }
            break;
        }
        return -1;
    }

    InsertResult insert_locked(const std::uint32_t key[8], std::uint32_t index, std::uint32_t kind) {
        while (true) {
            const std::uint32_t h = hash_descriptor_key(key, hash_kind_);
            int empty_slot = -1;

            for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                const std::uint32_t slot = (h + k) & mask_;
                auto& entry = entries_[slot];
                if (slot_occupied(entry)) {
                    if (std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                        if (entry.index == index && slot_kind(entry) == kind) return InsertResult::unchanged;
                        // A new index or a new kind: either way the slot's copy of the entry must reach the GPU.
                        entry.index = index;
                        entry.kind = kSlotOccupiedMarker | (kind & kKindMask);
                        mark(slot);
                        return InsertResult::replaced;
                    }
                } else if (empty_slot < 0) {
                    empty_slot = static_cast<int>(slot);
                }
            }

            if ((occupied_count_ + 1) * 2 > entries_.size()) {
                grow_locked();
                continue;
            }

            // Nine distinct keys with the same full hash can never fit the shader's eight probes,
            // regardless of capacity. Refuse before allocating ever larger tables.
            if (empty_slot < 0) {
                std::uint32_t same_hash = 0;
                for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                    const auto& entry = entries_[(h + k) & mask_];
                    if (hash_descriptor_key(entry.key, hash_kind_) == h) ++same_hash;
                }
                if (same_hash == kDescriptorProbes)
                    throw std::length_error("Descriptor hash collision exceeds shader probe limit");
            }
            const bool relocate = empty_slot < 0;
            std::uint32_t moved_to = 0;
            if (relocate) empty_slot = make_probe_slot(entries_, h, hash_kind_, &moved_to);
            if (empty_slot >= 0) {
                auto& entry = entries_[empty_slot];
                std::memcpy(entry.key, key, sizeof(entry.key));
                entry.index = index;
                entry.kind = kSlotOccupiedMarker | (kind & kKindMask);
                ++occupied_count_;
                if (relocate) mark(moved_to);
                mark(std::uint32_t(empty_slot));
                return relocate ? InsertResult::added_with_relocation : InsertResult::added;
            }

            grow_locked();
        }
    }

    bool erase_locked(const std::uint32_t key[8]) {
        const std::uint32_t h = hash_descriptor_key(key, hash_kind_);
        std::uint32_t found_slot = 0;
        bool found = false;
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask_;
            const auto& entry = entries_[slot];
            if (slot_occupied(entry) && std::memcmp(entry.key, key, sizeof(entry.key)) == 0) {
                found_slot = slot;
                found = true;
                break;
            }
        }
        if (!found) return false;

        // No backward shift: see erase().
        entries_[found_slot] = DescriptorEntry{};
        mark(found_slot);
        --occupied_count_;
        return true;
    }

    void grow_locked() {
        if (entries_.size() >= max_capacity) throw std::length_error("Descriptor table capacity exhausted");
        std::uint32_t new_cap = static_cast<std::uint32_t>(entries_.size() * 2);
        while (true) {
            std::vector<DescriptorEntry> new_entries(new_cap);
            const std::uint32_t new_mask = new_cap - 1;
            bool rehash_ok = true;

            for (const auto& old : entries_) {
                if (!slot_occupied(old)) continue;
                const std::uint32_t h = hash_descriptor_key(old.key, hash_kind_);
                int slot = -1;
                for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
                    const std::uint32_t s = (h + k) & new_mask;
                    if (!slot_occupied(new_entries[s])) {
                        slot = static_cast<int>(s);
                        break;
                    }
                }
                if (slot < 0) slot = make_probe_slot(new_entries, h, hash_kind_);
                if (slot < 0) {
                    rehash_ok = false;
                    break;
                }
                new_entries[slot] = old;
            }

            if (rehash_ok) {
                entries_ = std::move(new_entries);
                mask_ = new_mask;
                all_changed_ = true;
                changed_slots_.clear();
                break;
            }
            if (new_cap >= max_capacity) throw std::length_error("Descriptor probe range exhausted");
            new_cap *= 2;
        }
    }

    static constexpr std::uint32_t max_capacity = std::uint32_t{1} << 31;
    mutable std::mutex mutex_;
    const DescriptorHashKind hash_kind_;
    std::uint32_t mask_ = 15;
    std::vector<DescriptorEntry> entries_;
    std::size_t occupied_count_ = 0;
    bool all_changed_ = false;                  // clear() or a grow: the mask and every slot changed
    std::vector<std::uint32_t> changed_slots_;  // slots inserted, replaced or moved since take_changes()
};

} // namespace leonida
