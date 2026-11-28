// src/guarded_copy.h: a copy made in a __try frame turns an access violation into an error, under Wine too.
//
// Each helper reads from a reserved, never committed page inside a __try and must land in the __except. Then
// leonida::guarded::move is checked against memmove for every overlap direction and a range of sizes and offsets,
// and copy and read_word for plain values.
//
// guarded_copy_test --crt makes the same read with the CRT's memcpy instead. Under Wine the fault raised inside
// ucrtbase does not reach the __except and the process ends (exit 5 under Proton Experimental, Wine 11.0); on Windows
// it is caught. Not registered with CTest: it shows why the helpers exist.
#include "../src/guarded_copy.h"

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

unsigned checks = 0, failures = 0;
void expect(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

enum class How { copy, move, word, crt };

bool read_guarded(unsigned char* to, const unsigned char* from, std::size_t bytes, How how) {
    __try {
        switch (how) {
        case How::copy: leonida::guarded::copy(to, from, bytes); break;
        case How::move: leonida::guarded::move(to, from, bytes); break;
        case How::word: to[0] = static_cast<unsigned char>(leonida::guarded::read_word(from)); break;
        case How::crt: std::memcpy(to, from, bytes); break;
        }
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER
                                                                  : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    // 64 KiB reserved and never committed: every read of it faults.
    auto* reserved = static_cast<unsigned char*>(VirtualAlloc(nullptr, 1 << 16, MEM_RESERVE, PAGE_READWRITE));
    if (!reserved) return std::fprintf(stderr, "VirtualAlloc failed\n"), 1;
    static unsigned char out[8192];

    if (argc > 1 && !std::strcmp(argv[1], "--crt")) {
        const bool ok = read_guarded(out, reserved, sizeof out, How::crt);
        std::printf("guarded_copy_test --crt: the CRT's memcpy from a reserved page %s\n",
                    ok ? "returned (no fault?)" : "was caught by __except");
        return 0;
    }

    expect(!read_guarded(out, reserved, sizeof out, How::copy), "a fault in guarded::copy reaches the __except");
    expect(!read_guarded(out, reserved, sizeof out, How::move), "a fault in guarded::move reaches the __except");
    expect(!read_guarded(out, reserved, 8, How::word), "a fault in guarded::read_word reaches the __except");
    // A fault part-way: the first page readable, the next one not.
    auto* half = static_cast<unsigned char*>(VirtualAlloc(nullptr, 2 * 4096, MEM_RESERVE, PAGE_READWRITE));
    expect(half && VirtualAlloc(half, 4096, MEM_COMMIT, PAGE_READWRITE), "commit one page of two");
    if (half) {
        std::memset(half, 0x5a, 4096);
        expect(!read_guarded(out, half + 2048, 4096, How::copy), "a copy that runs into an uncommitted page faults");
        expect(out[0] == 0x5a && out[2047] == 0x5a, "the bytes before the fault were copied");
        VirtualFree(half, 0, MEM_RELEASE);
    }

    // memmove semantics, both overlap directions, every small size and a few offsets.
    std::vector<unsigned char> a(1024), b(1024);
    for (std::size_t size : {0u, 1u, 7u, 8u, 15u, 64u, 255u, 600u}) {
        for (std::ptrdiff_t shift : {-33, -8, -1, 1, 8, 33, 0}) {
            for (std::size_t i = 0; i < a.size(); ++i) a[i] = b[i] = static_cast<unsigned char>(i * 31 + size);
            const std::size_t from = 200;
            const std::size_t to = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(from) + shift);
            std::memmove(b.data() + to, b.data() + from, size);
            expect(read_guarded(a.data() + to, a.data() + from, size, How::move), "move of readable bytes succeeds");
            expect(a == b, "guarded::move matches memmove");
        }
    }
    unsigned char src[300], dst[300] = {};
    for (int i = 0; i < 300; ++i) src[i] = static_cast<unsigned char>(i ^ 0xa5);
    expect(read_guarded(dst, src, sizeof src, How::copy) && !std::memcmp(dst, src, sizeof src), "guarded::copy copies");
    const std::uintptr_t word = 0x1122334455667788ull;
    expect(leonida::guarded::read_word(&word) == word, "guarded::read_word reads the word");

    VirtualFree(reserved, 0, MEM_RELEASE);
    if (failures) {
        std::fprintf(stderr, "guarded_copy_test: %u of %u checks failed\n", failures, checks);
        return 1;
    }
    std::printf("guarded_copy_test: %u checks passed\n", checks);
    return 0;
}
