// XUserImpl through QueryApiImpl, called the way xgameruntime.lib's XUser* thunks call it: QueryApiImpl with the
// class id (or a later interface id), one vtable slot with `this` first. Slots follow xgameruntime-docs' layout;
// arguments and expected values follow the public GDK reference. No game data; nothing online.
//
//   xgameruntime_user_test                          the offline user, its handles, gamertag "Player"
//   xgameruntime_user_test --expect-gamertag TAG    the same, run with LEONIDA_GAMERTAG=TAG
#include <windows.h>

#include <io.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

#include "shims.h"

namespace rt::shims {
long xgameruntime_user_open_handles();
}

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

using XUserHandle = void*;
struct XUserLocalId {
    uint64_t value;
};

constexpr HRESULT E_GAMEUSER_USER_NOT_FOUND = static_cast<HRESULT>(0x89245104);
constexpr HRESULT E_GAMEUSER_DEFERRAL_NOT_AVAILABLE = static_cast<HRESULT>(0x89245103);
constexpr HRESULT E_GAMEUSER_FAILED_TO_GET_TOKEN = static_cast<HRESULT>(0x89245111);

constexpr GUID kUserClass = {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}};
constexpr GUID kUserImpl3 = {0x1bf2f8c5, 0xd507, 0x4e52, {0xbb, 0x05, 0xf7, 0x26, 0xd0, 0xe7, 0x11, 0x61}};
constexpr GUID kUserImpl6 = {0x5131d685, 0x4394, 0x4ee6, {0x8c, 0x18, 0xbf, 0xb5, 0xd4, 0xae, 0xf1, 0xff}};
constexpr GUID kGamertag = {0xcef4fac0, 0x7676, 0x4a94, {0xa1, 0x19, 0x4c, 0x43, 0xf9, 0xeb, 0x5b, 0x74}};
constexpr GUID kOther = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};  // XGameImpl

using Query = HRESULT(WINAPI*)(const GUID*, REFIID, void**);
using Fn = void*;
Fn slot(void* obj, int index) { return (*static_cast<Fn* const*>(obj))[index]; }
template <class F> F at(void* obj, int index) { return reinterpret_cast<F>(slot(obj, index)); }

// Waits for an XAsyncBlock on the process queue, as a title polling XAsyncGetStatus(wait = true) does.
bool finished(XAsyncBlock* async) { return XAsyncGetStatus(async, true) != E_PENDING; }

// Sends stderr, where the shim's "[rt] xgameruntime: XUser ... first call" lines go, to a temporary file until
// finish(), which returns what was written.
struct StderrCapture {
    char path[MAX_PATH] = {};
    int saved = -1;
    StderrCapture() {
        char dir[MAX_PATH] = {};
        GetTempPathA(sizeof(dir), dir);
        GetTempFileNameA(dir, "xur", 0, path);
        std::fflush(stderr);
        saved = _dup(2);
        std::FILE* out = nullptr;
        if (fopen_s(&out, path, "wb") == 0 && out) {
            if (saved >= 0) _dup2(_fileno(out), 2);
            std::fclose(out);
        }
    }
    std::string finish() {
        std::fflush(stderr);
        if (saved >= 0) {
            _dup2(saved, 2);
            _close(saved);
            saved = -1;
        }
        std::string text;
        std::FILE* in = nullptr;
        if (fopen_s(&in, path, "rb") == 0 && in) {
            char chunk[4096];
            for (size_t n; (n = std::fread(chunk, 1, sizeof(chunk), in)) > 0;) text.append(chunk, n);
            std::fclose(in);
        }
        DeleteFileA(path);
        return text;
    }
    ~StderrCapture() { finish(); }
};

int count_of(const std::string& text, const std::string& needle) {
    int count = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++count;
    return count;
}

// The line of `text` that contains `needle`, or "" when none does.
std::string line_with(const std::string& text, const std::string& needle) {
    const size_t at = text.find(needle);
    if (at == std::string::npos) return {};
    const size_t newline_before = text.rfind('\n', at);
    const size_t begin = newline_before == std::string::npos ? 0 : newline_before + 1;
    const size_t end = text.find('\n', at);
    return text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
}

}  // namespace

int main(int argc, char** argv) {
    const char* expect_tag = "Player";
    if (argc == 3 && std::strcmp(argv[1], "--expect-gamertag") == 0) expect_tag = argv[2];
    std::printf("xgameruntime_user_test: XUserImpl through QueryApiImpl (gamertag \"%s\")\n", expect_tag);

    auto query = reinterpret_cast<Query>(rt::shims::find("xgameruntime.dll", "QueryApiImpl"));
    check(query != nullptr, "QueryApiImpl is registered");
    if (!query) return 1;

    // ---- the object and its interfaces ----
    void* user = nullptr;
    check(query(&kUserClass, kUserClass, &user) == S_OK && user, "QueryApiImpl hands out XUserImpl for its class id");
    if (!user) return 1;
    void* user6 = nullptr;
    void* user3 = nullptr;
    check(query(&kUserClass, kUserImpl6, &user6) == S_OK && user6 == user &&
              query(&kUserClass, kUserImpl3, &user3) == S_OK && user3 == user,
          "IXUserImpl3 and IXUserImpl6 (later versions of the same vtable) answer with the same object");
    void* tag_face = nullptr;
    check(query(&kUserClass, kGamertag, &tag_face) == S_OK && tag_face && tag_face != user,
          "IXUserGamertagImpl is a second interface of the object");
    void* none = reinterpret_cast<void*>(1);
    check(query(&kUserClass, kOther, &none) == E_NOINTERFACE && none == nullptr,
          "an interface the class doesn't have is E_NOINTERFACE with a cleared out pointer");

    // Every call from here to the gamertag checks is the slot's first in this process: its first-call lines are read
    // back below.
    StderrCapture capture;

    // ---- XUserAddAsync / XUserAddResult (slots 7, 8) ----
    using AddAsync = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t, XAsyncBlock*);
    using AddResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*, XUserHandle*);
    XAsyncBlock add{};
    check(at<AddAsync>(user, 7)(user, 0x01, &add) == S_OK, "XUserAddAsync(AddDefaultUserSilently) starts");
    check(finished(&add) && XAsyncGetStatus(&add, false) == S_OK, "it completes with S_OK");
    XUserHandle h = nullptr;
    check(at<AddResult>(user, 8)(user, &add, &h) == S_OK && h != nullptr, "XUserAddResult hands out a user handle");
    XAsyncBlock bad{};
    check(at<AddAsync>(user, 7)(user, 0x01 | 0x02, &bad) == E_INVALIDARG,
          "AllowGuests with AddDefaultUserSilently is refused (a guest can't be the default user)");
    XAsyncBlock picker{};
    XUserHandle h_picker = nullptr;
    check(at<AddAsync>(user, 7)(user, 0x00, &picker) == S_OK && finished(&picker) &&
              at<AddResult>(user, 8)(user, &picker, &h_picker) == S_OK && h_picker,
          "XUserAddAsync without a silent option also adds the offline user (no account picker offline)");

    // ---- identity (slots 5, 6, 9-14, 19, 20) ----
    using Compare = int32_t(STDMETHODCALLTYPE*)(void*, XUserHandle, XUserHandle);
    using MaxUsers = HRESULT(STDMETHODCALLTYPE*)(void*, uint32_t*);
    using GetLocalId = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, XUserLocalId*);
    using FindLocal = HRESULT(STDMETHODCALLTYPE*)(void*, XUserLocalId, XUserHandle*);
    using GetId = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, uint64_t*);
    using FindId = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, XUserHandle*);
    using GetBool = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, bool*);
    using GetU32 = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, uint32_t*);
    using CheckPrivilege = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, uint32_t, uint32_t, bool*, uint32_t*);
    uint32_t max_users = 0;
    check(at<MaxUsers>(user, 6)(user, &max_users) == S_OK && max_users == 1, "XUserGetMaxUsers: one local user");
    check(at<Compare>(user, 5)(user, h, h_picker) == 0, "XUserCompare: both handles name the same user");
    XUserLocalId local{};
    check(at<GetLocalId>(user, 9)(user, h, &local) == S_OK && local.value != 0, "XUserGetLocalId: a non-zero local id");
    uint64_t xuid = 0;
    check(at<GetId>(user, 11)(user, h, &xuid) == S_OK && xuid != 0, "XUserGetId: a non-zero user id");
    XUserHandle by_local = nullptr, by_id = nullptr, missing = reinterpret_cast<XUserHandle>(1);
    check(at<FindLocal>(user, 10)(user, local, &by_local) == S_OK && by_local &&
              at<Compare>(user, 5)(user, by_local, h) == 0,
          "XUserFindUserByLocalId finds it");
    check(at<FindLocal>(user, 10)(user, XUserLocalId{local.value + 1}, &missing) == E_GAMEUSER_USER_NOT_FOUND &&
              missing == nullptr,
          "another local id is E_GAMEUSER_USER_NOT_FOUND");
    check(at<FindId>(user, 12)(user, xuid, &by_id) == S_OK && by_id, "XUserFindUserById finds it");
    missing = reinterpret_cast<XUserHandle>(1);
    check(at<FindId>(user, 12)(user, xuid + 1, &missing) == E_GAMEUSER_USER_NOT_FOUND && missing == nullptr,
          "another user id is E_GAMEUSER_USER_NOT_FOUND");
    bool guest = true;
    check(at<GetBool>(user, 13)(user, h, &guest) == S_OK && !guest, "XUserGetIsGuest: not a guest");
    uint32_t state = 99, age = 99;
    check(at<GetU32>(user, 14)(user, h, &state) == S_OK && state == 0, "XUserGetState: SignedIn (0)");
    check(at<GetU32>(user, 19)(user, h, &age) == S_OK && age == 3, "XUserGetAgeGroup: Adult (3)");
    bool has = false;
    uint32_t reason = 99;
    check(at<CheckPrivilege>(user, 20)(user, h, 0, 254, &has, &reason) == S_OK && has && reason == 0,
          "XUserCheckPrivilege: granted, deny reason None (0)");
    check(at<GetBool>(user, 13)(user, nullptr, &guest) == E_INVALIDARG, "a null handle is E_INVALIDARG");

    // ---- XUserResolvePrivilegeWithUiAsync / Result (21, 22) ----
    using ResolveAsync = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, uint32_t, uint32_t, XAsyncBlock*);
    using ResolveResult = HRESULT(STDMETHODCALLTYPE*)(void*, XAsyncBlock*);
    XAsyncBlock resolve{};
    check(at<ResolveAsync>(user, 21)(user, h, 0, 254, &resolve) == S_OK && finished(&resolve) &&
              at<ResolveResult>(user, 22)(user, &resolve) == S_OK,
          "XUserResolvePrivilegeWithUiAsync completes with S_OK (nothing to resolve)");

    // ---- tokens: not available offline (23) ----
    using TokenAsync = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, uint32_t, const char*, const char*, size_t,
                                                   const void*, size_t, const void*, XAsyncBlock*);
    XAsyncBlock token{};
    check(at<TokenAsync>(user, 23)(user, h, 0, "GET", "https://example.invalid/", 0, nullptr, 0, nullptr, &token) ==
              E_GAMEUSER_FAILED_TO_GET_TOKEN,
          "XUserGetTokenAndSignatureAsync fails before starting: E_GAMEUSER_FAILED_TO_GET_TOKEN");

    // ---- change events, sign-out deferral (33-36) ----
    using Register = HRESULT(STDMETHODCALLTYPE*)(void*, XTaskQueueHandle, void*, void(CALLBACK*)(void*, XUserLocalId, uint32_t),
                                                 XTaskQueueRegistrationToken*);
    using Unregister = bool(STDMETHODCALLTYPE*)(void*, XTaskQueueRegistrationToken, bool);
    using Deferral = HRESULT(STDMETHODCALLTYPE*)(void*, void**);
    auto on_change = [](void*, XUserLocalId, uint32_t) {};
    XTaskQueueRegistrationToken reg{};
    check(at<Register>(user, 33)(user, nullptr, nullptr, on_change, &reg) == S_OK && reg.token != 0,
          "XUserRegisterForChangeEvent hands out a token");
    check(at<Unregister>(user, 34)(user, reg, false) && at<Unregister>(user, 34)(user, reg, false) &&
              at<Unregister>(user, 34)(user, XTaskQueueRegistrationToken{999}, false),
          "XUserUnregisterForChangeEvent: true, also with wait = false for a token already gone or never issued "
          "(false would mean still pending, which never happens offline)");
    void* deferral = reinterpret_cast<void*>(1);
    check(at<Deferral>(user, 35)(user, &deferral) == E_GAMEUSER_DEFERRAL_NOT_AVAILABLE && deferral == nullptr,
          "XUserGetSignOutDeferral: E_GAMEUSER_DEFERRAL_NOT_AVAILABLE (nothing signs out offline)");

    // ---- IXUserImpl2: XUserAddByIdWithUiAsync / Result (37, 38) ----
    using AddByIdAsync = HRESULT(STDMETHODCALLTYPE*)(void*, uint64_t, XAsyncBlock*);
    XAsyncBlock by_id_add{}, by_other{};
    XUserHandle h_by_id = nullptr, h_other = reinterpret_cast<XUserHandle>(1);
    check(at<AddByIdAsync>(user, 37)(user, xuid, &by_id_add) == S_OK && finished(&by_id_add) &&
              at<AddResult>(user, 38)(user, &by_id_add, &h_by_id) == S_OK && h_by_id,
          "XUserAddByIdWithUiAsync with the local user's id adds it");
    check(at<AddByIdAsync>(user, 37)(user, xuid + 1, &by_other) == S_OK && finished(&by_other) &&
              at<AddResult>(user, 38)(user, &by_other, &h_other) == E_GAMEUSER_USER_NOT_FOUND && h_other == nullptr,
          "with another id it completes with E_GAMEUSER_USER_NOT_FOUND");

    // ---- IXUserImpl4: XUserIsStoreUser (42) ----
    using IsStore = bool(STDMETHODCALLTYPE*)(void*, XUserHandle);
    check(at<IsStore>(user, 42)(user, h) && at<IsStore>(user, 42)(user, nullptr),
          "XUserIsStoreUser: true, also for a null handle (_In_opt_; always true on console)");
    using Unknown = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, size_t, char*, size_t*);
    char probe[16] = "untouched";
    size_t probe_used = 42;
    check(at<Unknown>(user, 15)(user, h, sizeof(probe), probe, &probe_used) == E_NOTIMPL,
          "slot 15 (no public signature) answers E_NOTIMPL rather than S_OK with nothing written");

    // ---- IXUserGamertagImpl: XUserGetGamertag (3) ----
    using GetTag = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, uint32_t, size_t, char*, size_t*);
    auto tag = at<GetTag>(tag_face, 3);
    char buf[101] = {};
    size_t used = 0;
    const size_t want = std::strlen(expect_tag) + 1;
    check(tag(tag_face, h, 0, sizeof(buf), buf, &used) == S_OK && std::strcmp(buf, expect_tag) == 0 && used == want,
          "XUserGetGamertag(Classic): the gamertag, gamertagUsed counting the terminator");
    check(tag(tag_face, h, 3, sizeof(buf), buf, &used) == S_OK && std::strcmp(buf, expect_tag) == 0,
          "UniqueModern of a classic gamertag: the same text, no '#' and no suffix");
    check(tag(tag_face, h, 1, sizeof(buf), buf, &used) == S_OK && std::strcmp(buf, expect_tag) == 0,
          "Modern: the classic gamertag when none is set");
    check(tag(tag_face, h, 2, sizeof(buf), buf, &used) == S_OK && buf[0] == 0 && used == 1,
          "ModernSuffix: empty (just the terminator)");
    check(tag(tag_face, h, 0, want - 1, buf, &used) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER),
          "a buffer one byte short is E_INSUFFICIENT_BUFFER");
    check(tag(tag_face, h, 7, sizeof(buf), buf, &used) == E_INVALIDARG, "an unknown component is E_INVALIDARG");

    // ---- first-call lines: one per slot (per gamertag component), whatever the number of calls ----
    const std::string log = capture.finish();
    std::fputs(log.c_str(), stdout);
    check(count_of(log, "[rt] xgameruntime: XUser AddAsync first call: options 0x1,") == 1 &&
              count_of(log, "XUser AddAsync first call") == 1,
          "XUserAddAsync logs its first call (options 0x1) once, not for the two calls after it");
    check(count_of(log, "[rt] xgameruntime: XUser AddResult first call:") == 1 &&
              count_of(log, "[rt] xgameruntime: XUser CheckPrivilege first call:") == 1 &&
              count_of(log, "[rt] xgameruntime: XUser GetState first call:") == 1 &&
              count_of(log, "[rt] xgameruntime: XUser RegisterForChangeEvent first call:") == 1,
          "AddResult, CheckPrivilege, GetState and RegisterForChangeEvent log their first call once");
    const std::string classic = line_with(log, "[rt] xgameruntime: XUser GetGamertag first call: Classic (0)");
    check(count_of(log, "XUser GetGamertag first call: Classic (0)") == 1 &&
              classic.find(std::string("-> 0x00000000, \"") + expect_tag + "\"") != std::string::npos,
          "XUserGetGamertag logs the first Classic call once, with S_OK and the gamertag (not the short-buffer call)");
    check(count_of(log, "[rt] xgameruntime: XUser GetGamertag first call:") == 5 &&
              line_with(log, "GetGamertag first call: unknown component (7)").find("0x80070057") != std::string::npos,
          "one GetGamertag line per component asked for (four, and the unknown one with E_INVALIDARG)");

    // ---- handles: duplicate, close, all accounted for (3, 4) ----
    using Dup = HRESULT(STDMETHODCALLTYPE*)(void*, XUserHandle, XUserHandle*);
    using Close = void(STDMETHODCALLTYPE*)(void*, XUserHandle);
    XUserHandle dup = nullptr;
    check(at<Dup>(user, 3)(user, h, &dup) == S_OK && dup && dup != h && at<Compare>(user, 5)(user, dup, h) == 0,
          "XUserDuplicateHandle: a new handle for the same user");
    check(rt::shims::xgameruntime_user_open_handles() == 6, "six handles are open: two adds, two finds, an add by id and a duplicate");
    for (XUserHandle x : {h, h_picker, by_local, by_id, h_by_id, dup}) at<Close>(user, 4)(user, x);
    check(rt::shims::xgameruntime_user_open_handles() == 0, "XUserCloseHandle closes each of them");
    at<Close>(user, 4)(user, h);   // a double close: the handle is looked up, not read, so this is a no-op
    check(rt::shims::xgameruntime_user_open_handles() == 0 && at<GetBool>(user, 13)(user, h, &guest) == E_INVALIDARG,
          "a closed handle is refused and closing it again does nothing");

    std::printf("xgameruntime_user_test: %d of %d checks failed\n", g_failures, g_checks);
    return g_failures ? 1 : 0;
}
