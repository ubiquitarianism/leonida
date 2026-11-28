// xgameruntime: XUserImpl, the class xgameruntime.lib's XUser* functions reach through QueryApiImpl, offline.
//
// A GDK title links XUserAddAsync, XUserGetId, XUserGetGamertag and the rest statically (xgameruntime.lib); each of
// those thunks asks QueryApiImpl for XUserImpl and calls one slot with the function's own arguments after `this`
// (docs/runtime_imports.md, "Run-time loaded modules"; xgameruntime.cpp serves the other classes the same way).
// The class and interface ids and the slot order are xodus-gaming/xgameruntime-docs' "Layout" lists (GPL-3.0; facts
// only, nothing copied). Signatures, enum values and error codes follow Microsoft's public GDK reference
// (learn.microsoft.com/gaming/gdk, XUser and Error Codes pages). No GDK header or library code is used.
//
// Offline there is one local user, always signed in: the default user XUserAddAsync returns (with or without
// AddDefaultUserSilently), with a fixed Xbox user id and local id, and a gamertag from LEONIDA_GAMERTAG (15 ASCII
// characters at most, "Player" otherwise). Nothing goes online: token and gamer-picture requests fail as they would
// without a network, and the user never changes, so registered change callbacks are never called.
//
// The executable links thunks for IXUserImpl slots 3-5, 7-14, 19-25 and 33-34 and IXUserGamertagImpl slot 3
// (tools/gdk_runtime_calls.py, from the file alone; docs/runtime_imports.md). None of them is a private or PC-only
// slot; those still log once when called.
//
// Every served slot writes one "[rt] xgameruntime: XUser <name> first call: ..." line with its arguments and answer
// (XUserGetGamertag one per component), so a boot log shows how far the title got with its user: the player-select
// prompt needs a signed-in user and its gamertag (docs/re/notes/firstframe.md). Later calls are silent.
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <set>
#include <unordered_set>

#include <XAsync.h>
#include <XAsyncProvider.h>
#include <XTaskQueue.h>

namespace {

// ---- Public GDK types (XUser.h, as the GDK reference documents them) ----

using XUserHandle = struct XUserLocal*;              // opaque to the title
using XUserSignOutDeferralHandle = struct XUserSignOutDeferral*;
struct XUserLocalId {
    uint64_t value;
};
enum class XUserAddOptions : uint32_t { None = 0x00, AddDefaultUserSilently = 0x01, AllowGuests = 0x02, AddDefaultUserAllowingUI = 0x04 };
enum class XUserState : uint32_t { SignedIn = 0, SigningOut = 1, SignedOut = 2 };
enum class XUserAgeGroup : uint32_t { Unknown = 0, Child = 1, Teen = 2, Adult = 3 };
enum class XUserGamertagComponent : uint32_t { Classic = 0, Modern = 1, ModernSuffix = 2, UniqueModern = 3 };
enum class XUserPrivilegeDenyReason : uint32_t { None = 0 };
enum class XUserChangeEvent : uint32_t { SignedInAgain = 0 };
using XUserChangeEventCallback = void(CALLBACK*)(void* context, XUserLocalId user, XUserChangeEvent event);

// XGameErr.h values, from the GDK "Error codes" page.
constexpr HRESULT E_GAMEUSER_USER_NOT_FOUND = static_cast<HRESULT>(0x89245104);
constexpr HRESULT E_GAMEUSER_DEFERRAL_NOT_AVAILABLE = static_cast<HRESULT>(0x89245103);
constexpr HRESULT E_GAMEUSER_FAILED_TO_RESOLVE = static_cast<HRESULT>(0x89245107);
constexpr HRESULT E_GAMEUSER_FAILED_TO_GET_TOKEN = static_cast<HRESULT>(0x89245111);
constexpr HRESULT kInsufficientBuffer = HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);

// ---- The offline user ----

// Fixed, non-zero ids. The Xbox user id has the shape of a real XUID (a 64-bit number); nothing checks it offline.
constexpr uint64_t kOfflineXuid = 0x0009000000000001ull;
constexpr uint64_t kOfflineLocalId = 1;

const char* gamertag() {
    static char tag[16] = "Player";
    static std::once_flag once;
    std::call_once(once, [] {
        size_t length = 0;   // with the terminator; 0 when unset
        if (getenv_s(&length, nullptr, 0, "LEONIDA_GAMERTAG") != 0 || length <= 1) return;
        if (length - 1 > 15) {
            std::fprintf(stderr, "[rt] xgameruntime: LEONIDA_GAMERTAG is longer than 15 characters; using \"Player\"\n");
            return;
        }
        char value[16] = {};
        if (getenv_s(&length, value, "LEONIDA_GAMERTAG") != 0) return;
        for (size_t i = 0; i + 1 < length; ++i)
            if (static_cast<unsigned char>(value[i]) < 0x20 || static_cast<unsigned char>(value[i]) > 0x7e) {
                std::fprintf(stderr, "[rt] xgameruntime: LEONIDA_GAMERTAG is not printable ASCII; using \"Player\"\n");
                return;
            }
        std::memcpy(tag, value, length);
    });
    return tag;
}

// A title must close every XUserHandle it gets exactly once, so each one is its own allocation; they all name the
// one local user. A handle is valid while it is in the live set: it's looked up, never read, so a closed or foreign
// pointer is refused without touching its memory. The block is freed on close, so the heap may hand its address out
// again; a stale handle then names a newer live one, which is harmless here since every handle names the same user.
struct XUserLocal {
    uint32_t unused;
};
std::mutex g_handles_mutex;
std::unordered_set<const void*> g_handles;

XUserHandle new_handle() {
    auto* h = new XUserLocal{};
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    g_handles.insert(h);
    return h;
}
bool valid(XUserHandle h) {
    if (!h) return false;
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return g_handles.count(h) != 0;
}

// The slot's one first-call line (see the top of the file); the format string is a literal.
#define USER_FIRST_CALL(...)                                                                             \
    do {                                                                                                 \
        static std::atomic<bool> s_logged{false};                                                        \
        if (!s_logged.exchange(true)) std::fprintf(stderr, "[rt] xgameruntime: XUser " __VA_ARGS__);     \
    } while (0)
unsigned long hex(HRESULT hr) { return static_cast<unsigned long>(hr); }

// ---- IXUserImpl (slots 3-36) ----

// 3: HRESULT XUserDuplicateHandle(XUserHandle handle, XUserHandle* duplicatedHandle)
HRESULT STDMETHODCALLTYPE user_duplicate(void*, XUserHandle handle, XUserHandle* dup) {
    const HRESULT hr = [&]() -> HRESULT {
        if (!dup) return E_POINTER;
        *dup = nullptr;
        if (!valid(handle)) return E_INVALIDARG;
        *dup = new_handle();
        return S_OK;
    }();
    USER_FIRST_CALL("DuplicateHandle first call: handle %p -> 0x%08lx, new handle %p\n", static_cast<void*>(handle),
                    hex(hr), dup ? static_cast<void*>(*dup) : nullptr);
    return hr;
}

// 4: void XUserCloseHandle(XUserHandle user)
void STDMETHODCALLTYPE user_close(void*, XUserHandle user) {
    bool ours = false;
    {
        std::lock_guard<std::mutex> lock(g_handles_mutex);
        ours = user && g_handles.erase(user) != 0;   // false: closed already, or not ours
    }
    USER_FIRST_CALL("CloseHandle first call: handle %p (%s)\n", static_cast<void*>(user),
                    ours ? "closed" : "not open; ignored");
    if (ours) delete user;
}

// 5: int32_t XUserCompare(XUserHandle user1, XUserHandle user2): 0 when both name the same user. Every valid handle
// names the one local user.
int32_t STDMETHODCALLTYPE user_compare(void*, XUserHandle a, XUserHandle b) {
    const bool va = valid(a), vb = valid(b);
    const int32_t order = va && vb ? 0 : va != vb ? (va ? 1 : -1) : a == b ? 0 : (a < b ? -1 : 1);
    USER_FIRST_CALL("Compare first call: handles %p, %p -> %d\n", static_cast<void*>(a), static_cast<void*>(b), order);
    return order;
}

// 6: HRESULT XUserGetMaxUsers(uint32_t* maxUsers). One local user offline.
HRESULT STDMETHODCALLTYPE user_get_max_users(void*, uint32_t* max_users) {
    const HRESULT hr = max_users ? S_OK : E_POINTER;
    if (max_users) *max_users = 1;
    USER_FIRST_CALL("GetMaxUsers first call: -> 0x%08lx, 1 user\n", hex(hr));
    return hr;
}

// XAsync providers. The result of an add is a new handle, created when the title collects it, so a result the
// title never collects leaks nothing.
const char kAddIdentity = 0;
const char kAddByIdIdentity = 0;
const char kResolveIdentity = 0;

HRESULT CALLBACK add_provider(XAsyncOp op, const XAsyncProviderData* data) {
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork:
        XAsyncComplete(data->async, S_OK, sizeof(XUserHandle));
        return S_OK;
    case XAsyncOp::GetResult:
        if (data->bufferSize < sizeof(XUserHandle) || !data->buffer) return E_INVALIDARG;
        *static_cast<XUserHandle*>(data->buffer) = new_handle();
        return S_OK;
    default:
        return S_OK;
    }
}

// The add-by-id flavour: the context is the requested id; an id other than the local user's completes with
// E_GAMEUSER_USER_NOT_FOUND (no other user can sign in offline).
HRESULT CALLBACK add_by_id_provider(XAsyncOp op, const XAsyncProviderData* data) {
    switch (op) {
    case XAsyncOp::Begin:
        return XAsyncSchedule(data->async, 0);
    case XAsyncOp::DoWork: {
        const bool known = reinterpret_cast<uintptr_t>(data->context) == static_cast<uintptr_t>(kOfflineXuid);
        XAsyncComplete(data->async, known ? S_OK : E_GAMEUSER_USER_NOT_FOUND, known ? sizeof(XUserHandle) : 0);
        return S_OK;
    }
    case XAsyncOp::GetResult:
        if (data->bufferSize < sizeof(XUserHandle) || !data->buffer) return E_INVALIDARG;
        *static_cast<XUserHandle*>(data->buffer) = new_handle();
        return S_OK;
    default:
        return S_OK;
    }
}

// Resolving a privilege: every privilege is granted offline, so there is nothing to show; completes with S_OK.
HRESULT CALLBACK resolve_provider(XAsyncOp op, const XAsyncProviderData* data) {
    if (op == XAsyncOp::Begin) return XAsyncSchedule(data->async, 0);
    if (op == XAsyncOp::DoWork) XAsyncComplete(data->async, S_OK, 0);
    return S_OK;
}

// 7: HRESULT XUserAddAsync(XUserAddOptions options, XAsyncBlock* async). AllowGuests with AddDefaultUserSilently is
// refused, as documented ("A guest cannot be the default user"). Without a silent option the console would show the
// account picker; offline the one user is the answer either way.
HRESULT STDMETHODCALLTYPE user_add_async(void*, XUserAddOptions options, XAsyncBlock* async) {
    const auto bits = static_cast<uint32_t>(options);
    const HRESULT hr = [&]() -> HRESULT {
        if (!async) return E_POINTER;
        if ((bits & static_cast<uint32_t>(XUserAddOptions::AddDefaultUserSilently)) &&
            (bits & static_cast<uint32_t>(XUserAddOptions::AllowGuests)))
            return E_INVALIDARG;
        return XAsyncBegin(async, nullptr, &kAddIdentity, "XUserAddAsync", add_provider);
    }();
    USER_FIRST_CALL("AddAsync first call: options 0x%x, block %p -> 0x%08lx (the offline user \"%s\")\n", bits,
                    static_cast<void*>(async), hex(hr), gamertag());
    return hr;
}

// 8: HRESULT XUserAddResult(XAsyncBlock* async, XUserHandle* newUser)
HRESULT STDMETHODCALLTYPE user_add_result(void*, XAsyncBlock* async, XUserHandle* user) {
    const HRESULT hr = [&]() -> HRESULT {
        if (!user) return E_POINTER;
        *user = nullptr;
        return XAsyncGetResult(async, &kAddIdentity, sizeof(*user), user, nullptr);
    }();
    USER_FIRST_CALL("AddResult first call: block %p -> 0x%08lx, handle %p\n", static_cast<void*>(async), hex(hr),
                    user ? static_cast<void*>(*user) : nullptr);
    return hr;
}

// 9: HRESULT XUserGetLocalId(XUserHandle user, XUserLocalId* userLocalId)
HRESULT STDMETHODCALLTYPE user_get_local_id(void*, XUserHandle user, XUserLocalId* id) {
    const HRESULT hr = !id ? E_POINTER : !valid(user) ? E_INVALIDARG : S_OK;
    if (hr == S_OK) id->value = kOfflineLocalId;
    USER_FIRST_CALL("GetLocalId first call: handle %p -> 0x%08lx, local id %llu\n", static_cast<void*>(user), hex(hr),
                    static_cast<unsigned long long>(hr == S_OK ? kOfflineLocalId : 0));
    return hr;
}

// 10: HRESULT XUserFindUserByLocalId(XUserLocalId userLocalId, XUserHandle* handle)
HRESULT STDMETHODCALLTYPE user_find_by_local_id(void*, XUserLocalId id, XUserHandle* handle) {
    const HRESULT hr = [&]() -> HRESULT {
        if (!handle) return E_POINTER;
        *handle = nullptr;
        if (id.value != kOfflineLocalId) return E_GAMEUSER_USER_NOT_FOUND;
        *handle = new_handle();
        return S_OK;
    }();
    USER_FIRST_CALL("FindUserByLocalId first call: local id %llu -> 0x%08lx, handle %p\n",
                    static_cast<unsigned long long>(id.value), hex(hr), handle ? static_cast<void*>(*handle) : nullptr);
    return hr;
}

// 11: HRESULT XUserGetId(XUserHandle user, uint64_t* userId)
HRESULT STDMETHODCALLTYPE user_get_id(void*, XUserHandle user, uint64_t* id) {
    const HRESULT hr = !id ? E_POINTER : !valid(user) ? E_INVALIDARG : S_OK;
    if (hr == S_OK) *id = kOfflineXuid;
    USER_FIRST_CALL("GetId first call: handle %p -> 0x%08lx, user id 0x%016llx\n", static_cast<void*>(user), hex(hr),
                    static_cast<unsigned long long>(hr == S_OK ? kOfflineXuid : 0));
    return hr;
}

// 12: HRESULT XUserFindUserById(uint64_t userId, XUserHandle* handle)
HRESULT STDMETHODCALLTYPE user_find_by_id(void*, uint64_t id, XUserHandle* handle) {
    const HRESULT hr = [&]() -> HRESULT {
        if (!handle) return E_POINTER;
        *handle = nullptr;
        if (id != kOfflineXuid) return E_GAMEUSER_USER_NOT_FOUND;
        *handle = new_handle();
        return S_OK;
    }();
    USER_FIRST_CALL("FindUserById first call: user id 0x%016llx -> 0x%08lx, handle %p\n",
                    static_cast<unsigned long long>(id), hex(hr), handle ? static_cast<void*>(*handle) : nullptr);
    return hr;
}

// 13: HRESULT XUserGetIsGuest(XUserHandle user, bool* isGuest)
HRESULT STDMETHODCALLTYPE user_get_is_guest(void*, XUserHandle user, bool* guest) {
    const HRESULT hr = !guest ? E_POINTER : !valid(user) ? E_INVALIDARG : S_OK;
    if (hr == S_OK) *guest = false;
    USER_FIRST_CALL("GetIsGuest first call: handle %p -> 0x%08lx, not a guest\n", static_cast<void*>(user), hex(hr));
    return hr;
}

// 14: HRESULT XUserGetState(XUserHandle user, XUserState* state)
HRESULT STDMETHODCALLTYPE user_get_state(void*, XUserHandle user, XUserState* state) {
    const HRESULT hr = !state ? E_POINTER : !valid(user) ? E_INVALIDARG : S_OK;
    if (hr == S_OK) *state = XUserState::SignedIn;
    USER_FIRST_CALL("GetState first call: handle %p -> 0x%08lx, SignedIn (0)\n", static_cast<void*>(user), hex(hr));
    return hr;
}

// A slot with no public signature, logged once. Slot 15 (the docs' "unknown") answers E_NOTIMPL, so a caller fails
// instead of reading outputs nothing wrote. Slot 47, IXUserImpl6's undocumented XUserIsSignOutPresent, is read as a
// bool, where E_NOTIMPL's low byte (1) would mean true, so it answers 0 (false: nothing signs out offline).
template <int Slot> HRESULT STDMETHODCALLTYPE unknown_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XUserImpl slot %d has no public signature; answering E_NOTIMPL\n", Slot);
    return E_NOTIMPL;
}
template <int Slot> uint64_t STDMETHODCALLTYPE private_slot(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XUserImpl private slot %d called; answering 0\n", Slot);
    return 0;
}

// An operation that needs the network or a system UI the offline runtime doesn't have. The *Async function fails
// before starting, so the XAsyncBlock is never used and no callback runs; its Result/ResultSize partners return the
// same error. Their arguments don't matter, so the functions take none (x64: the caller owns the argument area).
template <int Slot, HRESULT Error> HRESULT STDMETHODCALLTYPE offline_failure(void*) {
    static std::atomic<bool> s_logged{false};
    if (!s_logged.exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XUserImpl slot %d is not available offline (HRESULT 0x%08lx)\n", Slot,
                     static_cast<unsigned long>(Error));
    return Error;
}

// 19: HRESULT XUserGetAgeGroup(XUserHandle user, XUserAgeGroup* ageGroup)
HRESULT STDMETHODCALLTYPE user_get_age_group(void*, XUserHandle user, XUserAgeGroup* group) {
    const HRESULT hr = !group ? E_POINTER : !valid(user) ? E_INVALIDARG : S_OK;
    if (hr == S_OK) *group = XUserAgeGroup::Adult;
    USER_FIRST_CALL("GetAgeGroup first call: handle %p -> 0x%08lx, Adult (3)\n", static_cast<void*>(user), hex(hr));
    return hr;
}

// 20: HRESULT XUserCheckPrivilege(XUserHandle user, XUserPrivilegeOptions options, XUserPrivilege privilege,
//                                 bool* hasPrivilege, XUserPrivilegeDenyReason* reason). Everything is allowed offline
// (multiplayer and the like then fail at the network, as on a console without one).
HRESULT STDMETHODCALLTYPE user_check_privilege(void*, XUserHandle user, uint32_t options, uint32_t privilege, bool* has,
                                               XUserPrivilegeDenyReason* reason) {
    const HRESULT hr = !has ? E_POINTER : !valid(user) ? E_INVALIDARG : S_OK;
    if (hr == S_OK) {
        *has = true;
        if (reason) *reason = XUserPrivilegeDenyReason::None;
    }
    USER_FIRST_CALL("CheckPrivilege first call: handle %p, options %u, privilege %u -> 0x%08lx, granted\n",
                    static_cast<void*>(user), options, privilege, hex(hr));
    return hr;
}

// 21: HRESULT XUserResolvePrivilegeWithUiAsync(XUserHandle user, XUserPrivilegeOptions options, XUserPrivilege privilege,
//                                              XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE user_resolve_privilege_async(void*, XUserHandle user, uint32_t, uint32_t privilege,
                                                       XAsyncBlock* async) {
    const HRESULT hr = !async        ? E_POINTER
                       : !valid(user) ? E_INVALIDARG
                                      : XAsyncBegin(async, nullptr, &kResolveIdentity, "XUserResolvePrivilegeWithUiAsync",
                                                    resolve_provider);
    USER_FIRST_CALL("ResolvePrivilegeWithUiAsync first call: handle %p, privilege %u -> 0x%08lx (nothing to resolve)\n",
                    static_cast<void*>(user), privilege, hex(hr));
    return hr;
}

// 22: HRESULT XUserResolvePrivilegeWithUiResult(XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE user_resolve_privilege_result(void*, XAsyncBlock* async) {
    const HRESULT hr = XAsyncGetResult(async, &kResolveIdentity, 0, nullptr, nullptr);
    USER_FIRST_CALL("ResolvePrivilegeWithUiResult first call: block %p -> 0x%08lx\n", static_cast<void*>(async), hex(hr));
    return hr;
}

// 33: HRESULT XUserRegisterForChangeEvent(XTaskQueueHandle queue, void* context, XUserChangeEventCallback* callback,
//                                         XTaskQueueRegistrationToken* token)
// 34: bool XUserUnregisterForChangeEvent(XTaskQueueRegistrationToken token, bool wait)
// The user never changes offline, so the callbacks are kept only as tokens.
std::mutex g_tokens_mutex;
std::set<uint64_t> g_tokens;
uint64_t g_next_token = 1;

HRESULT STDMETHODCALLTYPE user_register_change(void*, XTaskQueueHandle, void*, XUserChangeEventCallback callback,
                                               XTaskQueueRegistrationToken* token) {
    const HRESULT hr = callback && token ? S_OK : E_POINTER;
    if (hr == S_OK) {
        std::lock_guard<std::mutex> lock(g_tokens_mutex);
        token->token = g_next_token++;
        g_tokens.insert(token->token);
    }
    USER_FIRST_CALL("RegisterForChangeEvent first call: -> 0x%08lx, token %llu (never called: the user never changes)\n",
                    hex(hr), static_cast<unsigned long long>(hr == S_OK ? token->token : 0));
    return hr;
}

// false means "wait was false and the unregistration is still pending"; nothing is ever pending offline, so it is
// true for every token, including one already unregistered or never handed out (a title polling with wait = false
// would otherwise spin).
bool STDMETHODCALLTYPE user_unregister_change(void*, XTaskQueueRegistrationToken token, bool) {
    {
        std::lock_guard<std::mutex> lock(g_tokens_mutex);
        g_tokens.erase(token.token);
    }
    USER_FIRST_CALL("UnregisterForChangeEvent first call: token %llu -> true\n",
                    static_cast<unsigned long long>(token.token));
    return true;
}

// 35: HRESULT XUserGetSignOutDeferral(XUserSignOutDeferralHandle* deferral): nothing signs out offline, so there is
// no sign-out to defer (the documented E_GAMEUSER_DEFERRAL_NOT_AVAILABLE).
HRESULT STDMETHODCALLTYPE user_get_signout_deferral(void*, XUserSignOutDeferralHandle* deferral) {
    if (deferral) *deferral = nullptr;
    USER_FIRST_CALL("GetSignOutDeferral first call: -> 0x%08lx (nothing signs out offline)\n",
                    hex(E_GAMEUSER_DEFERRAL_NOT_AVAILABLE));
    return E_GAMEUSER_DEFERRAL_NOT_AVAILABLE;
}

// 36: void XUserCloseSignOutDeferralHandle(XUserSignOutDeferralHandle deferral)
void STDMETHODCALLTYPE user_close_signout_deferral(void*, XUserSignOutDeferralHandle) {}

// ---- IXUserImpl2 (37-38) ----

// 37: HRESULT XUserAddByIdWithUiAsync(uint64_t userId, XAsyncBlock* async)
HRESULT STDMETHODCALLTYPE user_add_by_id_async(void*, uint64_t id, XAsyncBlock* async) {
    const HRESULT hr = !async ? E_POINTER
                              : XAsyncBegin(async, reinterpret_cast<void*>(static_cast<uintptr_t>(id)),
                                            &kAddByIdIdentity, "XUserAddByIdWithUiAsync", add_by_id_provider);
    USER_FIRST_CALL("AddByIdWithUiAsync first call: user id 0x%016llx -> 0x%08lx\n", static_cast<unsigned long long>(id),
                    hex(hr));
    return hr;
}

// 38: HRESULT XUserAddByIdWithUiResult(XAsyncBlock* async, XUserHandle* newUser)
HRESULT STDMETHODCALLTYPE user_add_by_id_result(void*, XAsyncBlock* async, XUserHandle* user) {
    const HRESULT hr = [&]() -> HRESULT {
        if (!user) return E_POINTER;
        *user = nullptr;
        return XAsyncGetResult(async, &kAddByIdIdentity, sizeof(*user), user, nullptr);
    }();
    USER_FIRST_CALL("AddByIdWithUiResult first call: block %p -> 0x%08lx, handle %p\n", static_cast<void*>(async),
                    hex(hr), user ? static_cast<void*>(*user) : nullptr);
    return hr;
}

// ---- IXUserImpl4 (42) ----

// 42: bool XUserIsStoreUser(XUserHandle user): the handle is _In_opt_, and the GDK reference says the function
// always returns true on console.
bool STDMETHODCALLTYPE user_is_store_user(void*, XUserHandle user) {
    USER_FIRST_CALL("IsStoreUser first call: handle %p -> true\n", static_cast<void*>(user));
    return true;
}

// ---- IXUserGamertagImpl (its slot 3) ----

// HRESULT XUserGetGamertag(XUserHandle user, XUserGamertagComponent component, size_t gamertagSize, char* gamertag,
//                          size_t* gamertagUsed). The offline user has a classic gamertag: Modern and UniqueModern are
// the same text (a classic tag has no '#' and no suffix) and ModernSuffix is empty. gamertagUsed counts the
// terminator. This is what the title's player-select prompt shows as "<gamertag> (A)"; the first call for each
// component is logged with its answer.
HRESULT STDMETHODCALLTYPE user_get_gamertag(void*, XUserHandle user, XUserGamertagComponent component, size_t size,
                                            char* out, size_t* used) {
    if (used) *used = 0;
    const char* text = nullptr;
    switch (component) {
    case XUserGamertagComponent::Classic:
    case XUserGamertagComponent::Modern:
    case XUserGamertagComponent::UniqueModern:
        text = gamertag();
        break;
    case XUserGamertagComponent::ModernSuffix:
        text = "";
        break;
    default:
        break;
    }
    const size_t need = text ? std::strlen(text) + 1 : 0;
    const HRESULT hr = !valid(user) || !text ? E_INVALIDARG : !out || size < need ? kInsufficientBuffer : S_OK;
    if (hr == S_OK) {
        std::memcpy(out, text, need);
        if (used) *used = need;
    }
    static const char* const kNames[] = {"Classic", "Modern", "ModernSuffix", "UniqueModern", "unknown component"};
    static std::atomic<bool> s_logged[std::size(kNames)];
    const uint32_t c = static_cast<uint32_t>(component);
    const size_t index = c < 4 ? c : 4;
    if (!s_logged[index].exchange(true))
        std::fprintf(stderr, "[rt] xgameruntime: XUser GetGamertag first call: %s (%u), handle %p, buffer %zu bytes -> "
                             "0x%08lx, \"%s\" (%zu bytes used)\n",
                     kNames[index], c, static_cast<void*>(user), size, hex(hr), hr == S_OK ? text : "",
                     hr == S_OK ? need : 0);
    return hr;
}

// ---- The object: one vtable for the IXUserImpl..IXUserImpl6 chain, one for IXUserGamertagImpl ----

constexpr GUID kUserClass = {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}};
// IXUserImpl's id is the class id; IXUserImpl2..6 extend it, so each of them answers with the same vtable.
constexpr GUID kUserInterfaces[] = {
    kUserClass,
    {0xeb9bf948, 0x18dc, 0x4d82, {0xbb, 0xcc, 0x40, 0xe0, 0xa8, 0x09, 0xc4, 0xc0}},   // IXUserImpl2
    {0x1bf2f8c5, 0xd507, 0x4e52, {0xbb, 0x05, 0xf7, 0x26, 0xd0, 0xe7, 0x11, 0x61}},   // IXUserImpl3
    {0x079415e3, 0x6727, 0x437f, {0x8e, 0x9d, 0x8f, 0x8f, 0x9b, 0x24, 0x39, 0xf7}},   // IXUserImpl4
    {0x26f3c674, 0xa2fe, 0x44fa, {0xb6, 0xc4, 0xa3, 0x23, 0xbc, 0x94, 0xff, 0x53}},   // IXUserImpl5
    {0x5131d685, 0x4394, 0x4ee6, {0x8c, 0x18, 0xbf, 0xb5, 0xd4, 0xae, 0xf1, 0xff}},   // IXUserImpl6
};
constexpr GUID kGamertagInterface = {0xcef4fac0, 0x7676, 0x4a94, {0xa1, 0x19, 0x4c, 0x43, 0xf9, 0xeb, 0x5b, 0x74}};
constexpr GUID kIUnknown = {0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};

struct UserObject {
    const void* const* vtable;            // IXUserImpl6
    const void* const* gamertag_vtable;   // IXUserGamertagImpl
};
UserObject* user_object();

HRESULT query(REFIID iid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    UserObject* o = user_object();
    if (iid == kIUnknown) {
        *out = &o->vtable;
        return S_OK;
    }
    for (const GUID& id : kUserInterfaces)
        if (iid == id) {
            *out = &o->vtable;
            return S_OK;
        }
    if (iid == kGamertagInterface) {
        *out = &o->gamertag_vtable;
        return S_OK;
    }
    return E_NOINTERFACE;
}

// A process-wide singleton, as the other runtime classes: AddRef and Release don't free.
HRESULT STDMETHODCALLTYPE object_query_interface(void*, REFIID iid, void** out) { return query(iid, out); }
#undef USER_FIRST_CALL
ULONG STDMETHODCALLTYPE object_add_ref(void*) { return 2; }
ULONG STDMETHODCALLTYPE object_release(void*) { return 1; }

#define FN(f) reinterpret_cast<const void*>(&f)

const void* const kUserVtable[] = {
    FN(object_query_interface), FN(object_add_ref), FN(object_release),          // 0-2  IUnknown
    FN(user_duplicate),                                                         // 3  XUserDuplicateHandle
    FN(user_close),                                                             // 4  XUserCloseHandle
    FN(user_compare),                                                           // 5  XUserCompare
    FN(user_get_max_users),                                                     // 6  XUserGetMaxUsers
    FN(user_add_async),                                                         // 7  XUserAddAsync
    FN(user_add_result),                                                        // 8  XUserAddResult
    FN(user_get_local_id),                                                      // 9  XUserGetLocalId
    FN(user_find_by_local_id),                                                  // 10 XUserFindUserByLocalId
    FN(user_get_id),                                                            // 11 XUserGetId
    FN(user_find_by_id),                                                        // 12 XUserFindUserById
    FN(user_get_is_guest),                                                      // 13 XUserGetIsGuest
    FN(user_get_state),                                                         // 14 XUserGetState
    FN(unknown_slot<15>),                                                       // 15 (unknown in the docs)
    FN((offline_failure<16, E_FAIL>)),                                          // 16 XUserGetGamerPictureAsync
    FN((offline_failure<17, E_FAIL>)),                                          // 17 XUserGetGamerPictureResultSize
    FN((offline_failure<18, E_FAIL>)),                                          // 18 XUserGetGamerPictureResult
    FN(user_get_age_group),                                                     // 19 XUserGetAgeGroup
    FN(user_check_privilege),                                                   // 20 XUserCheckPrivilege
    FN(user_resolve_privilege_async),                                           // 21 XUserResolvePrivilegeWithUiAsync
    FN(user_resolve_privilege_result),                                          // 22 XUserResolvePrivilegeWithUiResult
    FN((offline_failure<23, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 23 XUserGetTokenAndSignatureAsync
    FN((offline_failure<24, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 24 ...ResultSize
    FN((offline_failure<25, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 25 ...Result
    FN((offline_failure<26, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 26 XUserGetTokenAndSignatureUtf16Async
    FN((offline_failure<27, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 27 ...Utf16ResultSize
    FN((offline_failure<28, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 28 ...Utf16Result
    FN((offline_failure<29, E_GAMEUSER_FAILED_TO_RESOLVE>)),                    // 29 XUserResolveIssueWithUiAsync
    FN((offline_failure<30, E_GAMEUSER_FAILED_TO_RESOLVE>)),                    // 30 XUserResolveIssueWithUiResult
    FN((offline_failure<31, E_GAMEUSER_FAILED_TO_RESOLVE>)),                    // 31 XUserResolveIssueWithUiUtf16Async
    FN((offline_failure<32, E_GAMEUSER_FAILED_TO_RESOLVE>)),                    // 32 XUserResolveIssueWithUiUtf16Result
    FN(user_register_change),                                                   // 33 XUserRegisterForChangeEvent
    FN(user_unregister_change),                                                 // 34 XUserUnregisterForChangeEvent
    FN(user_get_signout_deferral),                                              // 35 XUserGetSignOutDeferral
    FN(user_close_signout_deferral),                                            // 36 XUserCloseSignOutDeferralHandle
    FN(user_add_by_id_async),                                                   // 37 XUserAddByIdWithUiAsync (2)
    FN(user_add_by_id_result),                                                  // 38 XUserAddByIdWithUiResult (2)
    FN((offline_failure<39, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 39 XUserGetMsaTokenSilentlyAsync (3)
    FN((offline_failure<40, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 40 ...Result (3)
    FN((offline_failure<41, E_GAMEUSER_FAILED_TO_GET_TOKEN>)),                  // 41 ...ResultSize (3)
    FN(user_is_store_user),                                                     // 42 XUserIsStoreUser (4)
    FN((offline_failure<43, E_NOTIMPL>)),                                       // 43 XUserPlatformRemoteConnectSetEventHandlers (5, PC)
    FN((offline_failure<44, E_NOTIMPL>)),                                       // 44 XUserPlatformRemoteConnectCancelPrompt (5, PC)
    FN((offline_failure<45, E_NOTIMPL>)),                                       // 45 XUserPlatformSpopPromptSetEventHandlers (5, PC)
    FN((offline_failure<46, E_NOTIMPL>)),                                       // 46 XUserPlatformSpopPromptComplete (5, PC)
    FN(private_slot<47>),                                                       // 47 XUserIsSignOutPresent (6): false
    FN((offline_failure<48, E_NOTIMPL>)),                                       // 48 XUserSignOutAsync (6)
    FN((offline_failure<49, E_NOTIMPL>)),                                       // 49 XUserSignOutResult (6)
};
static_assert(std::size(kUserVtable) == 50, "IUnknown + IXUserImpl's 34 methods + 2 + 3 + 1 + 4 + 3");

// The gamertag face's IUnknown slots get `this` = &gamertag_vtable; the object is a singleton, so they ignore it.
const void* const kGamertagVtable[] = {
    FN(object_query_interface), FN(object_add_ref), FN(object_release),
    FN(user_get_gamertag),                                                      // 3 XUserGetGamertag
};

#undef FN

UserObject* user_object() {
    static UserObject o{kUserVtable, kGamertagVtable};
    return &o;
}

}  // namespace

namespace rt::shims {

// Called by xgameruntime.cpp's QueryApiImpl for every class: true when the class is XUserImpl, with *hr set.
bool xgameruntime_user_query(const GUID& class_id, REFIID interface_id, void** out, HRESULT* hr) {
    if (class_id != kUserClass) return false;
    *hr = query(interface_id, out);
    return true;
}

// A new handle to the offline user, for the classes that hand one out too (XUserFindForDevice, in
// xgameruntime_offline.cpp). The title closes it with XUserCloseHandle like any other.
void* xgameruntime_user_new_handle() { return new_handle(); }

// For tests: the XUserHandles a title holds open.
long xgameruntime_user_open_handles() {
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return static_cast<long>(g_handles.size());
}

}  // namespace rt::shims
