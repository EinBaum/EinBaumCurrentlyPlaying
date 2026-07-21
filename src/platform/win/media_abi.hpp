// Hand-declared Windows.Media.Control ABI: MinGW ships no windows.media.control.h.
// Only the methods the poller calls are declared; declaration order matches the WinRT
// metadata so the vtable offsets are correct.
#pragma once
#include <cstdint>
#include <windows.h>
#include <roapi.h>
#include <winstring.h>

constexpr GUID IID_IMgrStatics = {0x2050c4ee,0x11a0,0x57de,{0xae,0xd7,0xc9,0x7c,0x70,0x33,0x82,0x45}};
constexpr GUID IID_IMgr        = {0xcace8eac,0xe86e,0x504a,{0xab,0x31,0x5f,0xf8,0xff,0x1b,0xce,0x49}};
constexpr GUID IID_ISession    = {0x7148c835,0x9b14,0x5ae2,{0xab,0x85,0xdc,0x9b,0x1c,0x14,0xe1,0xa8}};
constexpr GUID IID_IMediaProps = {0x68856cf6,0xadb4,0x54b2,{0xac,0x16,0x05,0x83,0x79,0x07,0xac,0xb6}};
constexpr GUID IID_IPlayback   = {0x94b4b6cf,0xe8ba,0x51ad,{0x87,0xa7,0xc1,0x0a,0xde,0x10,0x61,0x27}};
constexpr GUID IID_ITimeline   = {0xede34136,0x6f25,0x588d,{0x8e,0xcf,0xea,0x5b,0x67,0x35,0xaa,0xa5}};
constexpr GUID IID_IVVSession  = {0x9b2672da,0x5088,0x5a1d,{0xac,0xd9,0xa3,0xfc,0x5e,0xf1,0xcf,0xa4}};
constexpr GUID IID_IAsyncMgr   = {0x3eec115e,0x7346,0x5c27,{0x8c,0x5f,0xda,0x78,0x51,0x4a,0x27,0x7b}};
constexpr GUID IID_IAsyncProps = {0xb185e6f3,0xe0d8,0x51cb,{0x91,0x3f,0xc9,0x8d,0x48,0xc9,0x3c,0x46}};
constexpr GUID IID_IAsyncInfo  = {0x00000036,0x0000,0x0000,{0xc0,0x00,0x00,0x00,0x00,0x00,0x00,0x46}};
constexpr GUID IID_IRandomAccessStreamReference = {0x33ee3134,0x1dd6,0x4e3a,{0x80,0x67,0xd1,0xc1,0x62,0xe8,0x64,0x2b}};
constexpr GUID IID_IAsyncStream = {0xc4a57c5e,0x32b0,0x55b3,{0xad,0x13,0xce,0x1c,0x23,0x04,0x1e,0xd6}};

struct TimeSpan { int64_t Duration; };       // 100-ns ticks
struct DateTime { int64_t UniversalTime; };  // 100-ns ticks since 1601-01-01 UTC (FILETIME epoch)

// Pasting the six IInspectable slots directly into each vtable struct keeps the type
// standard-layout, so the member offsets equal the COM vtable slot offsets. Inheriting a
// shared base would leave the layout implementation-defined and break the offsets.
#define INSPECTABLE_HEAD(T) \
    HRESULT (__stdcall *QueryInterface)(T*, const GUID*, void**); \
    ULONG   (__stdcall *AddRef)(T*); \
    ULONG   (__stdcall *Release)(T*); \
    HRESULT (__stdcall *GetIids)(T*, ULONG*, GUID**); \
    HRESULT (__stdcall *GetRuntimeClassName)(T*, HSTRING*); \
    HRESULT (__stdcall *GetTrustLevel)(T*, int*);

#define DECL_IFACE(T) struct T##Vtbl; struct T { const T##Vtbl* v; };

DECL_IFACE(IAsyncInfo)
DECL_IFACE(IAsyncOp)
DECL_IFACE(IMgrStatics)
DECL_IFACE(IMgr)
DECL_IFACE(IVVSession)
DECL_IFACE(ISession)
DECL_IFACE(IMediaProps)
DECL_IFACE(ITimeline)
DECL_IFACE(IPlayback)
DECL_IFACE(IStreamRef)

struct IAsyncInfoVtbl { INSPECTABLE_HEAD(IAsyncInfo)
    HRESULT (__stdcall *get_Id)(IAsyncInfo*, unsigned*);
    HRESULT (__stdcall *get_Status)(IAsyncInfo*, int*);
};
struct IAsyncOpVtbl { INSPECTABLE_HEAD(IAsyncOp)
    void* put_Completed; void* get_Completed;
    HRESULT (__stdcall *GetResults)(IAsyncOp*, void**);
};
struct IMgrStaticsVtbl { INSPECTABLE_HEAD(IMgrStatics)
    HRESULT (__stdcall *RequestAsync)(IMgrStatics*, IAsyncOp**);
};
struct IMgrVtbl { INSPECTABLE_HEAD(IMgr)
    HRESULT (__stdcall *GetCurrentSession)(IMgr*, ISession**);
    HRESULT (__stdcall *GetSessions)(IMgr*, IVVSession**);
};
struct IVVSessionVtbl { INSPECTABLE_HEAD(IVVSession)
    HRESULT (__stdcall *GetAt)(IVVSession*, unsigned, ISession**);
    HRESULT (__stdcall *get_Size)(IVVSession*, unsigned*);
};
struct ISessionVtbl { INSPECTABLE_HEAD(ISession)
    HRESULT (__stdcall *get_SourceAppUserModelId)(ISession*, HSTRING*);
    HRESULT (__stdcall *TryGetMediaPropertiesAsync)(ISession*, IAsyncOp**);
    HRESULT (__stdcall *GetTimelineProperties)(ISession*, ITimeline**);
    HRESULT (__stdcall *GetPlaybackInfo)(ISession*, IPlayback**);
};
struct IMediaPropsVtbl { INSPECTABLE_HEAD(IMediaProps)
    HRESULT (__stdcall *get_Title)(IMediaProps*, HSTRING*);
    HRESULT (__stdcall *get_Subtitle)(IMediaProps*, HSTRING*);
    HRESULT (__stdcall *get_AlbumArtist)(IMediaProps*, HSTRING*);
    HRESULT (__stdcall *get_Artist)(IMediaProps*, HSTRING*);
    HRESULT (__stdcall *get_AlbumTitle)(IMediaProps*, HSTRING*);
    void* get_TrackNumber; void* get_Genres;
    void* get_AlbumTrackCount; void* get_PlaybackType;
    HRESULT (__stdcall *get_Thumbnail)(IMediaProps*, IStreamRef**);  // slot [15]
};
struct ITimelineVtbl { INSPECTABLE_HEAD(ITimeline)
    HRESULT (__stdcall *get_StartTime)(ITimeline*, TimeSpan*);
    HRESULT (__stdcall *get_EndTime)(ITimeline*, TimeSpan*);
    void* get_MinSeekTime; void* get_MaxSeekTime;
    HRESULT (__stdcall *get_Position)(ITimeline*, TimeSpan*);          // slot [10]
    HRESULT (__stdcall *get_LastUpdatedTime)(ITimeline*, DateTime*);   // slot [11]
};
struct IPlaybackVtbl { INSPECTABLE_HEAD(IPlayback)
    void* get_Controls;
    HRESULT (__stdcall *get_PlaybackStatus)(IPlayback*, int*);  // slot [7]
};
struct IStreamRefVtbl { INSPECTABLE_HEAD(IStreamRef)
    HRESULT (__stdcall *OpenReadAsync)(IStreamRef*, IAsyncOp**);  // slot [6]
};
