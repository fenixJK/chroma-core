#pragma once

#include <stdint.h>

#ifdef _WIN32
#define CHROMA_API extern "C" __declspec(dllexport)
#define CHROMA_CALL __stdcall
#else
#define CHROMA_API extern "C"
#define CHROMA_CALL
#endif

struct ChromaPoint {
    int32_t x;
    int32_t y;
};

#define CHROMA_MAX_HUE_RANGES 8

struct ChromaHueRange {
    int32_t minHue;
    int32_t maxHue;
};

struct ChromaChannelRange {
    int32_t minValue;
    int32_t maxValue;
};

struct ChromaConfigV1 {
    int32_t structSize;

    int32_t centerHueRangeCount;
    ChromaHueRange centerHueRanges[CHROMA_MAX_HUE_RANGES];
    ChromaChannelRange centerSatRange;
    ChromaChannelRange centerValRange;

    int32_t centerMorphOpenIterations;
    int32_t centerMorphCloseIterations;
    int32_t centerDilateIterations;

    int32_t minBlobArea;
    int32_t maxBlobArea;
    float minCircularity;
    float minCenterFillRatio;

    int32_t requireContextRing;
    int32_t ringInnerRadiusPercent;
    int32_t ringOuterRadiusPercent;

    ChromaChannelRange contextSupportSatRange;
    ChromaChannelRange contextSupportValRange;

    int32_t contextExcludeHueRangeCount;
    ChromaHueRange contextExcludeHueRanges[CHROMA_MAX_HUE_RANGES];
    float contextMinSupportRatio;

    int32_t drawRejectedCandidates;
};

struct ChromaDebugImageV1 {
    int32_t structSize;
    void* bgraPixels;
    int32_t bgraCapacityBytes;
    int32_t width;
    int32_t height;
    int32_t strideBytes;
    int32_t bytesRequired;
    int32_t bytesWritten;
};
enum ChromaStatusCode : int32_t {
    CHROMA_STATUS_OK = 0,
    CHROMA_STATUS_INVALID_ARGUMENT = 1,
    CHROMA_STATUS_CONFIG_ERROR = 2,
    CHROMA_STATUS_RUNTIME_ERROR = 3,
    CHROMA_STATUS_BUFFER_TOO_SMALL = 4
};

CHROMA_API int32_t CHROMA_CALL Chroma_GetApiVersion();
CHROMA_API int32_t CHROMA_CALL Chroma_GetConfigStructSize();

// Runtime config control (process-wide).
CHROMA_API int32_t CHROMA_CALL Chroma_GetDefaultConfig(
    ChromaConfigV1* outConfig,
    wchar_t* outError,
    int32_t outErrorChars);
CHROMA_API int32_t CHROMA_CALL Chroma_GetActiveConfig(
    ChromaConfigV1* outConfig,
    wchar_t* outError,
    int32_t outErrorChars);
CHROMA_API int32_t CHROMA_CALL Chroma_SetActiveConfig(
    const ChromaConfigV1* config,
    wchar_t* outError,
    int32_t outErrorChars);
CHROMA_API int32_t CHROMA_CALL Chroma_ResetConfigToDefault(
    wchar_t* outError,
    int32_t outErrorChars);

// Single-call locate APIs:
// - uses the active runtime config set via Chroma_SetActiveConfig.
// - outPoints may be null; if non-null, up to outCapacity points are written.
// - outTotalFound receives total accepted detections (optional).
// - outWritten receives how many points were copied to outPoints (optional).
// Bitmap input as BGRA32 pixel buffer.
// - bgraPixels points to top-down or bottom-up rows (controlled by strideBytes sign).
// - strideBytes may be negative for bottom-up buffers.
CHROMA_API int32_t CHROMA_CALL Chroma_LocateBitmapBGRAW(
    const void* bgraPixels,
    int32_t width,
    int32_t height,
    int32_t strideBytes,
    ChromaPoint* outPoints,
    int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    int32_t outErrorChars);

// Per-call override when the caller does not want to mutate active config.
CHROMA_API int32_t CHROMA_CALL Chroma_LocateBitmapWithConfigBGRAW(
    const void* bgraPixels,
    int32_t width,
    int32_t height,
    int32_t strideBytes,
    const ChromaConfigV1* config,
    ChromaPoint* outPoints,
    int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    int32_t outErrorChars);

// Debug-image variant:
// - same detection outputs as Chroma_LocateBitmapBGRAW
// - optionally writes a BGRA debug image (side-by-side overlay/mask) into outDebugImage
CHROMA_API int32_t CHROMA_CALL Chroma_LocateBitmapWithDebugBGRAW(
    const void* bgraPixels,
    int32_t width,
    int32_t height,
    int32_t strideBytes,
    ChromaPoint* outPoints,
    int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    ChromaDebugImageV1* outDebugImage,
    wchar_t* outError,
    int32_t outErrorChars);
// Direct GDI bitmap input (HBITMAP).
CHROMA_API int32_t CHROMA_CALL Chroma_LocateHBitmap(
    const void* hBitmap,
    ChromaPoint* outPoints,
    int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    int32_t outErrorChars);

// Capture from a window handle and run detection.
// - captureClientArea: 1 = client area, 0 = full window area.
CHROMA_API int32_t CHROMA_CALL Chroma_LocateHWND(
    const void* hWnd,
    int32_t captureClientArea,
    ChromaPoint* outPoints,
    int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    int32_t outErrorChars);

