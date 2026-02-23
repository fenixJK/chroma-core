#include "ChromaCore.h"
#include "ChromaApi.h"

#include <algorithm>
#include <cstdint>
#include <cwchar>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

void WriteErrorMessage(wchar_t* outError, const int32_t outErrorChars, const wchar_t* message) {
    if (outError == nullptr || outErrorChars <= 0) {
        return;
    }
#ifdef _WIN32
    wcsncpy_s(outError, static_cast<size_t>(outErrorChars), message, _TRUNCATE);
#else
    std::wcsncpy(outError, message, static_cast<size_t>(outErrorChars - 1));
    outError[outErrorChars - 1] = L'\0';
#endif
}

std::wstring Utf8ToWide(const std::string& utf8) {
#ifdef _WIN32
    if (utf8.empty()) {
        return L"";
    }
    const int required = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (required <= 0) {
        return L"";
    }
    std::wstring out(static_cast<size_t>(required), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, out.data(), required);
    if (!out.empty() && out.back() == L'\0') {
        out.pop_back();
    }
    return out;
#else
    return std::wstring(utf8.begin(), utf8.end());
#endif
}

vision::ColorPatternConfig BuildDefaultPatternConfig() {
    vision::ColorPatternConfig cfg;
    cfg.centerColor.hues = vision::HueRangeSet({ {16, 32} });
    cfg.centerColor.satRange = { 50, 125 };
    cfg.centerColor.valRange = { 85, 255 };

    cfg.centerMorph.openIterations = 5;
    cfg.centerMorph.closeIterations = 3;
    cfg.centerMorph.dilateIterations = 1;

    cfg.shape.minArea = 20;
    cfg.shape.maxArea = 800;
    cfg.shape.minCircularity = 0.75F;
    cfg.shape.minFillRatio = 0.68F;

    cfg.context.enabled = true;
    cfg.context.innerRadiusPercent = 105;
    cfg.context.outerRadiusPercent = 225;
    cfg.context.supportColor.hues = vision::HueRangeSet({ {0, 179} });
    cfg.context.supportColor.satRange = { 0, 255 };
    cfg.context.supportColor.valRange = { 120, 255 };
    cfg.context.excludeHues = vision::HueRangeSet({ {52, 68}, {24, 48} });
    cfg.context.excludeSatRange = cfg.context.supportColor.satRange;
    cfg.context.excludeValRange = cfg.context.supportColor.valRange;
    cfg.context.minSupportRatio = 0.42F;

    cfg.debug.drawRejected = false;
    cfg.debug.drawLabels = true;
    cfg.debug.drawLabelBackground = true;
    cfg.debug.acceptedColor = cv::Scalar(0, 255, 0);
    cfg.debug.rejectedColor = cv::Scalar(0, 165, 255);
    cfg.debug.textColor = cv::Scalar(0, 255, 0);
    cfg.debug.labelBgColor = cv::Scalar(0, 0, 0);
    cfg.debug.fontScale = 0.45;
    cfg.debug.lineThickness = 1;
    cfg.debug.labelPaddingPx = 2;
    return cfg;
}

std::mutex g_cfgMutex;
vision::ColorPatternConfig g_activeConfig = BuildDefaultPatternConfig();

vision::ColorPatternConfig GetActiveConfigCopy() {
    std::lock_guard<std::mutex> lock(g_cfgMutex);
    return g_activeConfig;
}

void SetActiveConfig(const vision::ColorPatternConfig& cfg) {
    std::lock_guard<std::mutex> lock(g_cfgMutex);
    g_activeConfig = cfg;
}

bool ValidateChannelRange(
    const ChromaChannelRange& range,
    const char* name,
    std::string& errorOut) {
    if (range.minValue < 0 || range.minValue > 255 || range.maxValue < 0 || range.maxValue > 255) {
        errorOut = std::string(name) + " must be in [0,255].";
        return false;
    }
    if (range.minValue > range.maxValue) {
        errorOut = std::string(name) + " must satisfy minValue <= maxValue.";
        return false;
    }
    return true;
}

bool ValidateHueRange(
    const ChromaHueRange& range,
    const char* name,
    std::string& errorOut) {
    if (range.minHue < 0 || range.minHue > 179 || range.maxHue < 0 || range.maxHue > 179) {
        errorOut = std::string(name) + " must be in [0,179].";
        return false;
    }
    return true;
}

vision::ChannelRange ToChannelRange(const ChromaChannelRange& in) {
    vision::ChannelRange out;
    out.minValue = in.minValue;
    out.maxValue = in.maxValue;
    return out;
}

ChromaChannelRange ToApiChannelRange(const vision::ChannelRange& in) {
    ChromaChannelRange out{};
    out.minValue = in.minValue;
    out.maxValue = in.maxValue;
    return out;
}

int32_t ConvertApiConfigToPattern(
    const ChromaConfigV1& in,
    vision::ColorPatternConfig& outCfg,
    std::string& errorOut) {
    if (in.structSize < static_cast<int32_t>(sizeof(ChromaConfigV1))) {
        errorOut = "ChromaConfigV1.structSize is smaller than required.";
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (in.centerHueRangeCount < 1 || in.centerHueRangeCount > CHROMA_MAX_HUE_RANGES) {
        errorOut = "centerHueRangeCount must be in [1, CHROMA_MAX_HUE_RANGES].";
        return CHROMA_STATUS_CONFIG_ERROR;
    }
    if (in.contextExcludeHueRangeCount < 0 || in.contextExcludeHueRangeCount > CHROMA_MAX_HUE_RANGES) {
        errorOut = "contextExcludeHueRangeCount must be in [0, CHROMA_MAX_HUE_RANGES].";
        return CHROMA_STATUS_CONFIG_ERROR;
    }
    if (!ValidateChannelRange(in.centerSatRange, "centerSatRange", errorOut) ||
        !ValidateChannelRange(in.centerValRange, "centerValRange", errorOut) ||
        !ValidateChannelRange(in.contextSupportSatRange, "contextSupportSatRange", errorOut) ||
        !ValidateChannelRange(in.contextSupportValRange, "contextSupportValRange", errorOut)) {
        return CHROMA_STATUS_CONFIG_ERROR;
    }

    vision::ColorPatternConfig cfg{};
    cfg.centerColor.hues.Clear();
    for (int32_t i = 0; i < in.centerHueRangeCount; ++i) {
        const ChromaHueRange h = in.centerHueRanges[i];
        if (!ValidateHueRange(h, "centerHueRanges[i]", errorOut)) {
            return CHROMA_STATUS_CONFIG_ERROR;
        }
        cfg.centerColor.hues.Add({ h.minHue, h.maxHue });
    }

    cfg.centerColor.satRange = ToChannelRange(in.centerSatRange);
    cfg.centerColor.valRange = ToChannelRange(in.centerValRange);

    cfg.centerMorph.openIterations = in.centerMorphOpenIterations;
    cfg.centerMorph.closeIterations = in.centerMorphCloseIterations;
    cfg.centerMorph.dilateIterations = in.centerDilateIterations;

    cfg.shape.minArea = in.minBlobArea;
    cfg.shape.maxArea = in.maxBlobArea;
    cfg.shape.minCircularity = in.minCircularity;
    cfg.shape.minFillRatio = in.minCenterFillRatio;

    cfg.context.enabled = (in.requireContextRing != 0);
    cfg.context.innerRadiusPercent = in.ringInnerRadiusPercent;
    cfg.context.outerRadiusPercent = in.ringOuterRadiusPercent;
    cfg.context.supportColor.hues = vision::HueRangeSet({ {0, 179} });
    cfg.context.supportColor.satRange = ToChannelRange(in.contextSupportSatRange);
    cfg.context.supportColor.valRange = ToChannelRange(in.contextSupportValRange);

    cfg.context.excludeHues.Clear();
    for (int32_t i = 0; i < in.contextExcludeHueRangeCount; ++i) {
        const ChromaHueRange h = in.contextExcludeHueRanges[i];
        if (!ValidateHueRange(h, "contextExcludeHueRanges[i]", errorOut)) {
            return CHROMA_STATUS_CONFIG_ERROR;
        }
        cfg.context.excludeHues.Add({ h.minHue, h.maxHue });
    }
    cfg.context.excludeSatRange = cfg.context.supportColor.satRange;
    cfg.context.excludeValRange = cfg.context.supportColor.valRange;
    cfg.context.minSupportRatio = in.contextMinSupportRatio;

    cfg.debug.drawRejected = (in.drawRejectedCandidates != 0);
    cfg.debug.drawLabels = true;
    cfg.debug.drawLabelBackground = true;
    cfg.debug.acceptedColor = cv::Scalar(0, 255, 0);
    cfg.debug.rejectedColor = cv::Scalar(0, 165, 255);
    cfg.debug.textColor = cv::Scalar(0, 255, 0);
    cfg.debug.labelBgColor = cv::Scalar(0, 0, 0);
    cfg.debug.fontScale = 0.45;
    cfg.debug.lineThickness = 1;
    cfg.debug.labelPaddingPx = 2;

    std::string validationError;
    if (!vision::ColorPatternFinder::ValidateConfig(cfg, &validationError)) {
        errorOut = validationError;
        return CHROMA_STATUS_CONFIG_ERROR;
    }

    outCfg = std::move(cfg);
    return CHROMA_STATUS_OK;
}

ChromaConfigV1 ConvertPatternToApiConfig(const vision::ColorPatternConfig& in) {
    ChromaConfigV1 out{};
    out.structSize = static_cast<int32_t>(sizeof(ChromaConfigV1));

    const std::vector<vision::HueRange>& centerHueRanges = in.centerColor.hues.Ranges();
    out.centerHueRangeCount = static_cast<int32_t>(
        std::min<size_t>(centerHueRanges.size(), static_cast<size_t>(CHROMA_MAX_HUE_RANGES)));
    for (int32_t i = 0; i < out.centerHueRangeCount; ++i) {
        out.centerHueRanges[i].minHue = centerHueRanges[static_cast<size_t>(i)].minHue;
        out.centerHueRanges[i].maxHue = centerHueRanges[static_cast<size_t>(i)].maxHue;
    }
    out.centerSatRange = ToApiChannelRange(in.centerColor.satRange);
    out.centerValRange = ToApiChannelRange(in.centerColor.valRange);

    out.centerMorphOpenIterations = in.centerMorph.openIterations;
    out.centerMorphCloseIterations = in.centerMorph.closeIterations;
    out.centerDilateIterations = in.centerMorph.dilateIterations;

    out.minBlobArea = in.shape.minArea;
    out.maxBlobArea = in.shape.maxArea;
    out.minCircularity = in.shape.minCircularity;
    out.minCenterFillRatio = in.shape.minFillRatio;

    out.requireContextRing = in.context.enabled ? 1 : 0;
    out.ringInnerRadiusPercent = in.context.innerRadiusPercent;
    out.ringOuterRadiusPercent = in.context.outerRadiusPercent;

    out.contextSupportSatRange = ToApiChannelRange(in.context.supportColor.satRange);
    out.contextSupportValRange = ToApiChannelRange(in.context.supportColor.valRange);

    const std::vector<vision::HueRange>& excludeHueRanges = in.context.excludeHues.Ranges();
    out.contextExcludeHueRangeCount = static_cast<int32_t>(
        std::min<size_t>(excludeHueRanges.size(), static_cast<size_t>(CHROMA_MAX_HUE_RANGES)));
    for (int32_t i = 0; i < out.contextExcludeHueRangeCount; ++i) {
        out.contextExcludeHueRanges[i].minHue = excludeHueRanges[static_cast<size_t>(i)].minHue;
        out.contextExcludeHueRanges[i].maxHue = excludeHueRanges[static_cast<size_t>(i)].maxHue;
    }
    out.contextMinSupportRatio = in.context.minSupportRatio;
    out.drawRejectedCandidates = in.debug.drawRejected ? 1 : 0;

    return out;
}

int32_t BuildConfigFromPointer(
    const ChromaConfigV1* config,
    vision::ColorPatternConfig& outCfg,
    wchar_t* outError,
    const int32_t outErrorChars) {
    if (config == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"config is null.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    std::string error;
    const int32_t status = ConvertApiConfigToPattern(*config, outCfg, error);
    if (status != CHROMA_STATUS_OK) {
        WriteErrorMessage(outError, outErrorChars, Utf8ToWide(error).c_str());
        return status;
    }
    return CHROMA_STATUS_OK;
}

int32_t DetectRunResultFromMat(
    const cv::Mat& sceneBgrOrBgra,
    const vision::ColorPatternConfig& cfg,
    vision::ColorPatternRunResult& outResult,
    wchar_t* outError,
    const int32_t outErrorChars) {
    outResult = {};
    WriteErrorMessage(outError, outErrorChars, L"");

    if (sceneBgrOrBgra.empty()) {
        WriteErrorMessage(outError, outErrorChars, L"Bitmap input is empty.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    try {
        const vision::ColorPatternFinder finder(cfg);
        outResult = finder.Find(sceneBgrOrBgra);
        return CHROMA_STATUS_OK;
    }
    catch (const std::exception& ex) {
        const std::wstring wmsg = Utf8ToWide(ex.what());
        WriteErrorMessage(outError, outErrorChars, wmsg.empty() ? L"Runtime error." : wmsg.c_str());
        return CHROMA_STATUS_RUNTIME_ERROR;
    }
    catch (...) {
        WriteErrorMessage(outError, outErrorChars, L"Unknown runtime error.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }
}

int32_t DetectAcceptedCentersFromMat(
    const cv::Mat& sceneBgrOrBgra,
    const vision::ColorPatternConfig& cfg,
    std::vector<ChromaPoint>& outCenters,
    wchar_t* outError,
    const int32_t outErrorChars) {
    outCenters.clear();

    vision::ColorPatternRunResult result;
    const int32_t status = DetectRunResultFromMat(sceneBgrOrBgra, cfg, result, outError, outErrorChars);
    if (status != CHROMA_STATUS_OK) {
        return status;
    }

    outCenters.reserve(result.acceptedCentersPx.size());
    for (const auto& p : result.acceptedCentersPx) {
        outCenters.push_back(ChromaPoint{ p.x, p.y });
    }
    return CHROMA_STATUS_OK;
}
int32_t WriteLocateOutputs(
    const std::vector<ChromaPoint>& centers,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    const int32_t total = static_cast<int32_t>(centers.size());
    if (outTotalFound != nullptr) {
        *outTotalFound = total;
    }

    const bool countOnly = (outPoints == nullptr || outCapacity <= 0);
    if (countOnly) {
        if (outWritten != nullptr) {
            *outWritten = 0;
        }
        return CHROMA_STATUS_OK;
    }

    const int32_t toCopy = std::min(total, outCapacity);
    for (int32_t i = 0; i < toCopy; ++i) {
        outPoints[i] = centers[static_cast<size_t>(i)];
    }
    if (outWritten != nullptr) {
        *outWritten = toCopy;
    }

    if (toCopy < total) {
        WriteErrorMessage(outError, outErrorChars, L"Output buffer too small.");
        return CHROMA_STATUS_BUFFER_TOO_SMALL;
    }
    return CHROMA_STATUS_OK;
}

int32_t WriteDebugImageOutput(
    const vision::ColorPatternRunResult& result,
    const cv::Mat& fallbackScene,
    ChromaDebugImageV1* outDebugImage,
    wchar_t* outError,
    const int32_t outErrorChars) {
    if (outDebugImage == nullptr) {
        return CHROMA_STATUS_OK;
    }

    if (outDebugImage->structSize < static_cast<int32_t>(sizeof(ChromaDebugImageV1))) {
        WriteErrorMessage(outError, outErrorChars, L"ChromaDebugImageV1.structSize is smaller than required.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    outDebugImage->width = 0;
    outDebugImage->height = 0;
    outDebugImage->strideBytes = 0;
    outDebugImage->bytesRequired = 0;
    outDebugImage->bytesWritten = 0;

    cv::Mat debugView;
    if (!result.sideBySideDebug.empty()) {
        debugView = result.sideBySideDebug;
    }
    else if (!result.debugOverlay.empty()) {
        debugView = result.debugOverlay;
    }
    else if (!result.debugMask.empty()) {
        debugView = result.debugMask;
    }
    else {
        debugView = fallbackScene;
    }

    if (debugView.empty()) {
        return CHROMA_STATUS_OK;
    }

    cv::Mat debugBgra;
    if (debugView.channels() == 4) {
        debugBgra = debugView;
    }
    else if (debugView.channels() == 3) {
        cv::cvtColor(debugView, debugBgra, cv::COLOR_BGR2BGRA);
    }
    else if (debugView.channels() == 1) {
        cv::cvtColor(debugView, debugBgra, cv::COLOR_GRAY2BGRA);
    }
    else {
        WriteErrorMessage(outError, outErrorChars, L"Unsupported debug image format.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    if (!debugBgra.isContinuous()) {
        debugBgra = debugBgra.clone();
    }

    const int32_t width = debugBgra.cols;
    const int32_t height = debugBgra.rows;
    const int64_t stride64 = static_cast<int64_t>(width) * 4;
    const int64_t required64 = stride64 * static_cast<int64_t>(height);
    if (width < 0 || height < 0 || stride64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
        required64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        WriteErrorMessage(outError, outErrorChars, L"Debug image is too large.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    outDebugImage->width = width;
    outDebugImage->height = height;
    outDebugImage->strideBytes = static_cast<int32_t>(stride64);
    outDebugImage->bytesRequired = static_cast<int32_t>(required64);

    if (outDebugImage->bgraPixels == nullptr || outDebugImage->bgraCapacityBytes <= 0) {
        return CHROMA_STATUS_OK;
    }

    if (outDebugImage->bgraCapacityBytes < outDebugImage->bytesRequired) {
        WriteErrorMessage(outError, outErrorChars, L"Debug image buffer too small.");
        return CHROMA_STATUS_BUFFER_TOO_SMALL;
    }

    std::memcpy(outDebugImage->bgraPixels, debugBgra.data, static_cast<size_t>(outDebugImage->bytesRequired));
    outDebugImage->bytesWritten = outDebugImage->bytesRequired;
    return CHROMA_STATUS_OK;
}

int32_t ValidateOutputArgs(
    const int32_t outCapacity,
    const ChromaPoint* outPoints,
    wchar_t* outError,
    const int32_t outErrorChars) {
    if (outCapacity < 0) {
        WriteErrorMessage(outError, outErrorChars, L"outCapacity must be >= 0.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (outCapacity > 0 && outPoints == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"outPoints is null while outCapacity > 0.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    return CHROMA_STATUS_OK;
}

int32_t LocateBitmapImpl(
    const void* bgraPixels,
    const int32_t width,
    const int32_t height,
    const int32_t strideBytes,
    const vision::ColorPatternConfig& cfg,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    if (outTotalFound != nullptr) {
        *outTotalFound = 0;
    }
    if (outWritten != nullptr) {
        *outWritten = 0;
    }
    if (bgraPixels == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"bgraPixels is null.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (width <= 0 || height <= 0) {
        WriteErrorMessage(outError, outErrorChars, L"width/height must be > 0.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (strideBytes == 0) {
        WriteErrorMessage(outError, outErrorChars, L"strideBytes must not be 0.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    const int64_t minStrideBytes = static_cast<int64_t>(width) * 4;
    const int64_t stride64 = static_cast<int64_t>(strideBytes);
    const int64_t absStride64 = (stride64 < 0) ? -stride64 : stride64;
    if (absStride64 < minStrideBytes) {
        WriteErrorMessage(outError, outErrorChars, L"strideBytes is smaller than width*4.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (height > 1) {
        const uint64_t lastRowOffset = static_cast<uint64_t>(absStride64) * static_cast<uint64_t>(height - 1);
        if (lastRowOffset > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            WriteErrorMessage(outError, outErrorChars, L"Bitmap dimensions are too large.");
            return CHROMA_STATUS_INVALID_ARGUMENT;
        }
    }

    const int32_t outputStatus = ValidateOutputArgs(outCapacity, outPoints, outError, outErrorChars);
    if (outputStatus != CHROMA_STATUS_OK) {
        return outputStatus;
    }

    const size_t absStride = static_cast<size_t>(absStride64);
    cv::Mat bgraView;
    cv::Mat scene;
    if (strideBytes > 0) {
        bgraView = cv::Mat(height, width, CV_8UC4, const_cast<void*>(bgraPixels), absStride);
        scene = bgraView;
    }
    else {
        const uint8_t* base = static_cast<const uint8_t*>(bgraPixels);
        const size_t lastRowOffset = static_cast<size_t>(static_cast<uint64_t>(absStride64) * static_cast<uint64_t>(height - 1));
        const uint8_t* topRow = base + lastRowOffset;
        bgraView = cv::Mat(height, width, CV_8UC4, const_cast<uint8_t*>(topRow), absStride);
        cv::flip(bgraView, scene, 0);
    }

    std::vector<ChromaPoint> centers;
    const int32_t detectStatus = DetectAcceptedCentersFromMat(scene, cfg, centers, outError, outErrorChars);
    if (detectStatus != CHROMA_STATUS_OK) {
        return detectStatus;
    }
    return WriteLocateOutputs(centers, outPoints, outCapacity, outTotalFound, outWritten, outError, outErrorChars);
}

} // namespace

int32_t CHROMA_CALL ChromaRuntime_GetApiVersion() {
    return 1;
}

int32_t CHROMA_CALL ChromaRuntime_GetConfigStructSize() {
    return static_cast<int32_t>(sizeof(ChromaConfigV1));
}

int32_t CHROMA_CALL ChromaRuntime_GetDefaultConfig(
    ChromaConfigV1* outConfig,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
    if (outConfig == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"outConfig is null.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    *outConfig = ConvertPatternToApiConfig(BuildDefaultPatternConfig());
    return CHROMA_STATUS_OK;
}

int32_t CHROMA_CALL ChromaRuntime_GetActiveConfig(
    ChromaConfigV1* outConfig,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
    if (outConfig == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"outConfig is null.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    *outConfig = ConvertPatternToApiConfig(GetActiveConfigCopy());
    return CHROMA_STATUS_OK;
}

int32_t CHROMA_CALL ChromaRuntime_SetActiveConfig(
    const ChromaConfigV1* config,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
    try {
        vision::ColorPatternConfig cfg;
        const int32_t status = BuildConfigFromPointer(config, cfg, outError, outErrorChars);
        if (status != CHROMA_STATUS_OK) {
            return status;
        }
        SetActiveConfig(cfg);
        return CHROMA_STATUS_OK;
    }
    catch (const std::exception& ex) {
        WriteErrorMessage(outError, outErrorChars, Utf8ToWide(ex.what()).c_str());
        return CHROMA_STATUS_CONFIG_ERROR;
    }
    catch (...) {
        WriteErrorMessage(outError, outErrorChars, L"Unknown config error.");
        return CHROMA_STATUS_CONFIG_ERROR;
    }
}

int32_t CHROMA_CALL ChromaRuntime_ResetConfigToDefault(
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
    SetActiveConfig(BuildDefaultPatternConfig());
    return CHROMA_STATUS_OK;
}

int32_t CHROMA_CALL ChromaRuntime_LocateBitmapBGRAW(
    const void* bgraPixels,
    const int32_t width,
    const int32_t height,
    const int32_t strideBytes,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
    const vision::ColorPatternConfig cfg = GetActiveConfigCopy();
    return LocateBitmapImpl(
        bgraPixels,
        width,
        height,
        strideBytes,
        cfg,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
}

int32_t CHROMA_CALL ChromaRuntime_LocateBitmapWithDebugBGRAW(
    const void* bgraPixels,
    const int32_t width,
    const int32_t height,
    const int32_t strideBytes,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    ChromaDebugImageV1* outDebugImage,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");

    if (outTotalFound != nullptr) {
        *outTotalFound = 0;
    }
    if (outWritten != nullptr) {
        *outWritten = 0;
    }
    if (outDebugImage != nullptr) {
        if (outDebugImage->structSize < static_cast<int32_t>(sizeof(ChromaDebugImageV1))) {
            WriteErrorMessage(outError, outErrorChars, L"ChromaDebugImageV1.structSize is smaller than required.");
            return CHROMA_STATUS_INVALID_ARGUMENT;
        }
        outDebugImage->width = 0;
        outDebugImage->height = 0;
        outDebugImage->strideBytes = 0;
        outDebugImage->bytesRequired = 0;
        outDebugImage->bytesWritten = 0;
    }

    if (bgraPixels == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"bgraPixels is null.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (width <= 0 || height <= 0) {
        WriteErrorMessage(outError, outErrorChars, L"width/height must be > 0.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (strideBytes == 0) {
        WriteErrorMessage(outError, outErrorChars, L"strideBytes must not be 0.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    const int64_t minStrideBytes = static_cast<int64_t>(width) * 4;
    const int64_t stride64 = static_cast<int64_t>(strideBytes);
    const int64_t absStride64 = (stride64 < 0) ? -stride64 : stride64;
    if (absStride64 < minStrideBytes) {
        WriteErrorMessage(outError, outErrorChars, L"strideBytes is smaller than width*4.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }
    if (height > 1) {
        const uint64_t lastRowOffset = static_cast<uint64_t>(absStride64) * static_cast<uint64_t>(height - 1);
        if (lastRowOffset > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            WriteErrorMessage(outError, outErrorChars, L"Bitmap dimensions are too large.");
            return CHROMA_STATUS_INVALID_ARGUMENT;
        }
    }

    const int32_t outputStatus = ValidateOutputArgs(outCapacity, outPoints, outError, outErrorChars);
    if (outputStatus != CHROMA_STATUS_OK) {
        return outputStatus;
    }

    const size_t absStride = static_cast<size_t>(absStride64);
    cv::Mat bgraView;
    cv::Mat scene;
    if (strideBytes > 0) {
        bgraView = cv::Mat(height, width, CV_8UC4, const_cast<void*>(bgraPixels), absStride);
        scene = bgraView;
    }
    else {
        const uint8_t* base = static_cast<const uint8_t*>(bgraPixels);
        const size_t lastRowOffset = static_cast<size_t>(static_cast<uint64_t>(absStride64) * static_cast<uint64_t>(height - 1));
        const uint8_t* topRow = base + lastRowOffset;
        bgraView = cv::Mat(height, width, CV_8UC4, const_cast<uint8_t*>(topRow), absStride);
        cv::flip(bgraView, scene, 0);
    }

    const vision::ColorPatternConfig cfg = GetActiveConfigCopy();
    vision::ColorPatternRunResult runResult;
    const int32_t detectStatus = DetectRunResultFromMat(scene, cfg, runResult, outError, outErrorChars);
    if (detectStatus != CHROMA_STATUS_OK) {
        return detectStatus;
    }

    std::vector<ChromaPoint> centers;
    centers.reserve(runResult.acceptedCentersPx.size());
    for (const auto& p : runResult.acceptedCentersPx) {
        centers.push_back(ChromaPoint{ p.x, p.y });
    }

    const int32_t pointsStatus = WriteLocateOutputs(
        centers,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);

    const int32_t debugStatus = WriteDebugImageOutput(runResult, scene, outDebugImage, outError, outErrorChars);

    if (pointsStatus != CHROMA_STATUS_OK && pointsStatus != CHROMA_STATUS_BUFFER_TOO_SMALL) {
        return pointsStatus;
    }
    if (debugStatus != CHROMA_STATUS_OK && debugStatus != CHROMA_STATUS_BUFFER_TOO_SMALL) {
        return debugStatus;
    }
    if (pointsStatus == CHROMA_STATUS_BUFFER_TOO_SMALL || debugStatus == CHROMA_STATUS_BUFFER_TOO_SMALL) {
        return CHROMA_STATUS_BUFFER_TOO_SMALL;
    }

    return CHROMA_STATUS_OK;
}
int32_t CHROMA_CALL ChromaRuntime_LocateBitmapWithConfigBGRAW(
    const void* bgraPixels,
    const int32_t width,
    const int32_t height,
    const int32_t strideBytes,
    const ChromaConfigV1* config,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
    vision::ColorPatternConfig cfg;
    const int32_t cfgStatus = BuildConfigFromPointer(config, cfg, outError, outErrorChars);
    if (cfgStatus != CHROMA_STATUS_OK) {
        return cfgStatus;
    }

    return LocateBitmapImpl(
        bgraPixels,
        width,
        height,
        strideBytes,
        cfg,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
}

int32_t CHROMA_CALL ChromaRuntime_LocateHBitmap(
    const void* hBitmap,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
#ifdef _WIN32
    if (hBitmap == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"hBitmap is null.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    BITMAP bm{};
    if (GetObjectW(reinterpret_cast<HBITMAP>(const_cast<void*>(hBitmap)), sizeof(BITMAP), &bm) == 0) {
        WriteErrorMessage(outError, outErrorChars, L"GetObjectW failed for HBITMAP.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    const int width = bm.bmWidth;
    const int height = bm.bmHeight < 0 ? -bm.bmHeight : bm.bmHeight;
    if (width <= 0 || height <= 0) {
        WriteErrorMessage(outError, outErrorChars, L"Invalid HBITMAP dimensions.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = width;
    bi.bmiHeader.biHeight = -height; // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    std::vector<uint8_t> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4U);

    HDC hdc = GetDC(nullptr);
    if (hdc == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"GetDC failed.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    const int lines = GetDIBits(
        hdc,
        reinterpret_cast<HBITMAP>(const_cast<void*>(hBitmap)),
        0U,
        static_cast<UINT>(height),
        pixels.data(),
        &bi,
        DIB_RGB_COLORS);
    ReleaseDC(nullptr, hdc);

    if (lines == 0) {
        WriteErrorMessage(outError, outErrorChars, L"GetDIBits failed.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    const vision::ColorPatternConfig cfg = GetActiveConfigCopy();
    return LocateBitmapImpl(
        pixels.data(),
        width,
        height,
        width * 4,
        cfg,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
#else
    (void)hBitmap;
    (void)outPoints;
    (void)outCapacity;
    (void)outTotalFound;
    (void)outWritten;
    WriteErrorMessage(outError, outErrorChars, L"Chroma_LocateHBitmap is only supported on Windows.");
    return CHROMA_STATUS_RUNTIME_ERROR;
#endif
}

int32_t CHROMA_CALL ChromaRuntime_LocateHWND(
    const void* hWnd,
    const int32_t captureClientArea,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    WriteErrorMessage(outError, outErrorChars, L"");
#ifdef _WIN32
    HWND hwnd = reinterpret_cast<HWND>(const_cast<void*>(hWnd));
    if (hwnd == nullptr || !IsWindow(hwnd)) {
        WriteErrorMessage(outError, outErrorChars, L"Invalid HWND.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    RECT rc{};
    BOOL gotRect = captureClientArea ? GetClientRect(hwnd, &rc) : GetWindowRect(hwnd, &rc);
    if (!gotRect) {
        WriteErrorMessage(outError, outErrorChars, L"Failed to query window bounds.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    const int width = rc.right - rc.left;
    const int height = rc.bottom - rc.top;
    if (width <= 0 || height <= 0) {
        WriteErrorMessage(outError, outErrorChars, L"Window bounds are empty.");
        return CHROMA_STATUS_INVALID_ARGUMENT;
    }

    HDC srcDc = captureClientArea ? GetDC(hwnd) : GetWindowDC(hwnd);
    if (srcDc == nullptr) {
        WriteErrorMessage(outError, outErrorChars, L"Failed to get window DC.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    HDC memDc = CreateCompatibleDC(srcDc);
    if (memDc == nullptr) {
        ReleaseDC(hwnd, srcDc);
        WriteErrorMessage(outError, outErrorChars, L"CreateCompatibleDC failed.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = width;
    bi.bmiHeader.biHeight = -height;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP dib = CreateDIBSection(srcDc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (dib == nullptr || bits == nullptr) {
        DeleteDC(memDc);
        ReleaseDC(hwnd, srcDc);
        WriteErrorMessage(outError, outErrorChars, L"CreateDIBSection failed.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    HGDIOBJ oldObj = SelectObject(memDc, dib);
    const UINT printFlags = captureClientArea ? PW_CLIENTONLY : 0U;
    BOOL copied = PrintWindow(hwnd, memDc, printFlags);
    if (!copied) {
        copied = BitBlt(memDc, 0, 0, width, height, srcDc, 0, 0, SRCCOPY | CAPTUREBLT);
    }

    cv::Mat captured;
    if (copied) {
        cv::Mat view(height, width, CV_8UC4, bits, static_cast<size_t>(width) * 4U);
        captured = view.clone();
    }

    SelectObject(memDc, oldObj);
    DeleteObject(dib);
    DeleteDC(memDc);
    ReleaseDC(hwnd, srcDc);

    if (!copied || captured.empty()) {
        WriteErrorMessage(outError, outErrorChars, L"Window capture failed.");
        return CHROMA_STATUS_RUNTIME_ERROR;
    }

    const vision::ColorPatternConfig cfg = GetActiveConfigCopy();
    return LocateBitmapImpl(
        captured.data,
        captured.cols,
        captured.rows,
        static_cast<int32_t>(captured.step),
        cfg,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
#else
    (void)hWnd;
    (void)captureClientArea;
    (void)outPoints;
    (void)outCapacity;
    (void)outTotalFound;
    (void)outWritten;
    WriteErrorMessage(outError, outErrorChars, L"Chroma_LocateHWND is only supported on Windows.");
    return CHROMA_STATUS_RUNTIME_ERROR;
#endif
}






#ifndef CHROMA_RUNTIME_ONLY
CHROMA_API int32_t CHROMA_CALL Chroma_GetApiVersion() {
    return ChromaRuntime_GetApiVersion();
}

CHROMA_API int32_t CHROMA_CALL Chroma_GetConfigStructSize() {
    return ChromaRuntime_GetConfigStructSize();
}

CHROMA_API int32_t CHROMA_CALL Chroma_GetDefaultConfig(
    ChromaConfigV1* outConfig,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_GetDefaultConfig(outConfig, outError, outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_GetActiveConfig(
    ChromaConfigV1* outConfig,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_GetActiveConfig(outConfig, outError, outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_SetActiveConfig(
    const ChromaConfigV1* config,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_SetActiveConfig(config, outError, outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_ResetConfigToDefault(
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_ResetConfigToDefault(outError, outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_LocateBitmapBGRAW(
    const void* bgraPixels,
    const int32_t width,
    const int32_t height,
    const int32_t strideBytes,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_LocateBitmapBGRAW(
        bgraPixels,
        width,
        height,
        strideBytes,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_LocateBitmapWithConfigBGRAW(
    const void* bgraPixels,
    const int32_t width,
    const int32_t height,
    const int32_t strideBytes,
    const ChromaConfigV1* config,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_LocateBitmapWithConfigBGRAW(
        bgraPixels,
        width,
        height,
        strideBytes,
        config,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_LocateBitmapWithDebugBGRAW(
    const void* bgraPixels,
    const int32_t width,
    const int32_t height,
    const int32_t strideBytes,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    ChromaDebugImageV1* outDebugImage,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_LocateBitmapWithDebugBGRAW(
        bgraPixels,
        width,
        height,
        strideBytes,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outDebugImage,
        outError,
        outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_LocateHBitmap(
    const void* hBitmap,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_LocateHBitmap(
        hBitmap,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
}

CHROMA_API int32_t CHROMA_CALL Chroma_LocateHWND(
    const void* hWnd,
    const int32_t captureClientArea,
    ChromaPoint* outPoints,
    const int32_t outCapacity,
    int32_t* outTotalFound,
    int32_t* outWritten,
    wchar_t* outError,
    const int32_t outErrorChars) {
    return ChromaRuntime_LocateHWND(
        hWnd,
        captureClientArea,
        outPoints,
        outCapacity,
        outTotalFound,
        outWritten,
        outError,
        outErrorChars);
}

#endif // CHROMA_RUNTIME_ONLY


