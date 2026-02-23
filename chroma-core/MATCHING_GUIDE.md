# Chroma Core DLL Guide

## Architecture

- `ChromaCore.h`: Core detection pipeline (`vision::ColorPatternFinder`) for general C++ use.
- `ChromaCore.cpp`: C ABI exports in `ChromaApi.h`, config marshaling, validation, and Windows capture adapters (`HBITMAP` / `HWND`).
- `ChromaApi.h`: Stable DLL surface designed for native callers and script wrappers.

## Detection Pipeline

`vision::ColorPatternFinder` is the reusable engine.

Main config groups:

- `centerColor`: center hue/saturation/value mask.
- `centerMorph`: mask cleanup (`open`, `close`, `dilate`).
- `shape`: geometric constraints (`area`, `circularity`, `fill ratio`).
- `context`: optional support/exclusion ring scoring.
- `debug`: overlay and label drawing controls.

Main outputs:

- `acceptedCentersPx`
- `acceptedBoxesPx`
- `detections` with full metrics
- `debugOverlay`
- `debugMask`
- `sideBySideDebug`

## DLL API Contract

Runtime config:

- `Chroma_GetDefaultConfig`
- `Chroma_GetActiveConfig`
- `Chroma_SetActiveConfig`
- `Chroma_ResetConfigToDefault`

Detection entry points:

- `Chroma_LocateBitmapBGRAW`
- `Chroma_LocateBitmapWithConfigBGRAW`
- `Chroma_LocateBitmapWithDebugBGRAW` (optional debug-image output)
- `Chroma_LocateHBitmap` (Windows-only)
- `Chroma_LocateHWND` (Windows-only)

Bitmap buffer rules:

- Pixel format is BGRA 8:8:8:8.
- `strideBytes` may be positive (top-down) or negative (bottom-up).
- `abs(strideBytes)` must be at least `width * 4`.
- `outPoints` may be null for count-only calls.

Status codes are returned as `ChromaStatusCode` values; detailed error text is written into `outError` when provided.

## Minimal Example

```cpp
#include "ChromaCore.h"

vision::ColorPatternConfig cfg;
cfg.centerColor.hues = vision::HueRangeSet({ {20, 32} });
cfg.centerColor.satRange = { 50, 255 };
cfg.centerColor.valRange = { 85, 255 };
cfg.shape.minArea = 20;
cfg.shape.maxArea = 800;
cfg.shape.minCircularity = 0.75f;

vision::ColorPatternFinder finder(cfg);
vision::ColorPatternRunResult out = finder.LoadAndFind(R"(C:\path\scene.png)");
```


