# Chroma Core

Modular C++ DLL/library for configurable HSV color-pattern detection.

## Repository Layout

- `ChromaCore.sln`: Visual Studio solution.
- `chroma-core/ChromaCore.h`: core detection engine (`vision::ColorPatternFinder`).
- `chroma-core/ChromaApi.h`: stable C ABI for DLL consumers.
- `chroma-core/ChromaCore.cpp`: DLL/API implementation and Win32 capture adapters.
- `chroma-core/MATCHING_GUIDE.md`: pipeline and API guide.

## Build

Visual Studio:

1. Open `ChromaCore.sln`.
2. Select `Release | x64`.
3. Build solution.

MSBuild:

```powershell
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" `
  ChromaCore.sln /t:Build /p:Configuration=Release /p:Platform=x64 /m
```

## Build Outputs

- DLL: `chroma-core/artifacts/x64/Release/bin/ChromaCore.dll`
- Import library: `chroma-core/artifacts/x64/Release/bin/ChromaCore.lib`
- PDB: `chroma-core/artifacts/x64/Release/bin/ChromaCore.pdb`

## API Surface

Primary exported functions are declared in `chroma-core/ChromaApi.h`, including:

- `Chroma_GetApiVersion`
- `Chroma_SetActiveConfig`
- `Chroma_LocateBitmapBGRAW`
- `Chroma_LocateBitmapWithConfigBGRAW`
- `Chroma_LocateBitmapWithDebugBGRAW` (returns optional BGRA debug image)
- `Chroma_LocateHBitmap` (Windows-only)
- `Chroma_LocateHWND` (Windows-only)

## Notes

- For deeper configuration and detection details, see `chroma-core/MATCHING_GUIDE.md`.

