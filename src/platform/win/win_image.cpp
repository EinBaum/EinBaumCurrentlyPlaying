// Windows album-art decode via WIC: decode -> center-crop to square -> scale -> RGBA. Accent
// extraction is shared (core/image.cpp).
#include "core/image.hpp"
#include <windows.h>
#include <wincodec.h>

namespace {
template <class T> void rel(T*& p) { if (p) { p->Release(); p = nullptr; } }
}  // namespace

DecodedImage decodeAlbumArt(const std::vector<std::uint8_t>& bytes, int target) {
    DecodedImage out;
    if (bytes.empty() || target <= 0) return out;

    IWICImagingFactory* factory = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_IWICImagingFactory, reinterpret_cast<void**>(&factory))) || !factory)
        return out;

    IWICStream* stream = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICBitmapClipper* clipper = nullptr;
    IWICBitmapScaler* scaler = nullptr;
    IWICFormatConverter* conv = nullptr;

    [&] {
        if (FAILED(factory->CreateStream(&stream))) return;
        if (FAILED(stream->InitializeFromMemory(const_cast<BYTE*>(bytes.data()),
                                                static_cast<DWORD>(bytes.size())))) return;
        if (FAILED(factory->CreateDecoderFromStream(stream, nullptr,
                                                    WICDecodeMetadataCacheOnLoad, &decoder))) return;
        if (FAILED(decoder->GetFrame(0, &frame))) return;

        UINT w = 0, h = 0;
        frame->GetSize(&w, &h);
        if (w == 0 || h == 0) return;

        UINT side = w < h ? w : h;
        WICRect crop{static_cast<INT>((w - side) / 2), static_cast<INT>((h - side) / 2),
                     static_cast<INT>(side), static_cast<INT>(side)};
        if (FAILED(factory->CreateBitmapClipper(&clipper))) return;
        if (FAILED(clipper->Initialize(frame, &crop))) return;

        if (FAILED(factory->CreateBitmapScaler(&scaler))) return;
        if (FAILED(scaler->Initialize(clipper, static_cast<UINT>(target), static_cast<UINT>(target),
                                      WICBitmapInterpolationModeFant))) return;

        // 32bpp RGBA channel order must match the VK_FORMAT_R8G8B8A8_UNORM texture it feeds
        if (FAILED(factory->CreateFormatConverter(&conv))) return;
        if (FAILED(conv->Initialize(scaler, GUID_WICPixelFormat32bppRGBA,
                                    WICBitmapDitherTypeNone, nullptr, 0.0,
                                    WICBitmapPaletteTypeMedianCut))) return;

        out.w = target;
        out.h = target;
        out.rgba.resize(static_cast<size_t>(target) * target * 4);
        UINT stride = static_cast<UINT>(target) * 4;
        if (FAILED(conv->CopyPixels(nullptr, stride, static_cast<UINT>(out.rgba.size()), out.rgba.data()))) {
            out = DecodedImage{};
        } else {
            computeAccent(out);
        }
    }();

    rel(conv); rel(scaler); rel(clipper); rel(frame); rel(decoder); rel(stream); rel(factory);
    return out;
}
