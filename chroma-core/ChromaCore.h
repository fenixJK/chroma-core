#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vision {

struct HueRange {
    int minHue = 0;   // OpenCV hue space: 0..179
    int maxHue = 179;
};

struct ChannelRange {
    int minValue = 0;
    int maxValue = 255;
};

class HueRangeSet {
public:
    HueRangeSet() = default;
    HueRangeSet(std::initializer_list<HueRange> ranges) {
        for (const HueRange& r : ranges) {
            Add(r);
        }
    }

    explicit HueRangeSet(const std::vector<HueRange>& ranges) {
        for (const HueRange& r : ranges) {
            Add(r);
        }
    }

    void Add(HueRange range) {
        range.minHue = ClampHue(range.minHue);
        range.maxHue = ClampHue(range.maxHue);
        ranges_.push_back(range);
    }

    void Clear() {
        ranges_.clear();
    }

    bool Empty() const {
        return ranges_.empty();
    }

    const std::vector<HueRange>& Ranges() const {
        return ranges_;
    }

    cv::Mat BuildMask(
        const cv::Mat& hsv,
        int satMin,
        int satMax,
        int valMin,
        int valMax) const {
        if (hsv.empty()) {
            return {};
        }
        if (hsv.type() != CV_8UC3) {
            throw std::invalid_argument("HueRangeSet::BuildMask expects CV_8UC3 HSV image.");
        }
        if (ranges_.empty()) {
            return cv::Mat::zeros(hsv.size(), CV_8U);
        }

        satMin = std::clamp(satMin, 0, 255);
        satMax = std::clamp(satMax, 0, 255);
        valMin = std::clamp(valMin, 0, 255);
        valMax = std::clamp(valMax, 0, 255);
        if (satMin > satMax) {
            std::swap(satMin, satMax);
        }
        if (valMin > valMax) {
            std::swap(valMin, valMax);
        }

        cv::Mat accumulated = cv::Mat::zeros(hsv.size(), CV_8U);
        for (const HueRange& r : ranges_) {
            cv::Mat part;
            if (r.minHue <= r.maxHue) {
                cv::inRange(hsv, cv::Scalar(r.minHue, satMin, valMin), cv::Scalar(r.maxHue, satMax, valMax), part);
            } else {
                cv::Mat a;
                cv::Mat b;
                cv::inRange(hsv, cv::Scalar(0, satMin, valMin), cv::Scalar(r.maxHue, satMax, valMax), a);
                cv::inRange(hsv, cv::Scalar(r.minHue, satMin, valMin), cv::Scalar(179, satMax, valMax), b);
                cv::bitwise_or(a, b, part);
            }
            cv::bitwise_or(accumulated, part, accumulated);
        }
        return accumulated;
    }

private:
    static int ClampHue(int h) {
        if (h < 0) {
            return 0;
        }
        if (h > 179) {
            return 179;
        }
        return h;
    }

    std::vector<HueRange> ranges_;
};

struct ColorMaskConfig {
    HueRangeSet hues;
    ChannelRange satRange{ 0, 255 };
    ChannelRange valRange{ 0, 255 };
};

struct MorphologyConfig {
    int openIterations = 0;
    int closeIterations = 0;
    int dilateIterations = 0;
};

struct ShapeFilterConfig {
    int minArea = 10;
    int maxArea = 5000;
    float minCircularity = 0.65F;
    float minFillRatio = 0.40F;
};

struct ContextRingConfig {
    bool enabled = false;

    int innerRadiusPercent = 110; // ring start = centerRadius * percent / 100
    int outerRadiusPercent = 220;

    ColorMaskConfig supportColor;

    HueRangeSet excludeHues;
    ChannelRange excludeSatRange{ 0, 255 };
    ChannelRange excludeValRange{ 0, 255 };

    float minSupportRatio = 0.20F;
};

struct DebugDrawConfig {
    bool drawRejected = false;
    bool drawLabels = true;
    bool drawLabelBackground = true;

    cv::Scalar acceptedColor = cv::Scalar(0, 255, 0);
    cv::Scalar rejectedColor = cv::Scalar(0, 0, 255);
    cv::Scalar textColor = cv::Scalar(0, 255, 0);
    cv::Scalar labelBgColor = cv::Scalar(0, 0, 0);

    double fontScale = 0.45;
    int lineThickness = 1;
    int labelPaddingPx = 2;
};

struct ColorPatternConfig {
    ColorMaskConfig centerColor;
    MorphologyConfig centerMorph;
    ShapeFilterConfig shape;
    ContextRingConfig context;
    DebugDrawConfig debug;
};

struct DetectionMetrics {
    float areaPx = 0.0F;
    float circularity = 0.0F;
    float centerFillRatio = 0.0F;
    float ringSupportRatio = 0.0F;
    float score = 0.0F;

    bool passesArea = false;
    bool passesCircularity = false;
    bool passesCenterFill = false;
    bool passesContext = false;
    bool accepted = false;
};

struct ColorPatternDetection {
    cv::Rect boxPx;
    cv::Point centerPx;
    float radiusPx = 0.0F;
    std::vector<cv::Point> contour;
    DetectionMetrics metrics;
};

struct ColorPatternRunResult {
    std::vector<ColorPatternDetection> detections;
    std::vector<cv::Point> acceptedCentersPx;
    std::vector<cv::Rect> acceptedBoxesPx;

    int rawCandidateCount = 0;
    int acceptedCount = 0;
    float acceptedRatio = 0.0F;
    float sceneMaskCoverage = 0.0F;
    float score = 0.0F;

    cv::Mat debugOverlay;  // color view with boxes/labels
    cv::Mat debugMask;     // mask view with boxes/labels
    cv::Mat sideBySideDebug;
};

namespace detail {

inline float SafeDiv(float num, float den) {
    return den <= 0.0F ? 0.0F : (num / den);
}

inline float Clamp01(float v) {
    if (v < 0.0F) {
        return 0.0F;
    }
    if (v > 1.0F) {
        return 1.0F;
    }
    return v;
}

inline cv::Mat BuildMask(const cv::Mat& hsv, const ColorMaskConfig& cfg) {
    return cfg.hues.BuildMask(
        hsv,
        cfg.satRange.minValue,
        cfg.satRange.maxValue,
        cfg.valRange.minValue,
        cfg.valRange.maxValue);
}

inline cv::Mat BuildExcludeMask(
    const cv::Mat& hsv,
    const HueRangeSet& ranges,
    const ChannelRange& satRange,
    const ChannelRange& valRange) {
    return ranges.BuildMask(hsv, satRange.minValue, satRange.maxValue, valRange.minValue, valRange.maxValue);
}

inline void ApplyMorphology(cv::Mat& mask, const MorphologyConfig& cfg) {
    if (mask.empty()) {
        return;
    }
    const cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    if (cfg.openIterations > 0) {
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k3, cv::Point(-1, -1), cfg.openIterations);
    }
    if (cfg.closeIterations > 0) {
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k3, cv::Point(-1, -1), cfg.closeIterations);
    }
    if (cfg.dilateIterations > 0) {
        cv::dilate(mask, mask, k3, cv::Point(-1, -1), cfg.dilateIterations);
    }
}

inline float ComputeCircularity(const std::vector<cv::Point>& contour) {
    const float area = static_cast<float>(cv::contourArea(contour));
    const float perimeter = static_cast<float>(cv::arcLength(contour, true));
    if (area <= 0.0F || perimeter <= 0.0F) {
        return 0.0F;
    }
    return static_cast<float>((4.0 * CV_PI * area) / (perimeter * perimeter));
}

inline cv::Mat EnsureColor(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    if (image.channels() == 3) {
        return image.clone();
    }
    if (image.channels() == 1) {
        cv::Mat out;
        cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
        return out;
    }
    cv::Mat out;
    cv::cvtColor(image, out, cv::COLOR_BGRA2BGR);
    return out;
}

inline void DrawLabel(
    cv::Mat& image,
    const std::string& text,
    const cv::Point& anchor,
    const DebugDrawConfig& dbg) {
    if (image.empty() || text.empty() || !dbg.drawLabels) {
        return;
    }
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, dbg.fontScale, dbg.lineThickness, &baseline);

    int x = std::max(0, anchor.x);
    int y = std::max(textSize.height + 1, anchor.y);
    if (x + textSize.width + 2 >= image.cols) {
        x = std::max(0, image.cols - textSize.width - 2);
    }
    if (y >= image.rows) {
        y = std::max(textSize.height + 1, image.rows - 2);
    }

    if (dbg.drawLabelBackground) {
        const cv::Point tl(std::max(0, x - dbg.labelPaddingPx), std::max(0, y - textSize.height - dbg.labelPaddingPx));
        const cv::Point br(std::min(image.cols - 1, x + textSize.width + dbg.labelPaddingPx), std::min(image.rows - 1, y + baseline + dbg.labelPaddingPx));
        cv::rectangle(image, tl, br, dbg.labelBgColor, cv::FILLED);
    }
    cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, dbg.fontScale, dbg.textColor, dbg.lineThickness, cv::LINE_AA);
}

inline std::string BuildMetricLabel(const DetectionMetrics& m) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << (m.accepted ? "A" : "R")
        << " rr=" << m.ringSupportRatio
        << " c=" << m.circularity
        << " f=" << m.centerFillRatio;
    return oss.str();
}

inline cv::Mat BuildSideBySide(const cv::Mat& leftBgr, const cv::Mat& rightBgr) {
    if (leftBgr.empty() || rightBgr.empty()) {
        return {};
    }
    cv::Mat left = EnsureColor(leftBgr);
    cv::Mat right = EnsureColor(rightBgr);
    if (left.rows != right.rows) {
        const int targetRows = std::max(left.rows, right.rows);
        const double leftScale = static_cast<double>(targetRows) / static_cast<double>(left.rows);
        const double rightScale = static_cast<double>(targetRows) / static_cast<double>(right.rows);
        cv::resize(left, left, cv::Size(), leftScale, leftScale, cv::INTER_NEAREST);
        cv::resize(right, right, cv::Size(), rightScale, rightScale, cv::INTER_NEAREST);
    }
    cv::Mat out;
    cv::hconcat(left, right, out);
    return out;
}

}

class ColorPatternFinder {
public:
    explicit ColorPatternFinder(ColorPatternConfig config = {}) : config_(std::move(config)) {}

    static bool ValidateConfig(const ColorPatternConfig& cfg, std::string* errorOut = nullptr) {
        auto setError = [&](const std::string& msg) {
            if (errorOut != nullptr) {
                *errorOut = msg;
            }
            return false;
        };
        auto validateRange = [&](const ChannelRange& range, const std::string& name) {
            if (range.minValue < 0 || range.minValue > 255 || range.maxValue < 0 || range.maxValue > 255) {
                return setError(name + " must be within [0,255].");
            }
            if (range.minValue > range.maxValue) {
                return setError(name + " must satisfy minValue <= maxValue.");
            }
            return true;
        };

        if (cfg.centerColor.hues.Empty()) {
            return setError("centerColor.hues is empty.");
        }
        if (!validateRange(cfg.centerColor.satRange, "centerColor.satRange")) {
            return false;
        }
        if (!validateRange(cfg.centerColor.valRange, "centerColor.valRange")) {
            return false;
        }
        if (cfg.shape.minArea < 1) {
            return setError("shape.minArea must be >= 1.");
        }
        if (cfg.shape.maxArea < cfg.shape.minArea) {
            return setError("shape.maxArea must be >= shape.minArea.");
        }
        if (cfg.shape.minCircularity < 0.0F || cfg.shape.minCircularity > 1.0F) {
            return setError("shape.minCircularity must be in [0,1].");
        }
        if (cfg.shape.minFillRatio < 0.0F || cfg.shape.minFillRatio > 1.0F) {
            return setError("shape.minFillRatio must be in [0,1].");
        }
        if (cfg.context.enabled) {
            if (cfg.context.innerRadiusPercent < 1 || cfg.context.outerRadiusPercent <= cfg.context.innerRadiusPercent) {
                return setError("context ring radius percents must satisfy: 1 <= inner < outer.");
            }
            if (cfg.context.minSupportRatio < 0.0F || cfg.context.minSupportRatio > 1.0F) {
                return setError("context.minSupportRatio must be in [0,1].");
            }
            if (!validateRange(cfg.context.supportColor.satRange, "context.supportColor.satRange")) {
                return false;
            }
            if (!validateRange(cfg.context.supportColor.valRange, "context.supportColor.valRange")) {
                return false;
            }
            if (!validateRange(cfg.context.excludeSatRange, "context.excludeSatRange")) {
                return false;
            }
            if (!validateRange(cfg.context.excludeValRange, "context.excludeValRange")) {
                return false;
            }
        }

        return true;
    }

    ColorPatternRunResult LoadAndFind(const std::string& scenePath) const {
        const cv::Mat scene = cv::imread(scenePath, cv::IMREAD_COLOR);
        if (scene.empty()) {
            throw std::runtime_error("Failed to load scene image: " + scenePath);
        }
        return Find(scene);
    }

    ColorPatternRunResult Find(const cv::Mat& sceneBgr) const {
        if (sceneBgr.empty()) {
            throw std::invalid_argument("Find received empty scene image.");
        }

        cv::Mat scene = detail::EnsureColor(sceneBgr);
        cv::Mat hsv;
        cv::cvtColor(scene, hsv, cv::COLOR_BGR2HSV);

        cv::Mat centerMask = detail::BuildMask(hsv, config_.centerColor);
        detail::ApplyMorphology(centerMask, config_.centerMorph);

        cv::Mat supportMask;
        cv::Mat excludeMask;
        if (config_.context.enabled) {
            supportMask = detail::BuildMask(hsv, config_.context.supportColor);
            if (!config_.context.excludeHues.Empty()) {
                excludeMask = detail::BuildExcludeMask(
                    hsv,
                    config_.context.excludeHues,
                    config_.context.excludeSatRange,
                    config_.context.excludeValRange);
            }
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(centerMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        ColorPatternRunResult result;
        result.rawCandidateCount = static_cast<int>(contours.size());
        result.sceneMaskCoverage = detail::SafeDiv(
            static_cast<float>(cv::countNonZero(centerMask)),
            static_cast<float>(centerMask.rows * centerMask.cols));

        cv::Mat overlay = scene.clone();
        cv::Mat maskDebug;
        cv::cvtColor(centerMask, maskDebug, cv::COLOR_GRAY2BGR);

        for (const std::vector<cv::Point>& contour : contours) {
            const float area = static_cast<float>(cv::contourArea(contour));
            if (area <= 0.0F) {
                continue;
            }

            cv::Point2f centerFloat;
            float radius = 0.0F;
            cv::minEnclosingCircle(contour, centerFloat, radius);
            const cv::Point center(static_cast<int>(std::lround(centerFloat.x)), static_cast<int>(std::lround(centerFloat.y)));
            const cv::Rect box = cv::boundingRect(contour);

            DetectionMetrics m;
            m.areaPx = area;
            m.circularity = detail::Clamp01(detail::ComputeCircularity(contour));
            m.passesArea = (area >= static_cast<float>(config_.shape.minArea) && area <= static_cast<float>(config_.shape.maxArea));
            m.passesCircularity = (m.circularity >= config_.shape.minCircularity);

            const float circleArea = std::max(1.0F, static_cast<float>(CV_PI) * radius * radius);
            m.centerFillRatio = detail::Clamp01(detail::SafeDiv(area, circleArea));
            m.passesCenterFill = (m.centerFillRatio >= config_.shape.minFillRatio);

            if (config_.context.enabled) {
                cv::Mat ringMask = cv::Mat::zeros(centerMask.size(), CV_8U);
                const int inner = std::max(1, static_cast<int>(std::lround(radius * (static_cast<float>(config_.context.innerRadiusPercent) / 100.0F))));
                const int outer = std::max(inner + 1, static_cast<int>(std::lround(radius * (static_cast<float>(config_.context.outerRadiusPercent) / 100.0F))));

                cv::circle(ringMask, center, outer, cv::Scalar(255), cv::FILLED);
                cv::circle(ringMask, center, inner, cv::Scalar(0), cv::FILLED);

                cv::Mat validRingMask = ringMask.clone();
                if (!excludeMask.empty()) {
                    cv::Mat excludedInRing;
                    cv::bitwise_and(ringMask, excludeMask, excludedInRing);
                    cv::bitwise_xor(validRingMask, excludedInRing, validRingMask);
                }

                cv::Mat supportInRing;
                cv::bitwise_and(supportMask, validRingMask, supportInRing);

                const float validPx = static_cast<float>(cv::countNonZero(validRingMask));
                const float supportPx = static_cast<float>(cv::countNonZero(supportInRing));
                m.ringSupportRatio = detail::Clamp01(detail::SafeDiv(supportPx, validPx));
                m.passesContext = (m.ringSupportRatio >= config_.context.minSupportRatio);
            } else {
                m.ringSupportRatio = 1.0F;
                m.passesContext = true;
            }

            const float shapeScore = (m.circularity * 0.55F) + (m.centerFillRatio * 0.45F);
            if (config_.context.enabled) {
                m.score = detail::Clamp01((shapeScore * 0.60F) + (m.ringSupportRatio * 0.40F));
            } else {
                m.score = detail::Clamp01(shapeScore);
            }

            m.accepted = (m.passesArea && m.passesCircularity && m.passesCenterFill && m.passesContext);

            ColorPatternDetection det;
            det.boxPx = box;
            det.centerPx = center;
            det.radiusPx = radius;
            det.contour = contour;
            det.metrics = m;
            result.detections.push_back(std::move(det));
        }

        std::sort(result.detections.begin(), result.detections.end(),
            [](const ColorPatternDetection& a, const ColorPatternDetection& b) {
                if (a.metrics.accepted != b.metrics.accepted) {
                    return a.metrics.accepted > b.metrics.accepted;
                }
                return a.metrics.score > b.metrics.score;
            });

        for (const ColorPatternDetection& det : result.detections) {
            if (det.metrics.accepted) {
                result.acceptedCentersPx.push_back(det.centerPx);
                result.acceptedBoxesPx.push_back(det.boxPx);
                result.acceptedCount += 1;
                result.score = std::max(result.score, det.metrics.score);
            }

            if (det.metrics.accepted || config_.debug.drawRejected) {
                const cv::Scalar stroke = det.metrics.accepted ? config_.debug.acceptedColor : config_.debug.rejectedColor;
                cv::rectangle(overlay, det.boxPx, stroke, 2, cv::LINE_AA);
                cv::circle(overlay, det.centerPx, std::max(2, static_cast<int>(std::lround(det.radiusPx))), stroke, 1, cv::LINE_AA);

                cv::rectangle(maskDebug, det.boxPx, stroke, 2, cv::LINE_AA);
                cv::circle(maskDebug, det.centerPx, std::max(2, static_cast<int>(std::lround(det.radiusPx))), stroke, 1, cv::LINE_AA);

                const std::string label = detail::BuildMetricLabel(det.metrics);
                const cv::Point labelPoint(det.boxPx.x, std::max(12, det.boxPx.y - 4));
                detail::DrawLabel(overlay, label, labelPoint, config_.debug);
                detail::DrawLabel(maskDebug, label, labelPoint, config_.debug);
            }
        }

        result.acceptedRatio = detail::SafeDiv(static_cast<float>(result.acceptedCount), static_cast<float>(std::max(1, result.rawCandidateCount)));
        result.debugOverlay = overlay;
        result.debugMask = maskDebug;
        result.sideBySideDebug = detail::BuildSideBySide(result.debugOverlay, result.debugMask);
        return result;
    }

private:
    ColorPatternConfig config_;
};

}
