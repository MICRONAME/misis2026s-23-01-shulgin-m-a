#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct DetectionResult {
    std::string name;
    bool success = false;
    std::vector<cv::Point2f> quad;
    double score = 0.0;
};

struct Args {
    std::string imagePath;
    std::string outputDir;
    std::string jsonPath;
    int outWidth = 1240;
    bool debug = true;
};

static void printUsage() {
    std::cout << R"(Usage:
  doc_rectifier --image input.jpg --out output_dir [--points points.json] [--width 1240]

Arguments:
  --image   Path to input image.
  --out     Output directory.
  --points  Optional JSON file with 4 document corner points.
  --width   Output rectified A4 width in pixels. Default: 1240.

Example:
  ./doc_rectifier --image photo.jpg --out result --points points.json --width 1240
)";
}

static bool parseArgs(int argc, char** argv, Args& args) {
    if (argc < 5) {
        printUsage();
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];

        auto needValue = [&](const std::string& name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (key == "--image") {
            auto v = needValue(key);
            if (!v) return false;
            args.imagePath = *v;
        } else if (key == "--out") {
            auto v = needValue(key);
            if (!v) return false;
            args.outputDir = *v;
        } else if (key == "--points") {
            auto v = needValue(key);
            if (!v) return false;
            args.jsonPath = *v;
        } else if (key == "--width") {
            auto v = needValue(key);
            if (!v) return false;
            args.outWidth = std::stoi(*v);
        } else if (key == "--help" || key == "-h") {
            printUsage();
            return false;
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            printUsage();
            return false;
        }
    }

    if (args.imagePath.empty() || args.outputDir.empty()) {
        std::cerr << "--image and --out are required\n";
        return false;
    }

    return true;
}

static void ensureDir(const fs::path& p) {
    fs::create_directories(p);
}

static void saveImage(const fs::path& path, const cv::Mat& img) {
    if (img.empty()) return;
    cv::imwrite(path.string(), img);
}

static cv::Mat resizeForProcessing(
    const cv::Mat& src,
    int maxDim,
    double& scaleToOriginal
) {
    int w = src.cols;
    int h = src.rows;
    int maxSide = std::max(w, h);

    if (maxSide <= maxDim) {
        scaleToOriginal = 1.0;
        return src.clone();
    }

    double scale = static_cast<double>(maxDim) / static_cast<double>(maxSide);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(), scale, scale, cv::INTER_AREA);

    scaleToOriginal = 1.0 / scale;
    return resized;
}

static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts) {
    CV_Assert(pts.size() == 4);

    std::vector<cv::Point2f> ordered(4);

    // top-left has smallest x+y, bottom-right has largest x+y
    // top-right has smallest y-x, bottom-left has largest y-x
    auto sumCmp = [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    };

    auto diffCmp = [](const cv::Point2f& a, const cv::Point2f& b) {
        return (a.y - a.x) < (b.y - b.x);
    };

    ordered[0] = *std::min_element(pts.begin(), pts.end(), sumCmp);   // TL
    ordered[2] = *std::max_element(pts.begin(), pts.end(), sumCmp);   // BR
    ordered[1] = *std::min_element(pts.begin(), pts.end(), diffCmp);  // TR
    ordered[3] = *std::max_element(pts.begin(), pts.end(), diffCmp);  // BL

    return ordered;
}
static double polygonArea(const std::vector<cv::Point2f>& pts) {
    return std::fabs(cv::contourArea(pts));
}

static cv::Mat drawQuad(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& quad,
    const std::string& label
) {
    cv::Mat vis = image.clone();

    if (quad.size() == 4) {
        std::vector<cv::Point> q;
        for (const auto& p : quad) {
            q.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        }

        for (int i = 0; i < 4; ++i) {
            cv::line(vis, q[i], q[(i + 1) % 4], cv::Scalar(0, 255, 0), 4);
            cv::circle(vis, q[i], 8, cv::Scalar(0, 0, 255), -1);
            cv::putText(
                vis,
                std::to_string(i),
                q[i] + cv::Point(10, -10),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(255, 0, 0),
                2
            );
        }
    }

    cv::putText(
        vis,
        label,
        cv::Point(30, 50),
        cv::FONT_HERSHEY_SIMPLEX,
        1.2,
        cv::Scalar(0, 255, 255),
        3
    );

    return vis;
}

static cv::Mat warpToA4(
    const cv::Mat& src,
    const std::vector<cv::Point2f>& quad,
    int outWidth
) {
    CV_Assert(quad.size() == 4);

    // A4 aspect ratio: height / width = sqrt(2)
    int outHeight = static_cast<int>(std::round(outWidth * std::sqrt(2.0)));

    std::vector<cv::Point2f> srcPts = orderPoints(quad);

    std::vector<cv::Point2f> dstPts = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(outWidth - 1), 0),
        cv::Point2f(static_cast<float>(outWidth - 1), static_cast<float>(outHeight - 1)),
        cv::Point2f(0, static_cast<float>(outHeight - 1))
    };

    cv::Mat H = cv::getPerspectiveTransform(srcPts, dstPts);

    cv::Mat warped;
    cv::warpPerspective(
        src,
        warped,
        H,
        cv::Size(outWidth, outHeight),
        cv::INTER_CUBIC,
        cv::BORDER_REPLICATE
    );

    return warped;
}

static cv::Mat enhanceForDocument(const cv::Mat& warped) {
    cv::Mat gray;
    cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);

    cv::Mat denoised;
    cv::bilateralFilter(gray, denoised, 7, 50, 50);

    cv::Mat enhanced;
    cv::adaptiveThreshold(
        denoised,
        enhanced,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY,
        31,
        8
    );

    return enhanced;
}

static std::optional<std::vector<cv::Point2f>> readPointsJson(const std::string& jsonPath) {
    if (jsonPath.empty()) return std::nullopt;

    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        std::cerr << "Cannot open JSON file: " << jsonPath << "\n";
        return std::nullopt;
    }

    json j;
    f >> j;

    if (!j.contains("points") || !j["points"].is_array() || j["points"].size() != 4) {
        std::cerr << "JSON should contain: { \"points\": [[x,y], [x,y], [x,y], [x,y]] }\n";
        return std::nullopt;
    }

    std::vector<cv::Point2f> pts;

    for (const auto& p : j["points"]) {
        if (p.is_array() && p.size() == 2) {
            pts.emplace_back(p[0].get<float>(), p[1].get<float>());
        } else if (p.is_object() && p.contains("x") && p.contains("y")) {
            pts.emplace_back(p["x"].get<float>(), p["y"].get<float>());
        } else {
            std::cerr << "Invalid point format in JSON\n";
            return std::nullopt;
        }
    }

    return orderPoints(pts);
}

static std::optional<std::vector<cv::Point2f>> findBestQuadFromBinary(
    const cv::Mat& binary,
    const cv::Size& imageSize,
    cv::Mat& contourDebug
) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        binary.clone(),
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    contourDebug = cv::Mat::zeros(binary.size(), CV_8UC3);
    cv::drawContours(contourDebug, contours, -1, cv::Scalar(80, 80, 80), 1);

    double imageArea = static_cast<double>(imageSize.width * imageSize.height);
    double bestScore = -1.0;
    std::vector<cv::Point2f> bestQuad;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < imageArea * 0.05) {
            continue;
        }

        std::vector<cv::Point> hull;
        cv::convexHull(c, hull);

        double peri = cv::arcLength(hull, true);

        for (double epsFactor : {0.015, 0.02, 0.025, 0.03, 0.04, 0.05}) {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(hull, approx, epsFactor * peri, true);

            if (approx.size() != 4) {
                continue;
            }

            if (!cv::isContourConvex(approx)) {
                continue;
            }

            std::vector<cv::Point2f> quad;
            for (const auto& p : approx) {
                quad.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
            }
            quad = orderPoints(quad);

            double quadArea = polygonArea(quad);
            if (quadArea < imageArea * 0.10) {
                continue;
            }

            // Простая эвристика:
            // чем больше площадь и чем ближе к разумной форме страницы, тем лучше.
            double areaScore = quadArea / imageArea;

            double wTop = cv::norm(quad[1] - quad[0]);
            double wBot = cv::norm(quad[2] - quad[3]);
            double hLeft = cv::norm(quad[3] - quad[0]);
            double hRight = cv::norm(quad[2] - quad[1]);

            double widthMean = 0.5 * (wTop + wBot);
            double heightMean = 0.5 * (hLeft + hRight);

            if (widthMean < 1.0 || heightMean < 1.0) {
                continue;
            }

            double aspect = heightMean / widthMean;
            double a4Portrait = std::sqrt(2.0);
            double a4Landscape = 1.0 / std::sqrt(2.0);

            double aspectError = std::min(
                std::abs(aspect - a4Portrait),
                std::abs(aspect - a4Landscape)
            );

            double parallelPenalty =
                std::abs(wTop - wBot) / std::max(wTop, wBot) +
                std::abs(hLeft - hRight) / std::max(hLeft, hRight);

            double score = areaScore * 10.0 - aspectError - parallelPenalty * 0.5;

            if (score > bestScore) {
                bestScore = score;
                bestQuad = quad;
            }
        }
    }

    if (bestQuad.empty()) {
        return std::nullopt;
    }

    std::vector<cv::Point> q;
    for (const auto& p : bestQuad) {
        q.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::polylines(contourDebug, q, true, cv::Scalar(0, 255, 0), 3);

    return bestQuad;
}

static DetectionResult detectCannyContours(
    const cv::Mat& srcSmall,
    const fs::path& outDir
) {
    DetectionResult result;
    result.name = "canny_contours";

    fs::path dir = outDir / result.name;
    ensureDir(dir);

    cv::Mat gray;
    cv::cvtColor(srcSmall, gray, cv::COLOR_BGR2GRAY);
    saveImage(dir / "01_gray.png", gray);

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    saveImage(dir / "02_gaussian_blur.png", blur);

    cv::Mat edges;
    cv::Canny(blur, edges, 50, 150);
    saveImage(dir / "03_canny.png", edges);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Mat closed;
    cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    saveImage(dir / "04_closed_edges.png", closed);

    cv::Mat contourDebug;
    auto quad = findBestQuadFromBinary(closed, srcSmall.size(), contourDebug);
    saveImage(dir / "05_contours_debug.png", contourDebug);

    if (quad) {
        result.success = true;
        result.quad = *quad;
        result.score = polygonArea(result.quad);

        cv::Mat overlay = drawQuad(srcSmall, result.quad, result.name);
        saveImage(dir / "06_detected_quad.png", overlay);
    }

    return result;
}

static DetectionResult detectAdaptiveThreshold(
    const cv::Mat& srcSmall,
    const fs::path& outDir
) {
    DetectionResult result;
    result.name = "adaptive_threshold";

    fs::path dir = outDir / result.name;
    ensureDir(dir);
    cv::Mat gray;
    cv::cvtColor(srcSmall, gray, cv::COLOR_BGR2GRAY);
    saveImage(dir / "01_gray.png", gray);

    cv::Mat denoised;
    cv::bilateralFilter(gray, denoised, 9, 75, 75);
    saveImage(dir / "02_bilateral.png", denoised);

    cv::Mat thr;
    cv::adaptiveThreshold(
        denoised,
        thr,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY,
        41,
        7
    );

    // Контур документа часто удобнее искать по инверсии:
    cv::Mat inv;
    cv::bitwise_not(thr, inv);

    saveImage(dir / "03_adaptive_threshold.png", thr);
    saveImage(dir / "04_inverted_threshold.png", inv);

    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

    cv::Mat morphed;
    cv::morphologyEx(inv, morphed, cv::MORPH_CLOSE, kernelClose, cv::Point(-1, -1), 2);
    cv::morphologyEx(morphed, morphed, cv::MORPH_OPEN, kernelOpen, cv::Point(-1, -1), 1);
    saveImage(dir / "05_morphed.png", morphed);

    cv::Mat contourDebug;
    auto quad = findBestQuadFromBinary(morphed, srcSmall.size(), contourDebug);
    saveImage(dir / "06_contours_debug.png", contourDebug);

    if (quad) {
        result.success = true;
        result.quad = *quad;
        result.score = polygonArea(result.quad);

        cv::Mat overlay = drawQuad(srcSmall, result.quad, result.name);
        saveImage(dir / "07_detected_quad.png", overlay);
    }

    return result;
}

static DetectionResult detectGradientMorph(
    const cv::Mat& srcSmall,
    const fs::path& outDir
) {
    DetectionResult result;
    result.name = "gradient_morph";

    fs::path dir = outDir / result.name;
    ensureDir(dir);

    cv::Mat gray;
    cv::cvtColor(srcSmall, gray, cv::COLOR_BGR2GRAY);
    saveImage(dir / "01_gray.png", gray);

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
    saveImage(dir / "02_blur.png", blur);

    cv::Mat gradX, gradY;
    cv::Sobel(blur, gradX, CV_16S, 1, 0, 3);
    cv::Sobel(blur, gradY, CV_16S, 0, 1, 3);

    cv::Mat absX, absY;
    cv::convertScaleAbs(gradX, absX);
    cv::convertScaleAbs(gradY, absY);

    cv::Mat grad;
    cv::addWeighted(absX, 0.5, absY, 0.5, 0, grad);
    saveImage(dir / "03_gradient.png", grad);

    cv::Mat binary;
    cv::threshold(grad, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    saveImage(dir / "04_otsu_gradient_binary.png", binary);

    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

    cv::Mat morphed;
    cv::morphologyEx(binary, morphed, cv::MORPH_CLOSE, kernelClose, cv::Point(-1, -1), 2);
    cv::morphologyEx(morphed, morphed, cv::MORPH_OPEN, kernelOpen, cv::Point(-1, -1), 1);
    saveImage(dir / "05_morphed.png", morphed);

    cv::Mat contourDebug;
    auto quad = findBestQuadFromBinary(morphed, srcSmall.size(), contourDebug);
    saveImage(dir / "06_contours_debug.png", contourDebug);

    if (quad) {
        result.success = true;
        result.quad = *quad;
        result.score = polygonArea(result.quad);

        cv::Mat overlay = drawQuad(srcSmall, result.quad, result.name);
        saveImage(dir / "07_detected_quad.png", overlay);
    }

    return result;
}

static std::vector<cv::Point2f> scaleQuad(
    const std::vector<cv::Point2f>& quad,
    double scale
) {
    std::vector<cv::Point2f> scaled;
    scaled.reserve(quad.size());

    for (const auto& p : quad) {
        scaled.emplace_back(
            static_cast<float>(p.x * scale),
            static_cast<float>(p.y * scale)
        );
    }

    return scaled;
}

static DetectionResult chooseBest(const std::vector<DetectionResult>& results) {
    DetectionResult best;
    best.name = "none";

    for (const auto& r : results) {
        if (!r.success) continue;

        if (!best.success || r.score > best.score) {
            best = r;
        }
    }

    return best;
}
static void writeResultJson(
    const fs::path& path,
    const std::vector<DetectionResult>& results,
    const DetectionResult& best,
    double scaleToOriginal
) {
    json j;

    //j["scale_to_original"] = scaleToOriginal;
    //j["best_method"] = best.name;
    j["methods"] = json::array();

    for (const auto& r : results) {
        json m;
        m["name"] = r.name;
        //m["success"] = r.success;
        //m["score"] = r.score;
        //m["quad_small"] = json::array();
        m["quad_original"] = json::array();

        if (r.success) {
            for (const auto& p : r.quad) {
                //m["quad_small"].push_back({p.x, p.y});
                m["points"].push_back({p.x * scaleToOriginal, p.y * scaleToOriginal});
            }
        }

        j["methods"].push_back(m);
    }

    std::ofstream f(path);
    f << j.dump(2);
}

int main(int argc, char** argv) {
    Args args;

    if (!parseArgs(argc, argv, args)) {
        return 1;
    }

    ensureDir(args.outputDir);

    cv::Mat image = cv::imread(args.imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Cannot read image: " << args.imagePath << "\n";
        return 1;
    }

    fs::path outDir(args.outputDir);
    saveImage(outDir / "00_input.png", image);

    double scaleToOriginal = 1.0;
    cv::Mat small = resizeForProcessing(image, 1600, scaleToOriginal);
    saveImage(outDir / "01_resized_for_detection.png", small);

    std::vector<DetectionResult> results;

    // 1. Manual JSON points
    auto jsonPoints = readPointsJson(args.jsonPath);
    if (jsonPoints) {
        DetectionResult manual;
        manual.name = "manual_json";
        manual.success = true;

        // JSON-точки считаются заданными в координатах исходного изображения.
        // Для сравнения с small переводим их в координаты small.
        double originalToSmall = 1.0 / scaleToOriginal;
        manual.quad = scaleQuad(*jsonPoints, originalToSmall);
        manual.score = polygonArea(manual.quad);

        fs::path dir = outDir / manual.name;
        ensureDir(dir);

        saveImage(dir / "01_detected_quad.png", drawQuad(small, manual.quad, manual.name));

        std::vector<cv::Point2f> originalQuad = *jsonPoints;

        cv::Mat warped = warpToA4(image, originalQuad, args.outWidth);
        saveImage(dir / "02_warped_a4_color.png", warped);

        cv::Mat enhanced = enhanceForDocument(warped);
        saveImage(dir / "03_warped_a4_enhanced_bw.png", enhanced);

        results.push_back(manual);
    }

    // 2. Automatic methods
    results.push_back(detectCannyContours(small, outDir));
    results.push_back(detectAdaptiveThreshold(small, outDir));
    results.push_back(detectGradientMorph(small, outDir));

    // Warp for every successful automatic method
    for (auto& r : results) {
        if (!r.success) {
            std::cout << "[FAIL] " << r.name << "\n";
            continue;
        }

        std::cout << "[OK]   " << r.name << ", score = " << r.score << "\n";

        fs::path dir = outDir / r.name;
        ensureDir(dir);

        std::vector<cv::Point2f> quadOriginal = scaleQuad(r.quad, scaleToOriginal);

        cv::Mat overlayOriginal = drawQuad(image, quadOriginal, r.name + " original");
        saveImage(dir / "08_detected_quad_original.png", overlayOriginal);

        cv::Mat warped = warpToA4(image, quadOriginal, args.outWidth);
        saveImage(dir / "09_warped_a4_color.png", warped);

        cv::Mat enhanced = enhanceForDocument(warped);
        saveImage(dir / "10_warped_a4_enhanced_bw.png", enhanced);
    }

    DetectionResult best = chooseBest(results);

    if (best.success) {
        std::vector<cv::Point2f> bestOriginal = scaleQuad(best.quad, scaleToOriginal);

        cv::Mat bestOverlay = drawQuad(image, bestOriginal, "BEST: " + best.name);
        saveImage(outDir / "best_detected_quad.png", bestOverlay);

        cv::Mat bestWarped = warpToA4(image, bestOriginal, args.outWidth);
        saveImage(outDir / "best_warped_a4_color.png", bestWarped);
        cv::Mat bestEnhanced = enhanceForDocument(bestWarped);
        saveImage(outDir / "best_warped_a4_enhanced_bw.png", bestEnhanced);

        std::cout << "Best method: " << best.name << "\n";
    } else {
        std::cerr << "No document quad found.\n";
    }

    writeResultJson(outDir / "results.json", results, best, scaleToOriginal);

    std::cout << "Output saved to: " << outDir << "\n";

    return best.success ? 0 : 2;
}
