/**
 * ============================================================
 *  Document Corner Detector — OpenCV C++ Pipeline
 * ============================================================
 *  Author  : Expert CV Pipeline
 *  OpenCV  : 4.x
 *  Compile : g++ document_corner_detector.cpp -o detector \
 *              `pkg-config --cflags --libs opencv4` -std=c++17
 *
 *  Usage   : ./detector <image_path>
 *            ./detector                 (uses built-in synthetic test)
 * ============================================================
 */

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────
//  Utility: display or save an intermediate result
//  In headless environments (CI, servers), define SAVE_INTERMEDIATES
//  to write PNGs instead of calling imshow.
// ─────────────────────────────────────────────────────────────
static int g_stepCounter = 0;

void showStep(const std::string& title, const cv::Mat& img, bool save = false)
{
    ++g_stepCounter;
    std::string label = "Step " + std::to_string(g_stepCounter) + ": " + title;
    std::cout << "[PIPELINE] " << label << "\n";

    if (save) {
        std::string filename = "step_" + std::to_string(g_stepCounter) + "_" + title + ".png";
        // Replace spaces with underscores for safe filenames
        std::replace(filename.begin(), filename.end(), ' ', '_');
        cv::imwrite(filename, img);
        std::cout << "           Saved → " << filename << "\n";
    } else {
        cv::imshow(label, img);
        cv::waitKey(0);   // Press any key to advance to next step
    }
}

// ─────────────────────────────────────────────────────────────
//  Utility: Sort 4 points into [TL, TR, BR, BL] order.
//
//  Strategy:
//    • Top-Left     → smallest  (x + y)   sum
//    • Bottom-Right → largest   (x + y)   sum
//    • Top-Right    → smallest  (y - x)   difference
//    • Bottom-Left  → largest   (y - x)   difference
//
//  This works because in image coordinates (y increases downward):
//    TL has the smallest  sum, BR the largest  sum.
//    TR has the most negative difference, BL the most positive.
// ─────────────────────────────────────────────────────────────
std::vector<cv::Point2f> sortCorners(const std::vector<cv::Point2f>& pts)
{
    std::vector<cv::Point2f> sorted(4);

    std::vector<float> sums(4), diffs(4);
    for (int i = 0; i < 4; ++i) {
        sums[i]  = pts[i].x + pts[i].y;
        diffs[i] = pts[i].y - pts[i].x;
    }

    // Indices of min/max sum and diff
    auto minSumIdx  = std::min_element(sums.begin(),  sums.end())  - sums.begin();
    auto maxSumIdx  = std::max_element(sums.begin(),  sums.end())  - sums.begin();
    auto minDiffIdx = std::min_element(diffs.begin(), diffs.end()) - diffs.begin();
    auto maxDiffIdx = std::max_element(diffs.begin(), diffs.end()) - diffs.begin();

    sorted[0] = pts[minSumIdx];   // Top-Left
    sorted[1] = pts[minDiffIdx];  // Top-Right
    sorted[2] = pts[maxSumIdx];   // Bottom-Right
    sorted[3] = pts[maxDiffIdx];  // Bottom-Left

    return sorted;
}



// ─────────────────────────────────────────────────────────────
//  Core pipeline function
//  Returns the 4 sorted corners, or empty vector on failure.
// ─────────────────────────────────────────────────────────────
std::vector<cv::Point2f> detectDocumentCorners(
    const cv::Mat& colorImage,
    bool           saveIntermediates = false)
{
    CV_Assert(!colorImage.empty() && colorImage.channels() == 3);

    const int    W = colorImage.cols;
    const int    H = colorImage.rows;
    const double imageArea = static_cast<double>(W) * H;


    // ══════════════════════════════════════════════════════════
    //  STEP 1 — Preprocessing
    //  Goal: produce a clean, noise-reduced grayscale image
    //        that preserves hard edges (document boundaries).
    //
    //  Why BilateralFilter over GaussianBlur?
    //    GaussianBlur smooths everything uniformly, blurring
    //    edges along with noise.  BilateralFilter is an
    //    edge-preserving filter: it averages nearby pixels only
    //    if they are BOTH spatially close AND have similar
    //    intensity.  This keeps the sharp contrast between the
    //    document and background, which is critical for the
    //    subsequent binarization step.
    //
    //  Parameters chosen:
    //    d=9        : diameter of the pixel neighbourhood
    //    sigmaColor=75 : controls how different intensities
    //                    can be while still being blended
    //    sigmaSpace=75 : controls spatial reach (larger →
    //                    smoother but slower)
    //    A pass of GaussianBlur after bilateral is optional but
    //    helps when the image has strong JPEG compression
    //    artefacts that bilateral alone doesn't fully suppress.
    // ══════════════════════════════════════════════════════════

    cv::Mat gray;
    cv::cvtColor(colorImage, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::bilateralFilter(gray, blurred,
        /*d=*/         9,
        /*sigmaColor=*/75,
        /*sigmaSpace=*/75);

    // Optional second-pass Gaussian to iron out JPEG artifacts
    cv::GaussianBlur(blurred, blurred, cv::Size(5, 5), 0);

    showStep("Blurred_Grayscale", blurred, saveIntermediates);

    // ══════════════════════════════════════════════════════════
    //  STEP 2 — Local Binarization + Morphological Closing
    //
    //  Why adaptiveThreshold (not Otsu / global)?
    //    Global thresholding computes ONE threshold for the
    //    entire image.  With uneven lighting (one corner bright,
    //    another in shadow) a single threshold will either clip
    //    the bright region or flood the dark region.
    //    adaptiveThreshold computes a LOCAL threshold for each
    //    pixel based on the mean (or Gaussian-weighted mean) of
    //    a surrounding block.  This compensates for any
    //    spatially varying illumination gradient.
    //
    //  Key parameters:
    //    ADAPTIVE_THRESH_GAUSSIAN_C — uses Gaussian-weighted
    //      block mean (smoother, less salt-and-pepper vs MEAN_C)
    //    THRESH_BINARY_INV — invert so the document edges
    //      appear WHITE on BLACK (easier for findContours)
    //    blockSize=21 — neighbourhood size. Must be ODD.
    //      Larger blocks → more robust to gradients but may
    //      merge nearby features.  For a document-sized object,
    //      21 is a good starting point.  Scale with image
    //      resolution: blockSize ≈ max(11, image_shorter_side/50)
    //    C=10 — constant subtracted from mean.  Positive C
    //      raises the effective threshold, suppressing very
    //      faint texture inside the document.
    //
    //  Morphological Closing (dilate then erode):
    //    After binarization, the document border may have small
    //    gaps where shadows or text break the edge.  Closing
    //    bridges those gaps so the contour forms a complete loop.
    //    Kernel size 5×5 is aggressive enough to close typical
    //    gaps without merging the document edge with the border
    //    of the image.
    // ══════════════════════════════════════════════════════════

    // Scale blockSize with image resolution so the pipeline
    // generalises from 640×480 thumbnails to 4K scans.
    int blockSize = std::max(11, static_cast<int>(std::min(W, H) / 50));
    if (blockSize % 2 == 0) ++blockSize;   // must be odd

    cv::Mat binary;
    cv::adaptiveThreshold(blurred, binary,
        /*maxValue=*/       255,
        /*adaptiveMethod=*/ cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        /*thresholdType=*/  cv::THRESH_BINARY_INV,
        /*blockSize=*/      blockSize,
        /*C=*/              10);


    // Morphological closing: close edge gaps
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel,
        cv::Point(-1, -1), /*iterations=*/2);

    // Also dilate slightly to thicken thin edge lines
    cv::dilate(binary, binary, kernel, cv::Point(-1, -1),
        /*iterations=*/1);

    showStep("Adaptive_Binary_Mask", binary, saveIntermediates);

    // ══════════════════════════════════════════════════════════
    //  STEP 3 — Contour Detection and Area-Based Filtering
    //
    //  findContours with RETR_LIST retrieves ALL contours
    //  without hierarchy — we don't need parent/child
    //  relationships; we only care about the largest one.
    //  CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and
    //  diagonal segments, storing only their endpoints and
    //  saving memory.
    //
    //  Filtering strategy:
    //    1. Reject contours whose bounding-box area is less than
    //       15% of the total image area — these are noise or
    //       small objects.
    //    2. From the survivors, keep only the LARGEST by
    //       contour area (cv::contourArea).
    //    3. Optionally reject convex-hull solidity < 0.5 to
    //       eliminate highly irregular shapes (text blocks, etc.)
    // ══════════════════════════════════════════════════════════

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours,
        cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "[STEP 3]  Total contours found: " << contours.size() << "\n";

    // Filter: keep contours with area ≥ 15% of image area
    const double minArea = 0.15 * imageArea;

    std::vector<std::vector<cv::Point>> candidates;
    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area >= minArea) {
            candidates.push_back(c);
        }
    }

    std::cout << "[STEP 3]  Candidates after area filter: "
              << candidates.size() << "\n";

    if (candidates.empty()) {
        std::cerr << "[ERROR] No large contours found. "
                     "Check image quality or lower minArea threshold.\n";
        return {};
    }

    // Keep only the single largest contour
    auto largest = std::max_element(candidates.begin(), candidates.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    std::vector<std::vector<cv::Point>> largestVec = {*largest};

    // Visualise: draw on a dark copy of the original
    cv::Mat contourVis = colorImage.clone();
    cv::drawContours(contourVis, largestVec, 0,
        cv::Scalar(0, 255, 0), /*thickness=*/3);

    showStep("Largest_Contour", contourVis, saveIntermediates);


    // ══════════════════════════════════════════════════════════
    //  STEP 4 — Polygon Approximation (approxPolyDP)
    //
    //  approxPolyDP uses the Ramer–Douglas–Peucker algorithm:
    //  it recursively simplifies a curve by removing points
    //  that deviate less than EPSILON from the simplified line.
    //
    //  Epsilon selection:
    //    epsilon = factor × arcLength(contour, closed=true)
    //    A factor of ~0.02 (2%) is a common heuristic.
    //    However, complex/noisy contours may need a higher
    //    factor before collapsing to 4 vertices, so we use
    //    a DYNAMIC SEARCH:
    //      • Start at factor = 0.01 (very fine approximation)
    //      • Increment by 0.01 each iteration
    //      • Stop as soon as we get exactly 4 vertices
    //      • Cap at factor = 0.35 to avoid over-simplification
    //    This makes the pipeline self-tuning across different
    //    document shapes and contour noisiness.
    //
    //  We also attempt the convex hull as a fallback:
    //    If the document edge is occluded or crinkled, the raw
    //    contour may not simplify cleanly to 4 points.  The
    //    convex hull removes concavities, making polygon
    //    approximation more stable.
    // ══════════════════════════════════════════════════════════

    std::vector<cv::Point> docContour = *largest;
    std::vector<cv::Point2f> approxPoly;
    bool found = false;

    double arcLen = cv::arcLength(docContour, true);

    // ── Attempt 1: approximate directly ──────────────────────
    for (double factor = 0.01; factor <= 0.35; factor += 0.01)
    {
        std::vector<cv::Point> poly;
        cv::approxPolyDP(docContour, poly, factor * arcLen, /*closed=*/true);

        if (poly.size() == 4) {
            for (auto& p : poly)
                approxPoly.push_back(cv::Point2f(
                    static_cast<float>(p.x),
                    static_cast<float>(p.y)));
            std::cout << "[STEP 4]  4-point polygon found at factor="
                      << factor << "\n";
            found = true;
            break;
        }
    }

    // ── Attempt 2: use convex hull, then approximate ──────────
    if (!found) {
        std::cout << "[STEP 4]  Direct approx failed. "
                     "Trying convex hull...\n";
        std::vector<cv::Point> hull;
        cv::convexHull(docContour, hull);
        double hullArcLen = cv::arcLength(hull, true);

        for (double factor = 0.01; factor <= 0.35; factor += 0.01) {
            std::vector<cv::Point> poly;
            cv::approxPolyDP(hull, poly, factor * hullArcLen, true);

            if (poly.size() == 4) {
                for (auto& p : poly)
                    approxPoly.push_back(cv::Point2f(
                        static_cast<float>(p.x),
                        static_cast<float>(p.y)));
                std::cout << "[STEP 4]  4-point polygon (hull) "
                             "found at factor=" << factor << "\n";
                found = true;
                break;
            }
        }
    }

    if (!found) {
        std::cerr << "[ERROR] Could not approximate contour "
                     "to 4 vertices. Image may not contain a "
                     "clear quadrilateral document.\n";
        return {};
    }

    // Visualise the 4-point polygon
    cv::Mat polyVis = colorImage.clone();
    for (int i = 0; i < 4; ++i) {
        cv::line(polyVis,
            cv::Point(approxPoly[i]),
            cv::Point(approxPoly[(i + 1) % 4]),
            cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
    for (auto& p : approxPoly) {
        cv::circle(polyVis, cv::Point(p), 8,
            cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    }

    showStep("4pt_Polygon_Approximation", polyVis, saveIntermediates);


    // ══════════════════════════════════════════════════════════
    //  STEP 5 — Corner Extraction and Sorting
    //
    //  Raw approxPolyDP vertices are in contour-traversal order
    //  (clockwise or counter-clockwise depending on image
    //  coordinates).  We need a CANONICAL order so that
    //  downstream perspective correction (getPerspectiveTransform)
    //  maps them consistently to the output rectangle corners.
    //
    //  The sortCorners() function (defined above) uses the
    //  sum/difference trick:
    //    TL = min(x+y)   TR = min(y-x)
    //    BR = max(x+y)   BL = max(y-x)
    // ══════════════════════════════════════════════════════════

    std::vector<cv::Point2f> corners = sortCorners(approxPoly);

    // ── Final visualisation ───────────────────────────────────
    cv::Mat finalVis = colorImage.clone();

    // Corner labels and distinct colours
    const std::vector<cv::Scalar> cornerColors = {
        {0,   255, 0  },   // TL — green
        {0,   200, 255},   // TR — yellow-ish
        {0,   0,   255},   // BR — red
        {255, 100, 0  },   // BL — blue-orange
    };
    const std::vector<std::string> cornerLabels = {"TL","TR","BR","BL"};

    // Draw connecting lines first (underneath circles)
    for (int i = 0; i < 4; ++i) {
        cv::line(finalVis,
            cv::Point(corners[i]),
            cv::Point(corners[(i + 1) % 4]),
            cv::Scalar(255, 255, 0), 3, cv::LINE_AA);
    }

    // Draw circles and labels
    for (int i = 0; i < 4; ++i) {
        cv::circle(finalVis, cv::Point(corners[i]),
            14, cornerColors[i], -1, cv::LINE_AA);
        cv::circle(finalVis, cv::Point(corners[i]),
            14, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);   // white ring

        // Offset label slightly from the corner
        cv::Point labelPos(
            static_cast<int>(corners[i].x) + 18,
            static_cast<int>(corners[i].y) - 10);
        cv::putText(finalVis, cornerLabels[i], labelPos,
            cv::FONT_HERSHEY_DUPLEX, 1.0,
            cv::Scalar(0, 0, 0), 4, cv::LINE_AA);   // black outline
        cv::putText(finalVis, cornerLabels[i], labelPos,
            cv::FONT_HERSHEY_DUPLEX, 1.0,
            cornerColors[i], 1, cv::LINE_AA);        // coloured text
    }

    showStep("Final_Detected_Corners", finalVis, saveIntermediates);

    // Print final corner coordinates to console
    std::cout << "\n══════════════════════════════════════\n";
    std::cout << "  Detected Document Corners:\n";
    std::cout << "──────────────────────────────────────\n";
    const std::vector<std::string> labels = {"Top-Left","Top-Right","Bottom-Right","Bottom-Left"};
    for (int i = 0; i < 4; ++i) {
        std::cout << "  " << labels[i] << ": ("
                  << std::fixed << std::setprecision(1)
                  << corners[i].x << ", " << corners[i].y << ")\n";
    }
    std::cout << "══════════════════════════════════════\n\n";

    return corners;
}

// ─────────────────────────────────────────────────────────────
//  Optional: Bonus — Perspective Warp to a flat document
//  (Uncomment and call after detectDocumentCorners if desired)
// ─────────────────────────────────────────────────────────────
cv::Mat perspectiveWarp(const cv::Mat& src,
                        const std::vector<cv::Point2f>& corners)
{
    if (corners.size() != 4) return {};

    // Estimate output dimensions from the detected quad
    float widthTop    = cv::norm(corners[1] - corners[0]);
    float widthBottom = cv::norm(corners[2] - corners[3]);
    float heightLeft  = cv::norm(corners[3] - corners[0]);
    float heightRight = cv::norm(corners[2] - corners[1]);

    int outW = static_cast<int>(std::max(widthTop,  widthBottom));
    int outH = static_cast<int>(std::max(heightLeft, heightRight));

    std::vector<cv::Point2f> dst = {
        {0.f,                  0.f       },   // TL
        {static_cast<float>(outW - 1), 0.f},  // TR
        {static_cast<float>(outW - 1),
         static_cast<float>(outH - 1)   },    // BR
        {0.f, static_cast<float>(outH - 1)}   // BL
    };


    cv::Mat M   = cv::getPerspectiveTransform(corners, dst);
    cv::Mat out;
    cv::warpPerspective(src, out, M, cv::Size(outW, outH));
    return out;
}

// ─────────────────────────────────────────────────────────────
//  Synthetic test image generator
//  Creates a white rectangle on a noisy gradient background,
//  rotated slightly to simulate a document on a table.
// ─────────────────────────────────────────────────────────────
cv::Mat createTestImage(int W = 800, int H = 600)
{
    cv::Mat img(H, W, CV_8UC3);

    // Gradient background
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            img.at<cv::Vec3b>(y, x) = {
                static_cast<uchar>(80 + x * 100 / W),
                static_cast<uchar>(60 + y *  80 / H),
                static_cast<uchar>(40)};

    // Add noise
    cv::Mat noise(H, W, CV_8SC3);
    cv::randn(noise, 0, 25);
    img += noise;

    // Draw a rotated white rectangle (the "document")
    cv::Point2f center(W / 2.f, H / 2.f);
    cv::Size2f  docSize(W * 0.6f, H * 0.7f);
    cv::RotatedRect rrect(center, docSize, 8.f);  // 8° tilt

    std::vector<cv::Point> verts(4);
    cv::Point2f pts[4];
    rrect.points(pts);
    for (int i = 0; i < 4; ++i)
        verts[i] = cv::Point(pts[i]);

    cv::fillConvexPoly(img, verts, cv::Scalar(240, 240, 230));

    // Add some "text" lines inside the doc
    cv::Point off(static_cast<int>(pts[0].x + 20),
                  static_cast<int>(pts[0].y + 20));
    for (int r = 0; r < 6; ++r)
        cv::line(img,
            cv::Point(off.x + r * 5, off.y + r * 30),
            cv::Point(off.x + r * 5 + 200, off.y + r * 30),
            cv::Scalar(100, 100, 100), 1);

    // Simulate uneven lighting with a dark vignette
    cv::Mat vignette(H, W, CV_8UC1, cv::Scalar(0));
    cv::circle(vignette, cv::Point(W / 3, H / 3),
        std::max(W, H) / 2, cv::Scalar(80), -1);
    cv::GaussianBlur(vignette, vignette, cv::Size(151, 151), 0);
    cv::cvtColor(vignette, vignette, cv::COLOR_GRAY2BGR);
    img -= vignette * 0.5;

    return img;
}

// ─────────────────────────────────────────────────────────────
//  main()
// ─────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║  Document Corner Detector — CV Pipeline ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    cv::Mat image;

    if (argc >= 2) {
        image = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "[ERROR] Cannot read image: " << argv[1] << "\n";
            return 1;
        }
        std::cout << "[INFO] Loaded image: " << argv[1]
                  << "  (" << image.cols << "×" << image.rows << ")\n\n";
    } else {
        std::cout << "[INFO] No image path given — using synthetic test image.\n\n";
        image = createTestImage();
    }

    // Set to true to save PNGs instead of displaying windows
    // (useful on headless servers or for automated testing)
    constexpr bool SAVE_TO_DISK = false;

    std::vector<cv::Point2f> corners =
        detectDocumentCorners(image, SAVE_TO_DISK);

    if (corners.size() == 4) {
        // ── Bonus: perspective warp ──────────────────────────
        cv::Mat warped = perspectiveWarp(image, corners);
        if (!warped.empty()) {
            if (SAVE_TO_DISK) {
                cv::imwrite("step_6_Warped_Document.png", warped);
            } else {
                cv::imshow("Bonus: Perspective Warped Document", warped);
                cv::waitKey(0);
            }
        }
        std::cout << "[SUCCESS] Pipeline complete.\n";
        return 0;
    } else {
        std::cerr << "[FAILURE] Document corners not detected.\n";
        return 1;
    }
}