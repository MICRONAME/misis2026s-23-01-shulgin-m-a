
/**
 * Document Corner Detection and Rectification
 * =============================================
 * A robust pipeline for detecting document corners on various surfaces
 * under challenging lighting conditions, then rectifying the perspective.
 *
 * Build:
 *   g++ -std=c++17 -O2 -o doc_detect doc_detect.cpp $(pkg-config --cflags --libs opencv4)
 *
 * Usage:
 *   ./doc_detect <input_image_path> [output_rectified_path]
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <filesystem>
#include <fstream>    // ← Добавлено
#include <iomanip>    // ← Добавлено
#include <nlohmann/json.hpp> // ← Библиотека для парсинга JSON (header-only)

// ============================================================================
// Configuration Constants
// ============================================================================

namespace Config {
    // Preprocessing
    constexpr int RESIZE_MAX_DIM        = 800;    // Max dimension for processing speed
    constexpr int BILATERAL_D           = 9;      // Bilateral filter diameter
    constexpr int BILATERAL_SIGMA_COLOR = 75;     // Color similarity sigma
    constexpr int BILATERAL_SIGMA_SPACE = 75;     // Spatial proximity sigma
    constexpr int GAUSSIAN_KERNEL       = 5;      // Gaussian blur kernel size

    // Adaptive Thresholding
    constexpr int ADAPTIVE_BLOCK_SIZE   = 15;     // Block size (must be odd, >= 3)
    constexpr int ADAPTIVE_C            = 2;      // Constant subtracted from mean

    // Morphological operations
    constexpr int MORPH_KERNEL_SIZE     = 5;      // Morphological kernel size
    constexpr int MORPH_ITERATIONS      = 2;      // Number of morphological iterations

    // Contour filtering
    constexpr double MIN_AREA_RATIO     = 0.05;   // Minimum contour area as ratio of image area
    constexpr double MAX_AREA_RATIO     = 0.98;   // Maximum contour area as ratio of image area

    // Polygon approximation
    constexpr double EPSILON_START      = 0.01;   // Starting epsilon ratio for approxPolyDP
    constexpr double EPSILON_END        = 0.10;   // Ending epsilon ratio
    constexpr double EPSILON_STEP       = 0.005;  // Step increment for epsilon search

    // Corner validation
    constexpr double MIN_ANGLE          = 30.0;   // Minimum interior angle (degrees)
    constexpr double MAX_ANGLE          = 150.0;  // Maximum interior angle (degrees)
    constexpr double MIN_SIDE_RATIO     = 0.15;   // Minimum side length ratio (shortest/longest)

    // Rectified output
    constexpr int RECTIFIED_WIDTH       = 850;    // A4 proportional width
    constexpr int RECTIFIED_HEIGHT      = 1100;   // A4 proportional height

    // Display
    constexpr bool SAVE_INTERMEDIATES   = true;   // Save intermediate results to disk
    constexpr bool SHOW_WINDOWS         = true;   // Show cv::imshow windows
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Displays and optionally saves an intermediate result image.
 */
void showIntermediate(const std::string& windowName, const cv::Mat& image,
                      const std::string& savePath = "") {
    if (Config::SHOW_WINDOWS) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 
            std::min(image.cols, 800), 
            std::min(image.rows, 800));
        cv::imshow(windowName, image);
    }
    if (Config::SAVE_INTERMEDIATES && !savePath.empty()) {
        cv::imwrite(savePath, image);
        std::cout << "  [Saved] " << savePath << std::endl;
    }
}

/**
 * Computes the Euclidean distance between two points.
 */
double pointDistance(const cv::Point2f& a, const cv::Point2f& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

/**
 * Computes the angle (in degrees) at vertex B for triangle A-B-C.
 */
double angleBetween(const cv::Point2f& A, const cv::Point2f& B, const cv::Point2f& C) {
    cv::Point2f BA = A - B;
    cv::Point2f BC = C - B;
    double dotProduct = BA.x * BC.x + BA.y * BC.y;
    double magBA = std::sqrt(BA.x * BA.x + BA.y * BA.y);
    double magBC = std::sqrt(BC.x * BC.x + BC.y * BC.y);
    if (magBA < 1e-6 || magBC < 1e-6) return 0.0;
    double cosAngle = dotProduct / (magBA * magBC);
    cosAngle = std::clamp(cosAngle, -1.0, 1.0);
    return std::acos(cosAngle) * 180.0 / CV_PI;
}

/**
 * Sorts 4 points into consistent order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
 *
 * Strategy:
 *   1. Compute the centroid of all 4 points.
 *   2. Classify each point by its position relative to the centroid:
 *      - Top-Left: x < cx, y < cy  (sum x+y is smallest)
 *      - Top-Right: x > cx, y < cy (diff x-y is largest)
 *      - Bottom-Right: x > cx, y > cy (sum x+y is largest)
 *      - Bottom-Left: x < cx, y > cy (diff x-y is smallest)
 *   
 *   The sum/difference approach is more robust than centroid-quadrant classification
 *   when points are close to the centroid axes.
 */
std::vector<cv::Point2f> sortCorners(const std::vector<cv::Point2f>& pts) {
    if (pts.size() != 4) {
        std::cerr << "Error: sortCorners expects exactly 4 points." << std::endl;
        return pts;
    }

    std::vector<cv::Point2f> sorted(4);

    // Compute sum (x+y) and difference (x-y) for each point
    std::vector<float> sums(4), diffs(4);
    for (int i = 0; i < 4; i++) {
        sums[i]  = pts[i].x + pts[i].y;
        diffs[i] = pts[i].x - pts[i].y;
    }

    // Top-Left has the smallest sum
    int tlIdx = std::min_element(sums.begin(), sums.end()) - sums.begin();
    // Bottom-Right has the largest sum
    int brIdx = std::max_element(sums.begin(), sums.end()) - sums.begin();
    // Top-Right has the largest difference (x - y)
    int trIdx = std::max_element(diffs.begin(), diffs.end()) - diffs.begin();
    // Bottom-Left has the smallest difference (x - y)
    int blIdx = std::min_element(diffs.begin(), diffs.end()) - diffs.begin();

    sorted[0] = pts[tlIdx]; // Top-Left
    sorted[1] = pts[trIdx]; // Top-Right
    sorted[2] = pts[brIdx]; // Bottom-Right
    sorted[3] = pts[blIdx]; // Bottom-Left

    return sorted;
}

/**
 * Validates that 4 corners form a reasonable quadrilateral for a document.
 * Checks:
 *   - All interior angles are within [MIN_ANGLE, MAX_ANGLE]
 *   - The ratio of shortest to longest side is above MIN_SIDE_RATIO
 *   - The quadrilateral is convex
 */
bool validateQuadrilateral(const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) return false;

    // Check convexity
    std::vector<cv::Point> intCorners;
    for (const auto& p : corners) intCorners.push_back(cv::Point((int)p.x, (int)p.y));
    if (!cv::isContourConvex(intCorners)) return false;

    // Check interior angles
    for (int i = 0; i < 4; i++) {
        double angle = angleBetween(
            corners[(i + 3) % 4], corners[i], corners[(i + 1) % 4]);
        if (angle < Config::MIN_ANGLE || angle > Config::MAX_ANGLE) {
            return false;
        }
    }

    // Check side length ratios
    std::vector<double> sides;
    for (int i = 0; i < 4; i++) {
        sides.push_back(pointDistance(corners[i], corners[(i + 1) % 4]));
    }
    double minSide = *std::min_element(sides.begin(), sides.end());
    double maxSide = *std::max_element(sides.begin(), sides.end());
    if (maxSide < 1e-6) return false;
    if (minSide / maxSide < Config::MIN_SIDE_RATIO) return false;

    return true;
}

// ============================================================================
// STEP 1: Preprocessing
// ============================================================================

/**
 * Preprocesses the input image:
 *   - Resizes for consistent processing (preserves aspect ratio)
 *   - Converts to grayscale
 *   - Applies bilateral filter (edge-preserving noise reduction)
 *     * Bilateral filter is preferred over Gaussian for document detection
 *       because it smooths flat regions while keeping edges sharp, which is
 *       critical for detecting document boundaries against textured surfaces.
 *   - Also applies CLAHE for contrast enhancement to handle uneven lighting
 */
struct PreprocessResult {
    cv::Mat original;       // Original resized color image
    cv::Mat gray;           // Grayscale
    cv::Mat blurred;        // Blurred grayscale
    cv::Mat enhanced;       // CLAHE-enhanced grayscale
    double scaleFactor;     // Scale factor used for resizing
};

PreprocessResult preprocess(const cv::Mat& input) {
    std::cout << "\n=== STEP 1: Preprocessing ===" << std::endl;
    PreprocessResult result;

    // Resize to manageable dimensions while preserving aspect ratio
    int maxDim = std::max(input.cols, input.rows);
    result.scaleFactor = (maxDim > Config::RESIZE_MAX_DIM) 
        ? (double)Config::RESIZE_MAX_DIM / maxDim 
        : 1.0;

    if (result.scaleFactor < 1.0) {
        cv::resize(input, result.original, cv::Size(), 
                   result.scaleFactor, result.scaleFactor, cv::INTER_AREA);
        std::cout << "  Resized from " << input.cols << "x" << input.rows
                  << " to " << result.original.cols << "x" << result.original.rows
                  << " (scale=" << result.scaleFactor << ")" << std::endl;
    } else {
        result.original = input.clone();
        std::cout << "  No resize needed: " << input.cols << "x" << input.rows << std::endl;
    }

    // Convert to grayscale
    cv::cvtColor(result.original, result.gray, cv::COLOR_BGR2GRAY);

    // Apply bilateral filter for edge-preserving smoothing
    // Parameters:
    //   d=9: Diameter of pixel neighborhood (larger = slower but smoother)
    //   sigmaColor=75: Filter sigma in color space (larger = more colors mixed)
    //   sigmaSpace=75: Filter sigma in coordinate space (larger = farther pixels influence)
    cv::bilateralFilter(result.gray, result.blurred, 
                        Config::BILATERAL_D, 
                        Config::BILATERAL_SIGMA_COLOR, 
                        Config::BILATERAL_SIGMA_SPACE);
    std::cout << "  Applied bilateral filter (d=" << Config::BILATERAL_D 
              << ", sigmaColor=" << Config::BILATERAL_SIGMA_COLOR
              << ", sigmaSpace=" << Config::BILATERAL_SIGMA_SPACE << ")" << std::endl;

    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle
    // uneven illumination. This normalizes local contrast, making the document
    // edges more visible even in shadowed/highlighted regions.
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(result.blurred, result.enhanced);
    std::cout << "  Applied CLAHE for contrast enhancement" << std::endl;

    // Show intermediate results
    showIntermediate("Step 1a: Grayscale", result.gray, "step1a_grayscale.png");
    showIntermediate("Step 1b: Bilateral Filtered", result.blurred, "step1b_bilateral.png");
    showIntermediate("Step 1c: CLAHE Enhanced", result.enhanced, "step1c_clahe.png");

    return result;
}

// ============================================================================
// STEP 2: Local Binarization (Multiple Strategies)
// ============================================================================

/**
 * Generates multiple binary masks using different strategies to handle
 * various surface/lighting conditions:
 * 
 * Strategy A: Adaptive Threshold (Gaussian) - Good for general cases
 * Strategy B: Adaptive Threshold (Mean) - Good for uniform surfaces
 * Strategy C: Canny Edge Detection + Dilation - Good for cluttered backgrounds
 * Strategy D: Combined edge-based approach with morphological closure
 *
 * Each strategy produces a binary mask. We try corner detection on each
 * and use the first one that succeeds validation.
 */
struct BinarizationResult {
    std::vector<cv::Mat> masks;         // Multiple binary masks to try
    std::vector<std::string> names;     // Names for debugging
};

BinarizationResult binarize(const PreprocessResult& preproc) {
    std::cout << "\n=== STEP 2: Local Binarization ===" << std::endl;
    BinarizationResult result;

    cv::Mat morphKernel = cv::getStructuringElement(
        cv::MORPH_RECT, 
        cv::Size(Config::MORPH_KERNEL_SIZE, Config::MORPH_KERNEL_SIZE));

    // --- Strategy A: Adaptive Threshold (Gaussian) ---
    // Uses a weighted sum of neighborhood values (Gaussian window).
    // Block size = 15: Neighborhood size for computing threshold.
    //   Must be odd. Larger values are more tolerant of gradual illumination changes.
    // C = 2: Constant subtracted from the computed mean.
    //   Positive C makes the threshold slightly lower, capturing more foreground.
    {
        cv::Mat binary;
        cv::adaptiveThreshold(preproc.enhanced, binary, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY,
                              Config::ADAPTIVE_BLOCK_SIZE,
                              Config::ADAPTIVE_C);
        // Invert so document boundary becomes white
        cv::bitwise_not(binary, binary);
        // Close gaps in the boundary using morphological closing
        // (dilation followed by erosion). This connects broken edge segments.
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, morphKernel, 
                         cv::Point(-1,-1), Config::MORPH_ITERATIONS);
        result.masks.push_back(binary);
        result.names.push_back("Adaptive Gaussian");
        std::cout << "  Strategy A: Adaptive Threshold (Gaussian), block=" 
                  << Config::ADAPTIVE_BLOCK_SIZE << ", C=" << Config::ADAPTIVE_C << std::endl;
    }

    // --- Strategy B: Adaptive Threshold (Mean) with larger block ---
    // Mean-based adaptive threshold with a bigger neighborhood.
    // Larger block size (51) helps when illumination varies gradually.
    {
        cv::Mat binary;
        cv::adaptiveThreshold(preproc.enhanced, binary, 255,
                              cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY,
                              51, 5);
        cv::bitwise_not(binary, binary);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, morphKernel, 
                         cv::Point(-1,-1), Config::MORPH_ITERATIONS);
        // Additional dilation to strengthen boundaries
        cv::dilate(binary, binary, morphKernel, cv::Point(-1,-1), 1);
        result.masks.push_back(binary);
        result.names.push_back("Adaptive Mean (large block)");
        std::cout << "  Strategy B: Adaptive Threshold (Mean), block=51, C=5" << std::endl;
    }

    // --- Strategy C: Canny Edge Detection ---
    // Canny is excellent for detecting sharp edges (document boundaries)
    // even on complex/textured surfaces.
    // We use auto-thresholding based on the median pixel intensity.
    {
        // Compute median for auto Canny thresholds
        // This makes the algorithm adaptive to different image brightnesses.
        cv::Mat sorted;
        cv::sort(preproc.blurred.reshape(1, 1), sorted, cv::SORT_ASCENDING);
        double median = sorted.at<uchar>(sorted.cols / 2);
        double lower = std::max(0.0, 0.67 * median);
        double upper = std::min(255.0, 1.33 * median);

        cv::Mat edges;
        cv::Canny(preproc.enhanced, edges, lower, upper);
        // Dilate edges to close small gaps, then apply closing
        cv::dilate(edges, edges, morphKernel, cv::Point(-1,-1), 2);
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, morphKernel, 
                         cv::Point(-1,-1), 2);
        result.masks.push_back(edges);
        result.names.push_back("Canny Edges");
        std::cout << "  Strategy C: Canny (auto thresholds: " 
                  << lower << "-" << upper << ")" << std::endl;
    }

    // --- Strategy D: Combined gradient-based ---
    // Compute gradient magnitude using Sobel, then threshold adaptively.
    // This approach emphasizes strong edges regardless of absolute brightness.
    {
        cv::Mat gradX, gradY, gradMag;
        cv::Sobel(preproc.enhanced, gradX, CV_64F, 1, 0, 3);
        cv::Sobel(preproc.enhanced, gradY, CV_64F, 0, 1, 3);
        cv::magnitude(gradX, gradY, gradMag);
        cv::normalize(gradMag, gradMag, 0, 255, cv::NORM_MINMAX);
        gradMag.convertTo(gradMag, CV_8U);

        cv::Mat binary;
        cv::adaptiveThreshold(gradMag, binary, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY,
                              31, 0);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, morphKernel, 
                         cv::Point(-1,-1), 3);
        cv::dilate(binary, binary, morphKernel, cv::Point(-1,-1), 1);
        result.masks.push_back(binary);
        result.names.push_back("Gradient-based");
        std::cout << "  Strategy D: Gradient magnitude + adaptive threshold" << std::endl;
    }

    // Show all binary masks
    for (size_t i = 0; i < result.masks.size(); i++) {
        std::string winName = "Step 2 [" + std::to_string(i) + "]: " + result.names[i];
        std::string saveName = "step2_binary_" + std::to_string(i) + ".png";
        showIntermediate(winName, result.masks[i], saveName);
    }

    return result;
}

// ============================================================================
// STEP 3: Contour Detection and Filtering
// ============================================================================

/**
 * Finds and filters contours from a binary mask:
 *   - Uses RETR_EXTERNAL to get only outermost contours (ignore holes)
 *   - Uses CHAIN_APPROX_SIMPLE to compress segments (saves memory)
 *   - Filters by area: keeps contours between MIN_AREA_RATIO and MAX_AREA_RATIO
 *     of the total image area
 *   - Sorts by area (largest first)
 */
std::vector<std::vector<cv::Point>> findAndFilterContours(
    const cv::Mat& binary, 
    const cv::Mat& originalForDebug,
    const std::string& strategyName,
    int strategyIndex) 
{
    double imageArea = binary.rows * binary.cols;
    double minArea = imageArea * Config::MIN_AREA_RATIO;
    double maxArea = imageArea * Config::MAX_AREA_RATIO;

    std::vector<std::vector<cv::Point>> allContours;
    cv::findContours(binary, allContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::cout << "  Found " << allContours.size() << " total contours" << std::endl;

    // Filter by area
    std::vector<std::vector<cv::Point>> filtered;
    for (const auto& contour : allContours) {
        double area = cv::contourArea(contour);
        if (area >= minArea && area <= maxArea) {
            filtered.push_back(contour);
        }
    }

    // Sort by area (largest first) - the document should be the largest contour
    std::sort(filtered.begin(), filtered.end(),
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return cv::contourArea(a) > cv::contourArea(b);
              });

    std::cout << "  After area filter: " << filtered.size() << " contours remain"
              << " (min_area=" << (int)minArea << ", max_area=" << (int)maxArea << ")" << std::endl;

    // Visualize
    cv::Mat contourVis = originalForDebug.clone();
    for (size_t i = 0; i < filtered.size() && i < 5; i++) {
        // Color code: largest=green, others=yellow to red
        cv::Scalar color;
        if (i == 0) color = cv::Scalar(0, 255, 0);
        else if (i == 1) color = cv::Scalar(0, 255, 255);
        else color = cv::Scalar(0, 0, 255);

        cv::drawContours(contourVis, filtered, (int)i, color, 2);
        double area = cv::contourArea(filtered[i]);
        std::cout << "    Contour " << i << ": area=" << (int)area 
                  << " (" << (area/imageArea*100) << "% of image)" << std::endl;
    }

    std::string winName = "Step 3 [" + std::to_string(strategyIndex) + "]: Contours (" + strategyName + ")";
    std::string saveName = "step3_contours_" + std::to_string(strategyIndex) + ".png";
    showIntermediate(winName, contourVis, saveName);

    return filtered;
}

// ============================================================================
// STEP 4: Polygon Approximation
// ============================================================================

/**
 * Attempts to approximate a contour as a 4-sided polygon.
 * 
 * Uses cv::approxPolyDP with dynamically adjusted epsilon:
 *   epsilon = fraction * arcLength(contour)
 * 
 * The fraction starts small (0.01 = 1% of perimeter) and increases gradually.
 * A small epsilon keeps more detail (more vertices), while a larger epsilon 
 * simplifies more aggressively (fewer vertices).
 * 
 * We search for the sweet spot where the polygon has exactly 4 vertices
 * and passes geometric validation.
 */
bool approximateQuad(const std::vector<cv::Point>& contour,
                     std::vector<cv::Point2f>& outCorners) 
{
    double perimeter = cv::arcLength(contour, true);

    // Try a range of epsilon values
    for (double epsFrac = Config::EPSILON_START; 
         epsFrac <= Config::EPSILON_END; 
         epsFrac += Config::EPSILON_STEP) 
    {
        double epsilon = epsFrac * perimeter;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, epsilon, true);

        if (approx.size() == 4) {
            // Convert to Point2f for sub-pixel accuracy
            std::vector<cv::Point2f> corners;
            for (const auto& p : approx) {
                corners.push_back(cv::Point2f((float)p.x, (float)p.y));
            }

            // Sort corners into consistent order
            corners = sortCorners(corners);

            // Validate the quadrilateral geometry
            if (validateQuadrilateral(corners)) {
                outCorners = corners;
                std::cout << "    Found valid quad at epsilon=" << epsFrac 
                          << " (" << epsilon << " px)" << std::endl;
                return true;
            }
        }
    }

    // Fallback: use minimum area rectangle
    // This handles cases where the contour is too noisy for approxPolyDP
    if (contour.size() >= 5) {
        cv::RotatedRect minRect = cv::minAreaRect(contour);
        cv::Point2f rectPts[4];
        minRect.points(rectPts);

        std::vector<cv::Point2f> corners(rectPts, rectPts + 4);
        corners = sortCorners(corners);

        // Check that the rectangle area is close to the contour area
        double contourArea = cv::contourArea(contour);
        double rectArea = minRect.size.width * minRect.size.height;
        double areaRatio = contourArea / rectArea;

        // If the contour fills at least 70% of its bounding rectangle,
        // the rectangle is a reasonable approximation
        if (areaRatio > 0.70 && validateQuadrilateral(corners)) {
            outCorners = corners;
            std::cout << "    Fallback: minAreaRect (fill ratio=" 
                      << areaRatio << ")" << std::endl;
            return true;
        }
    }

    return false;
}

// ============================================================================
// STEP 5: Corner Refinement and Final Detection
// ============================================================================

/**
 * Refines detected corners to sub-pixel accuracy using cv::cornerSubPix.
 * This provides more precise corner locations for better rectification.
 */
void refineCorners(const cv::Mat& gray, std::vector<cv::Point2f>& corners) {
    cv::Size winSize(5, 5);      // Half of the search window
    cv::Size zeroZone(-1, -1);   // No dead zone
    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 40, 0.001);

    try {
        cv::cornerSubPix(gray, corners, winSize, zeroZone, criteria);
        std::cout << "  Sub-pixel corner refinement applied" << std::endl;
    } catch (...) {
        std::cout << "  Sub-pixel refinement skipped (corners near edge)" << std::endl;
    }
}

// ============================================================================
// STEP 6: Perspective Rectification
// ============================================================================

/**
 * Warps the document region into a top-down, rectangular view.
 * 
 * The destination size is computed from the actual detected dimensions
 * to minimize distortion, then optionally scaled to standard sizes.
 * 
 * Points must be in order: TL, TR, BR, BL.
 */
cv::Mat rectifyDocument(const cv::Mat& original, 
                        const std::vector<cv::Point2f>& corners) 
{
    std::cout << "\n=== STEP 6: Perspective Rectification ===" << std::endl;

    // Compute the width and height of the document in the image
    // Top edge and bottom edge
    double widthTop = pointDistance(corners[0], corners[1]);
    double widthBottom = pointDistance(corners[3], corners[2]);
    double maxWidth = std::max(widthTop, widthBottom);

    // Left edge and right edge
    double heightLeft = pointDistance(corners[0], corners[3]);
    double heightRight = pointDistance(corners[1], corners[2]);
    double maxHeight = std::max(heightLeft, heightRight);

    std::cout << "  Detected document dimensions: " 
              << (int)maxWidth << " x " << (int)maxHeight << " px" << std::endl;

    // Determine output size - use detected aspect ratio but scale to reasonable size
    double aspectRatio = maxWidth / maxHeight;
    int outWidth, outHeight;

    if (aspectRatio > 1.0) {
        // Landscape
        outWidth = Config::RECTIFIED_HEIGHT;  // Use the larger dimension for width
        outHeight = (int)(outWidth / aspectRatio);
    } else {
        // Portrait
        outHeight = Config::RECTIFIED_HEIGHT;
        outWidth = (int)(outHeight * aspectRatio);
    }

    std::cout << "  Output size: " << outWidth << " x " << outHeight << std::endl;

    // Define destination points (rectangle)
    std::vector<cv::Point2f> dstPts = {
        cv::Point2f(0, 0),                          // Top-Left
        cv::Point2f((float)outWidth - 1, 0),        // Top-Right
        cv::Point2f((float)outWidth - 1, (float)outHeight - 1),  // Bottom-Right
        cv::Point2f(0, (float)outHeight - 1)         // Bottom-Left
    };

    // Compute the perspective transformation matrix
    cv::Mat M = cv::getPerspectiveTransform(corners, dstPts);

    // Apply the warp
    cv::Mat rectified;
    cv::warpPerspective(original, rectified, M, cv::Size(outWidth, outHeight),
                        cv::INTER_CUBIC,       // Bicubic interpolation for quality
                        cv::BORDER_REPLICATE); // Replicate border pixels

    std::cout << "  Perspective warp applied" << std::endl;

    return rectified;
}

// ============================================================================
// Main Pipeline
// ============================================================================

/**
 * Orchestrates the complete detection pipeline:
 * 1. Preprocess
 * 2. Binarize (multiple strategies)
 * 3. For each strategy: find contours, approximate polygon, validate
 * 4. Use the first successful detection
 * 5. Refine corners, rectify
 */

// ============================================================================
// PIXEL QUALITY EVALUATION (Precision / Recall / F1 / IoU)
// ============================================================================

/**
* Парсит Ground Truth координаты из JSON-файла аннотаций (формат VIA v2)
* Возвращает вектор из 4 точек (порядок обрабатывается sortCorners автоматически)
*/
std::vector<cv::Point2f> parseGTJSON(const std::string& jsonPath) {
    std::vector<cv::Point2f> points;
    try {
        std::ifstream f(jsonPath);
        if (!f.is_open()) {
            std::cerr << "Warning: Could not open GT JSON file: " << jsonPath << std::endl;
            return points;
        }

        nlohmann::json j = nlohmann::json::parse(f);

        // В файле может быть несколько изображений, берём первое
        for (auto& [imgKey, imgData] : j.items()) {
            if (imgData.contains("regions")) {
                const auto& regions = imgData["regions"];
                for (auto& [regionIdx, regionData] : regions.items()) {
                    if (regionData.contains("shape_attributes")) {
                        const auto& shape = regionData["shape_attributes"];
                        if (shape.contains("cx") && shape.contains("cy")) {
                            points.emplace_back(
                                shape["cx"].get<float>(),
                                shape["cy"].get<float>()
                            );
                        }
                    }
                }
            }
            break; // Обрабатываем только первую запись
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to parse GT JSON: " << e.what() << std::endl;
    }
    return points;
}
/**
* Оценивает качество работы Canny edge detector относительно Ground Truth контура.
*
* Метрики:
* - Contour Recall: Какая часть периметра листа (GT) найдена алгоритмом Canny.
* - Noise Ratio: Во сколько раз Canny нашел больше пикселей (включая текст),
*   чем пикселей самого контура. (Идеал = 1.0, чем меньше, тем лучше).
*/
/**
* Оценивает качество Canny и сохраняет визуальную отладку.
*/
void evaluateCannyQuality(const cv::Mat& enhancedImg,
                          const std::vector<cv::Point2f>& gtCorners,
                          double scaleFactor)
{
    if (gtCorners.empty()) return;

    std::cout << "\n=== CANNY EDGE QUALITY EVALUATION ===" << std::endl;

    // 1. Запускаем Canny (копируем параметры из Strategy C)
    cv::Mat sorted;
    cv::sort(enhancedImg.reshape(1, 1), sorted, cv::SORT_ASCENDING);
    double median = sorted.at<uchar>(sorted.cols / 2);
    double lower = std::max(0.0, 0.67 * median);
    double upper = std::min(255.0, 1.33 * median);

    cv::Mat cannyEdges;
    cv::Canny(enhancedImg, cannyEdges, lower, upper);

    // 2. Строим маску Ground Truth (периметр листа)
    cv::Mat gtMask = cv::Mat::zeros(enhancedImg.size(), CV_8U);
    std::vector<cv::Point> pts(4);

    // Масштабируем GT координаты
    for(int i = 0; i < 4; i++) {
        pts[i] = cv::Point((int)(gtCorners[i].x * scaleFactor),
                           (int)(gtCorners[i].y * scaleFactor));
    }

    // Рисуем контур GT (толстая линия)
    cv::polylines(gtMask, pts, true, 255, 1);

    // 3. Создаем "Коридор допуска" (Dilate)
    // Увеличим ядро до 31x31 (±15px) для отладки, чтобы точно увидеть, рядом ли мы
    cv::Mat gtCorridor;
    cv::dilate(gtMask, gtCorridor, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));

    // 4. Считаем пересечения
    cv::Mat intersection;
    cv::bitwise_and(cannyEdges, gtCorridor, intersection);

    int pixelsInCorridor = cv::countNonZero(intersection);
    int totalCannyPixels = cv::countNonZero(cannyEdges);
    int gtCorridorPixels = cv::countNonZero(gtCorridor);

    double recall = (double)pixelsInCorridor / (gtCorridorPixels > 0 ? gtCorridorPixels : 1);
    double noiseRatio = (double)totalCannyPixels / (pixelsInCorridor > 0 ? pixelsInCorridor : 1);

    std::cout << "  Contour Recall: " << std::fixed << std::setprecision(3) << recall << std::endl;
    std::cout << "  Noise Ratio:    " << std::fixed << std::setprecision(2) << noiseRatio << std::endl;

    // === ВИЗУАЛИЗАЦИЯ ДЛЯ ОТЛАДКИ ===
    cv::Mat debugImg = enhancedImg.clone();
    cv::cvtColor(debugImg, debugImg, cv::COLOR_GRAY2BGR);
/*
    // Рисуем зону ожидания (Красный полупрозрачный)
    // Сначала рисуем коридор на маску, затем накладываем
    cv::Mat corridorMask = gtCorridor.clone();
    cv::threshold(corridorMask, corridorMask, 0, 255, cv::THRESH_BINARY); // Ensure binary
    // Делаем зону красной
    cv::Mat redZone(debugImg.size(), CV_8UC3, cv::Scalar(0, 0, 100)); // Dark Red background
    redZone.copyTo(debugImg, corridorMask);
*/
    // Рисуем Canny (Зеленый)
    cv::Mat greenEdges(debugImg.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    greenEdges.copyTo(debugImg, cannyEdges);

    // Рисуем пересечения (Желтый)
    cv::Mat yellowOverlap(debugImg.size(), CV_8UC3, cv::Scalar(0, 255, 255));
    yellowOverlap.copyTo(debugImg, intersection);
    std::vector<cv::Point> gtPts(4);

    for(int i = 0; i < 4; i++) {
        gtPts[i] = cv::Point((int)(gtCorners[i].x * scaleFactor),
                             (int)(gtCorners[i].y * scaleFactor));
    }
    // Рисуем точки GT (Синий)
    // Мы рисуем их поверх всего, чтобы было видно точное местоположение
    for (int i = 0; i < 4; ++i) {
        // Белая обводка для контраста
        cv::circle(debugImg, gtPts[i], 10, cv::Scalar(255, 255, 255), 2);
        // Синяя точка внутри
        cv::circle(debugImg, gtPts[i], 8, cv::Scalar(255, 0, 0), -1);

        // Подпись номера точки
        cv::putText(debugImg, std::to_string(i+1), gtPts[i] + cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }

    cv::imwrite("../images/debug_canny_overlap_015.png", debugImg);
    std::cout << "  [Saved] Debug visualization: debug_canny_overlap.png" << std::endl;
    std::cout << "  -> Look at the image: Red=Expected, Green=Canny, Yellow=Match" << std::endl;
}
// ============================================================================
// POLYGON QUALITY EVALUATION (IoU Method)
// ============================================================================

/**
 * Вычисляет IoU (Intersection over Union) для двух полигонов.
 *
 * Алгоритм:
 * 1. Рисует оба полигона на пустых масках.
 * 2. Находит пересечение масок (битовое И).
 * 3. IoU = Area(Intersection) / Area(Union).
 */
struct PolygonMetrics {
    float iou;                // Насколько полигоны совпадают (0.0 - 1.0)
    float avgCornerDistance;  // Среднее расстояние между углами (в пикселях)
    float overlapPercent;     // Процент пересечения относительно GT
};

PolygonMetrics evaluatePolygonQuality(const std::vector<cv::Point2f>& detectedCorners,
                                      const std::vector<cv::Point2f>& gtCorners,
                                      cv::Size imgSize,
                                      double scaleFactor)
{
    PolygonMetrics m = {0.0f, 0.0f, 0.0f};

    if (gtCorners.empty() || detectedCorners.size() != 4 || gtCorners.size() != 4) {
        return m;
    }

    // 1. Масштабируем GT к размеру текущего изображения
    std::vector<cv::Point2f> gtScaled(4);
    for(int i=0; i<4; i++) {
        gtScaled[i] = cv::Point2f(gtCorners[i].x * scaleFactor,
                                  gtCorners[i].y * scaleFactor);
    }

    // 2. Сортируем углы, чтобы полигоны были нарисованы корректно (не "бантиком")
    // Важно: detectedCorners уже отсортированы вашим кодом, но GT лучше перестраховать
    std::vector<cv::Point2f> detSorted = sortCorners(detectedCorners);
    std::vector<cv::Point2f> gtSorted  = sortCorners(gtScaled);

    // 3. Создаем маски (черные фоны)
    cv::Mat maskDet = cv::Mat::zeros(imgSize, CV_8U);
    cv::Mat maskGt  = cv::Mat::zeros(imgSize, CV_8U);

    // Преобразуем Point2f -> Point для рисования
    std::vector<cv::Point> detPts(4), gtPts(4);
    for(int i=0; i<4; i++) {
        detPts[i] = cv::Point((int)detSorted[i].x, (int)detSorted[i].y);
        gtPts[i]  = cv::Point((int)gtSorted[i].x,  (int)gtSorted[i].y);
    }

    // Рисуем ЗАПОЛНЕННЫЕ полигоны (белым цветом)
    // fillConvexPoly заполняет внутренность фигуры
    cv::fillConvexPoly(maskDet, detPts, cv::Scalar(255));
    cv::fillConvexPoly(maskGt,  gtPts,  cv::Scalar(255));

    // 4. Считаем площади
    // Используем countNonZero, так как маски бинарные (0 или 255)
    // Делим на 255.0, чтобы получить количество "белых" пикселей
    double areaDet = (double)cv::countNonZero(maskDet) / 255.0;
    double areaGt  = (double)cv::countNonZero(maskGt)  / 255.0;

    // 5. Считаем пересечение (Intersection)
    cv::Mat intersection;
    cv::bitwise_and(maskDet, maskGt, intersection);
    double areaIntersection = (double)cv::countNonZero(intersection) / 255.0;

    // 6. Считаем объединение (Union)
    // Union = A + B - Intersection
    double areaUnion = areaDet + areaGt - areaIntersection;

    // 7. Рассчитываем метрики
    if (areaUnion > 0) {
        m.iou = (float)(areaIntersection / areaUnion);
    }

    if (areaGt > 0) {
        m.overlapPercent = (float)(areaIntersection / areaGt) * 100.0f;
    }

    // 8. Считаем среднее расстояние между углами
    double totalDist = 0.0;
    for(int i=0; i<4; i++) {
        totalDist += pointDistance(detSorted[i], gtSorted[i]);
    }
    m.avgCornerDistance = (float)(totalDist / 4.0);

    return m;
}

int main(int argc, char** argv) {
    // ------------------------------------------------------------------
    // Parse arguments
    // ------------------------------------------------------------------
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [output_rectified]" << std::endl;
        std::cerr << "Example: " << argv[0] << " document.jpg rectified.jpg" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    std::string gtJsonPath  = (argc >= 3) ? argv[2] : "";
    std::string outputPath = (argc >= 4) ? argv[3] : "rectified_output.jpg";

    // Загрузка Ground Truth
    std::vector<cv::Point2f> gtCorners;
    if (!gtJsonPath.empty()) {
        std::cout << "\nLoading Ground Truth from: " << gtJsonPath << std::endl;
        gtCorners = parseGTJSON(gtJsonPath);
        if (gtCorners.size() == 4) {
            std::cout << "  Successfully loaded 4 GT points." << std::endl;
        } else {
            std::cerr << "  Warning: Expected 4 points, found " << gtCorners.size() << ". Skipping evaluation." << std::endl;
            gtCorners.clear();
        }
    }
    // ------------------------------------------------------------------
    // Read input image
    // ------------------------------------------------------------------
    cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read image: " << inputPath << std::endl;
        return -1;
    }
    std::cout << "Input image: " << inputPath
              << " (" << inputImage.cols << "x" << inputImage.rows << ")" << std::endl;

    // ------------------------------------------------------------------
    // STEP 1: Preprocessing
    // ------------------------------------------------------------------
    PreprocessResult preproc = preprocess(inputImage);
    evaluateCannyQuality(preproc.enhanced, gtCorners, preproc.scaleFactor);

    // ------------------------------------------------------------------
    // STEP 2: Local Binarization (multiple strategies)
    // ------------------------------------------------------------------
    BinarizationResult binarization = binarize(preproc);

    // ------------------------------------------------------------------
    // STEPS 3-5: Full Strategy Evaluation & Processing
    // ------------------------------------------------------------------
    struct StrategyResult {
        int strategyIdx;
        std::string strategyName;
        std::vector<cv::Point2f> corners; // В масштабе resized изображения
    };

    std::vector<StrategyResult> allResults;

    std::cout << "\n=== STEPS 3-5: Running ALL Strategies ===" << std::endl;

    // 1. Перебираем ВСЕ стратегии без раннего выхода (!found больше нет)
    for (size_t s = 0; s < binarization.masks.size(); ++s) {
        std::cout << "\n--- Testing strategy " << s << ": " << binarization.names[s] << " ---" << std::endl;

        std::vector<std::vector<cv::Point>> contours = findAndFilterContours(
            binarization.masks[s], preproc.original, binarization.names[s], (int)s);

        if (contours.empty()) {
            std::cout << "  No contours found." << std::endl;
            continue;
        }

        // Проверяем топ-5 контуров для каждой стратегии
        for (size_t c = 0; c < contours.size() && c < 5; ++c) {
            std::vector<cv::Point2f> corners;
            if (approximateQuad(contours[c], corners)) {
                StrategyResult res;
                res.strategyIdx = (int)s;
                res.strategyName = binarization.names[s];
                res.corners = corners;
                allResults.push_back(res);
                std::cout << "  ✅ SUCCESS: Strategy " << s << ", Contour " << c << std::endl;
            }
        }
    }

    if (allResults.empty()) {
        std::cerr << "\n❌ Error: All strategies failed to detect a valid document." << std::endl;
        if (Config::SHOW_WINDOWS) {
            std::cout << "\nPress any key to close all windows..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        return -1;
    }

    std::cout << "\n📊 Total valid detections found: " << allResults.size() << std::endl;

    // 2. Обрабатываем, выпрямляем и сохраняем КАЖДЫЙ успешный результат
    for (size_t i = 0; i < allResults.size(); ++i) {
        const auto& res = allResults[i];
        std::string suffix = "_strat" + std::to_string(res.strategyIdx);
        std::vector<cv::Point2f> refined = res.corners;

        // STEP 5: Sub-pixel refinement
        refineCorners(preproc.gray, refined);

        // Масштабируем углы обратно к оригинальному изображению
        std::vector<cv::Point2f> origCorners(4);
        for (int k = 0; k < 4; ++k) {
            origCorners[k].x = refined[k].x / (float)preproc.scaleFactor;
            origCorners[k].y = refined[k].y / (float)preproc.scaleFactor;
        }

        // 🖼️ Визуализация и сохранение Detected Corners
        {
            cv::Mat vis = preproc.original.clone();
            cv::Scalar colors[] = {
                cv::Scalar(255, 0, 0),    // Blue: TL
                cv::Scalar(0, 255, 0),    // Green: TR
                cv::Scalar(0, 0, 255),    // Red: BR
                cv::Scalar(0, 255, 255)   // Yellow: BL
            };
            const char* labels[] = {"TL", "TR", "BR", "BL"};

            std::vector<cv::Point> polyPts;
            for (const auto& p : refined) polyPts.push_back(cv::Point((int)p.x, (int)p.y));
            cv::polylines(vis, polyPts, true, cv::Scalar(0, 255, 0), 3);

            for (int j = 0; j < 4; ++j) {
                cv::Point pt((int)refined[j].x, (int)refined[j].y);
                cv::circle(vis, pt, 12, colors[j], -1);
                cv::circle(vis, pt, 14, cv::Scalar(255, 255, 255), 2);
                cv::putText(vis, labels[j], pt + cv::Point(15, -10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[j], 1);
            }

            std::string saveName = "../iter_1_images/detected_corners" + suffix + ".jpg";
            cv::imwrite(saveName, vis);
            std::cout << "💾 Saved detected corners: " << saveName << std::endl;
        }

        // STEP 6: Perspective Rectification
        cv::Mat rectified = rectifyDocument(inputImage, origCorners);

        // Sharpening (как в оригинале)
        cv::Mat sharpened;
        cv::Mat sharpenKernel = (cv::Mat_<float>(3,3) << 0, -0.5, 0, -0.5, 3, -0.5, 0, -0.5, 0);
        cv::filter2D(rectified, sharpened, -1, sharpenKernel);

        std::string rectName = "../iter_1_images/rectified_document" + suffix + ".jpg";
        cv::imwrite(rectName, sharpened);
        std::cout << "💾 Saved rectified document: " << rectName << std::endl;
    }

    // ------------------------------------------------------------------
    // CLEANUP & EXIT
    // ------------------------------------------------------------------
    if (Config::SHOW_WINDOWS) {
        std::cout << "\nPress any key to close all windows..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

/*    // ------------------------------------------------------------------
    // ОЦЕНКА КАЧЕСТВА ПОЛИГОНА (Вместо Canny Evaluation)
    // ------------------------------------------------------------------
    if (!gtCorners.empty()) {
        std::cout << "\n=== POLYGON QUALITY EVALUATION (IoU) ===" << std::endl;

        // Считаем метрики
        PolygonMetrics metrics = evaluatePolygonQuality(
            detectedCorners,
            gtCorners,
            preproc.original.size(),
            preproc.scaleFactor
        );

        std::cout << "  IoU (Intersection over Union): " << std::fixed << std::setprecision(3) << metrics.iou << std::endl;
        std::cout << "  Overlap with GT:             " << std::fixed << std::setprecision(1) << metrics.overlapPercent << "%" << std::endl;
        std::cout << "  Avg Corner Distance:         " << std::fixed << std::setprecision(1) << metrics.avgCornerDistance << " px" << std::endl;


        // Интерпретация результата
        if (metrics.iou > 0.85) {
            std::cout << "  Result: EXCELLENT (The document is detected perfectly)" << std::endl;
        } else if (metrics.iou > 0.60) {
            std::cout << "  Result: GOOD (Minor deviations)" << std::endl;
        } else {
            std::cout << "  Result: POOR (Significant mismatch)" << std::endl;
        }
    }

    // Optional: enhance the rectified document
    // Apply slight sharpening for cleaner text
    cv::Mat sharpened;
    cv::Mat sharpenKernel = (cv::Mat_<float>(3, 3) <<
         0, -0.5,  0,
        -0.5,  3, -0.5,
         0, -0.5,  0);
    cv::filter2D(rectified, sharpened, -1, sharpenKernel);

    showIntermediate("Step 6a: Rectified Document", rectified, "step6a_rectified.png");
    showIntermediate("Step 6b: Rectified + Sharpened", sharpened, "step6b_sharpened.png");

    // Save final output
    cv::imwrite(outputPath, rectified);
    std::cout << "\n=== COMPLETE ===" << std::endl;
    std::cout << "Rectified document saved to: " << outputPath << std::endl;

    // Also produce a binary (black & white) version for OCR
    cv::Mat rectGray, rectBinary;
    cv::cvtColor(sharpened, rectGray, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(rectGray, rectBinary, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 15, 8);
    cv::imwrite("rectified_bw.png", rectBinary);
    showIntermediate("Step 6c: Rectified B&W (for OCR)", rectBinary, "step6c_bw.png");
    std::cout << "B&W version saved to: rectified_bw.png" << std::endl;

    // ------------------------------------------------------------------
    // Wait for keypress and cleanup
    // ------------------------------------------------------------------
    if (Config::SHOW_WINDOWS) {
        std::cout << "\nPress any key to close all windows..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
*/
    return 0;
}
/*
```

---

## Explanation of Each Step

### Step 1: Preprocessing

**Goal:** Prepare the image for robust edge/boundary detection under varying conditions.

| Operation | Why |
|-----------|-----|
| **Resize** | Keeps processing fast and memory-bounded regardless of input resolution. We track the scale factor to map corners back to the original. |
| **Grayscale** | Reduces 3-channel color data to 1 channel, simplifying all downstream operations. |
| **Bilateral Filter** | Unlike Gaussian blur which blurs edges too, bilateral filtering smooths flat regions (noise) while preserving sharp edges (document boundary). Critical when the document sits on a textured surface (wood grain, fabric). |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization normalizes local contrast. This is essential for uneven illumination — a shadow across one corner of the document won't cause that edge to disappear. |

### Step 2: Local Binarization (Multi-Strategy)

**Goal:** Convert the grayscale image into a binary mask where document boundaries are clearly delineated.

**Why multiple strategies?** A single binarization method cannot handle all surface/lighting combinations:

| Strategy | Best For | How It Works |
|----------|----------|--------------|
| **Adaptive Gaussian** | General cases, moderate lighting variation | Computes threshold from Gaussian-weighted neighborhood. Block size 15 captures local illumination gradients. |
| **Adaptive Mean (large block)** | Uniform surfaces, gradual shadows | Larger block size (51) averages over bigger neighborhoods, tolerating slow illumination changes. |
| **Canny Edges** | Cluttered backgrounds, strong textures | Detects gradient magnitude peaks. Auto-thresholding based on median intensity adapts to image brightness. |
| **Gradient-based** | Low contrast documents, subtle edges | Sobel gradient magnitude captures edge strength regardless of absolute brightness, then adaptive threshold binarizes. |

**Morphological closing** (dilation → erosion) bridges small gaps in the document boundary caused by text, shadows, or surface texture bleeding into the edge.

### Step 3: Contour Detection and Filtering

**Goal:** Extract the document's outline from the binary mask.

- **`RETR_EXTERNAL`**: Only retrieves outermost contours. We don't care about text or internal features.
- **`CHAIN_APPROX_SIMPLE`**: Compresses horizontal/vertical/diagonal segments to just their endpoints, saving memory.
- **Area filtering**: Since the document occupies the majority of the frame (given in constraints), we discard contours smaller than 5% of image area and larger than 98% (which would be the image border itself).
- **Sorting by area**: The largest remaining contour is most likely the document.

### Step 4: Polygon Approximation (approxPolyDP)

**Goal:** Simplify the contour into a 4-vertex polygon (quadrilateral).

**The epsilon parameter** controls approximation aggressiveness:
- `epsilon = fraction × perimeter`
- Small fraction (0.01) → many vertices (follows contour closely)
- Large fraction (0.10) → few vertices (aggressive simplification)

We sweep epsilon from 1% to 10% in 0.5% steps, looking for exactly 4 vertices. Each candidate quadrilateral is validated:
- Must be **convex** (a document is always convex from above)
- Interior angles must be between 30°–150° (rejects degenerate shapes)
- Side length ratio > 0.15 (rejects extremely skewed shapes)

**Fallback**: If `approxPolyDP` fails (too noisy contour), we use `minAreaRect` which always gives 4 corners. We accept it only if the contour fills >70% of the rectangle.

### Step 5: Corner Extraction and Sorting

**Goal:** Produce 4 consistently-ordered corners: TL, TR, BR, BL.

**Sorting logic** (sum/difference method):
- **Top-Left**: smallest `x + y` (closest to origin)
- **Bottom-Right**: largest `x + y` (farthest from origin)
- **Top-Right**: largest `x - y` (far right, near top)
- **Bottom-Left**: smallest `x - y` (far left, near bottom)

This is more robust than centroid-quadrant classification when the document is rotated.

**Sub-pixel refinement** (`cornerSubPix`) searches a small neighborhood around each detected corner for the exact sub-pixel location where gradient lines intersect, improving rectification accuracy.

### Step 6: Perspective Rectification

**Goal:** Warp the detected quadrilateral into a flat, rectangular image.

- **Output dimensions** are computed from the detected document's aspect ratio to avoid distortion.
- **`getPerspectiveTransform`** computes the 3×3 homography matrix mapping source quadrilateral to destination rectangle.
- **`warpPerspective`** with `INTER_CUBIC` interpolation produces smooth, high-quality output.
- **Sharpening** (unsharp mask) compensates for slight blurring introduced by the perspective warp.
- **B&W output** with adaptive thresholding produces clean binary text suitable for OCR.

---

## Build and Run

```bash
# Build
g++ -std=c++17 -O2 -o doc_detect doc_detect.cpp $(pkg-config --cflags --libs opencv4)

# Run with display
./doc_detect photo_of_document.jpg output_rectified.jpg

# Run on multiple test images
for img in test_images/*.jpg; do
    ./doc_detect "$img" "output/$(basename $img)"
done
```

## Output Files

| File | Description |
|------|-------------|
| `step1a_grayscale.png` | Grayscale conversion |
| `step1b_bilateral.png` | After bilateral filtering |
| `step1c_clahe.png` | After CLAHE enhancement |
| `step2_binary_0..3.png` | Binary masks from each strategy |
| `step3_contours_*.png` | Filtered contours drawn on image |
| `step4_polygon.png` | Approximated quadrilateral |
| `step5_corners.png` | Final detected corners with labels |
| `step6a_rectified.png` | Perspective-corrected document |
| `step6b_sharpened.png` | Rectified + sharpened |
| `step6c_bw.png` | Black & white version for OCR |
| `rectified_output.jpg` | Final output file |
*/