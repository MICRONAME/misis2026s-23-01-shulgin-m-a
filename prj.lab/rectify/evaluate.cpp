/**
 * evaluate.cpp
 * Система оценки качества алгоритма ректификации документов.
 * Оценивает только итоговый полигон (4 угла) относительно эталона.
 *
 * Сборка (Linux/macOS):
 *   g++ -std=c++17 -O2 -o evaluate evaluate.cpp $(pkg-config --cflags --libs opencv4)
 *
 * Запуск (одно изображение):
 *   ./evaluate input.jpg results.json METHOD_NAME gt.json
 *
 * Запуск (батч по папке):
 *   ./evaluate --batch images_dir/ results_dir/ METHOD_NAME gt_dir/
 *
 * Форматы JSON:
 *   results.json — формат с массивом methods (от doc_rectifier)
 *   gt.json      — простой формат [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
 *   порядок точек: TL, TR, BR, BL
 *
 * Доступные METHOD_NAME:
 *   - manual_json
 *   - canny_contours
 *   - adaptive_threshold
 *   - gradient_morph
 *
 * Выводит метрики:
 *   - IoU (Intersection over Union)
 *   - Overlap with GT (доля покрытия эталона)
 *   - Average Corner Distance (средняя ошибка по углам, px)
 *   - CornerDist по каждому углу (TL, TR, BR, BL)
 */
#ifdef _WIN32
#include <windows.h>
#endif
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <optional>
#include <map>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ─────────────────────────────────────────────────────────
//  СТРУКТУРЫ ДАННЫХ
// ─────────────────────────────────────────────────────────

struct Quad {
    cv::Point2f pts[4];  // TL, TR, BR, BL
};

struct EvalResult {
    double iou              = 0.0;
    double overlapWithGT    = 0.0;
    double avgCornerDist    = 0.0;
    double cornerDist[4]    = {0, 0, 0, 0};  // TL, TR, BR, BL

    std::string imageName;
    std::string methodName;
    std::string group;
};

// ─────────────────────────────────────────────────────────
//  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ─────────────────────────────────────────────────────────

double euclidean(cv::Point2f a, cv::Point2f b) {
    return cv::norm(a - b);
}

/**
 * Парсит новый формат JSON с несколькими методами.
 * Возвращает map: имя_метода -> Quad
 */
std::optional<std::map<std::string, Quad>> parseMultiMethodJson(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << path << "\n";
        return std::nullopt;
    }

    json j;
    try {
        j = json::parse(f);
    } catch (const json::parse_error& e) {
        std::cerr << "Ошибка парсинга JSON: " << e.what() << "\n";
        return std::nullopt;
    }

    if (!j.contains("methods") || !j["methods"].is_array()) {
        std::cerr << "Ошибка: в JSON отсутствует массив 'methods'\n";
        return std::nullopt;
    }

    std::map<std::string, Quad> results;

    for (const auto& method : j["methods"]) {
        if (!method.contains("name") || !method.contains("points")) {
            continue;
        }

        std::string name = method["name"].get<std::string>();
        const auto& pts_array = method["points"];

        if (!pts_array.is_array() || pts_array.size() != 4) {
            continue;
        }

        Quad q;
        bool valid = true;
        for (int i = 0; i < 4; ++i) {
            if (pts_array[i].is_array() && pts_array[i].size() == 2) {
                q.pts[i].x = pts_array[i][0].get<float>();
                q.pts[i].y = pts_array[i][1].get<float>();
            } else {
                valid = false;
                break;
            }
        }

        if (valid) {
            results[name] = q;
        }
    }

    return results;
}

/** Простой JSON-парсер для gt.json: [[x,y],[x,y],[x,y],[x,y]] */
std::optional<Quad> parseQuadJSON(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return std::nullopt;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    Quad q;
    size_t pos = 0;
    int count = 0;
    while (count < 4) {
        size_t lb    = content.find('[', pos);
        if (lb == std::string::npos) break;
        if (count == 0 && lb == content.find_first_of('[')) {
            pos = lb + 1;
            lb  = content.find('[', pos);
            if (lb == std::string::npos) break;
        }
        size_t comma = content.find(',', lb);
        size_t rb    = content.find(']', lb);
        if (comma == std::string::npos || rb == std::string::npos) break;
        try {
            q.pts[count].x = std::stof(content.substr(lb+1, comma-lb-1));
            q.pts[count].y = std::stof(content.substr(comma+1, rb-comma-1));
        } catch (...) { return std::nullopt; }
        ++count;
        pos = rb + 1;
    }
    if (count < 4) return std::nullopt;
    return q;
}

std::string detectGroup(const std::string& name) {
    if (name.find("simple") != std::string::npos ||
        name.find("easy")   != std::string::npos) return "simple";
    if (name.find("hard")   != std::string::npos ||
        name.find("problem")!= std::string::npos) return "hard";
    if (name.find("complex")!= std::string::npos ||
        name.find("difficult")!=std::string::npos) return "complex";
    return "unknown";
}

// ─────────────────────────────────────────────────────────
//  ОЦЕНКА ПОЛИГОНА
// ─────────────────────────────────────────────────────────

/** IoU двух четырёхугольников через растеризацию */
double polyIoU(const Quad& pred, const Quad& gt, cv::Size imgSize) {
    cv::Mat maskP = cv::Mat::zeros(imgSize, CV_8U);
    cv::Mat maskG = cv::Mat::zeros(imgSize, CV_8U);

    std::vector<cv::Point> pP, pG;
    for (int i = 0; i < 4; ++i) {
        pP.push_back(cv::Point(pred.pts[i]));
        pG.push_back(cv::Point(gt.pts[i]));
    }
    cv::fillConvexPoly(maskP, pP, 255);
    cv::fillConvexPoly(maskG, pG, 255);

    cv::Mat inter, uni;
    cv::bitwise_and(maskP, maskG, inter);
    cv::bitwise_or (maskP, maskG, uni);

    double aI = cv::countNonZero(inter);
    double aU = cv::countNonZero(uni);
    return (aU < 1.0) ? 0.0 : aI / aU;
}

/** Overlap with GT: какую долю площади эталона покрывает предсказание */
double overlapWithGT(const Quad& pred, const Quad& gt, cv::Size imgSize) {
    cv::Mat maskP = cv::Mat::zeros(imgSize, CV_8U);
    cv::Mat maskG = cv::Mat::zeros(imgSize, CV_8U);

    std::vector<cv::Point> pP, pG;
    for (int i = 0; i < 4; ++i) {
        pP.push_back(cv::Point(pred.pts[i]));
        pG.push_back(cv::Point(gt.pts[i]));
    }
    cv::fillConvexPoly(maskP, pP, 255);
    cv::fillConvexPoly(maskG, pG, 255);

    cv::Mat inter;
    cv::bitwise_and(maskP, maskG, inter);

    double aI = cv::countNonZero(inter);
    double aG = cv::countNonZero(maskG);
    return (aG < 1.0) ? 0.0 : aI / aG;
}

/** Average Corner Distance + расстояния по каждому углу */
void cornerDistances(const Quad& pred, const Quad& gt,
                     double out[4], double& avg) {
    avg = 0;
    for (int i = 0; i < 4; ++i) {
        out[i] = euclidean(pred.pts[i], gt.pts[i]);
        avg += out[i];
    }
    avg /= 4.0;
}

// ─────────────────────────────────────────────────────────
//  ВЫЧИСЛЕНИЕ ВСЕХ МЕТРИК ДЛЯ ОДНОГО ИЗОБРАЖЕНИЯ
// ─────────────────────────────────────────────────────────

EvalResult evaluate(const std::string& imagePath,
                    const std::string& resultsJsonPath,
                    const std::string& methodName,
                    const std::string& gtPath) {
    EvalResult res;
    res.imageName  = fs::path(imagePath).filename().string();
    res.methodName = methodName;
    res.group      = detectGroup(res.imageName);

    // 1. Парсим results.json и ищем нужный метод
    auto allMethods = parseMultiMethodJson(resultsJsonPath);
    if (!allMethods) {
        std::cerr << "Ошибка загрузки results.json для: " << imagePath << "\n";
        return res;
    }

    if (allMethods->find(methodName) == allMethods->end()) {
        std::cerr << "Метод '" << methodName << "' не найден в " << resultsJsonPath << "\n";
        return res;
    }

    Quad pred = (*allMethods)[methodName];

    // 2. Парсим gt.json
    auto gtOpt = parseQuadJSON(gtPath);
    if (!gtOpt) {
        std::cerr << "Ошибка загрузки GT для: " << gtPath << "\n";
        return res;
    }
    Quad gt = *gtOpt;

    // 3. Загружаем изображение (нужен только размер)
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Не удалось открыть: " << imagePath << "\n";
        return res;
    }
    cv::Size sz = img.size();

    // 4. Считаем метрики полигона
    res.iou           = polyIoU(pred, gt, sz);
    res.overlapWithGT = overlapWithGT(pred, gt, sz);
    cornerDistances(pred, gt, res.cornerDist, res.avgCornerDist);

    return res;
}

// ─────────────────────────────────────────────────────────
//  ВЫВОД РЕЗУЛЬТАТОВ
// ─────────────────────────────────────────────────────────

void printResult(const EvalResult& r) {
    std::cout << "\n── " << r.imageName << " [" << r.methodName << "] ──\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  IoU            : " << r.iou            << "  ";
    if      (r.iou > 0.85) std::cout << "[ОТЛИЧНО]\n";
    else if (r.iou > 0.60) std::cout << "[УДОВЛ.]\n";
    else                    std::cout << "[ПЛОХО]\n";
    std::cout << "  Overlap GT     : " << r.overlapWithGT  << "\n";
    std::cout << "  Avg Corner Dist: " << r.avgCornerDist  << " px\n";
    std::cout << "    TL=" << r.cornerDist[0]
              << "  TR="   << r.cornerDist[1]
              << "  BR="   << r.cornerDist[2]
              << "  BL="   << r.cornerDist[3] << " px\n";
}

void printBatchSummary(const std::vector<EvalResult>& results, const std::string& methodName) {
    std::map<std::string, std::vector<double>> iouByGroup;
    std::map<std::string, std::vector<double>> acdByGroup;

    for (const auto& r : results) {
        iouByGroup[r.group].push_back(r.iou);
        acdByGroup[r.group].push_back(r.avgCornerDist);
    }

    auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    std::cout << "\n══════════════════════════════════════════════\n";
    std::cout << " БАТЧ-СВОДКА (метод: " << methodName << ")\n";
    std::cout << "══════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::left
              << std::setw(12) << "Группа"
              << std::setw(8)  << "N"
              << std::setw(12) << "IoU (avg)"
              << std::setw(18) << "AvgCorner (px)"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (auto& [grp, ious] : iouByGroup) {
        std::cout << std::setw(12) << grp
                  << std::setw(8)  << ious.size()
                  << std::setw(12) << mean(ious)
                  << std::setw(18) << mean(acdByGroup[grp])
                  << "\n";
    }

    std::vector<double> allIoU, allAcd;
    for (auto& r : results) {
        allIoU.push_back(r.iou);
        allAcd.push_back(r.avgCornerDist);
    }
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::setw(12) << "ИТОГО"
              << std::setw(8)  << results.size()
              << std::setw(12) << mean(allIoU)
              << std::setw(18) << mean(allAcd)
              << "\n";

    int ok85 = 0, ok60 = 0, fail = 0;
    for (auto& r : results) {
        if      (r.iou > 0.85) ++ok85;
        else if (r.iou > 0.60) ++ok60;
        else                    ++fail;
    }
    double n = results.size();
    std::cout << "\nРаспределение по IoU-порогам:\n";
    std::cout << "  IoU > 0.85 (отлично):        "
              << ok85 << " / " << (int)n
              << "  (" << 100.0*ok85/n << "%)\n";
    std::cout << "  0.60 < IoU <= 0.85 (удовл.): "
              << ok60 << " / " << (int)n
              << "  (" << 100.0*ok60/n << "%)\n";
    std::cout << "  IoU <= 0.60 (плохо):         "
              << fail << " / " << (int)n
              << "  (" << 100.0*fail/n << "%)\n";
}

// ─────────────────────────────────────────────────────────
//  ТОЧКА ВХОДА
// ─────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Принудительно устанавливаем кодировку вывода консоли в UTF-8
    SetConsoleOutputCP(65001);
#endif
    if (argc < 2) {
        std::cerr << "Использование:\n"
                  << "  Одно изображение:\n"
                  << "    ./evaluate input.jpg results.json METHOD_NAME gt.json\n"
                  << "  Батч:\n"
                  << "    ./evaluate --batch images/ results_dir/ METHOD_NAME gts/\n\n"
                  << "Доступные METHOD_NAME:\n"
                  << "  - manual_json\n"
                  << "  - canny_contours\n"
                  << "  - adaptive_threshold\n"
                  << "  - gradient_morph\n";
        return 1;
    }

    std::string mode = argv[1];

    // ── Батч-режим ────────────────────────────────────────
    if (mode == "--batch") {
        if (argc < 6) {
            std::cerr << "Батч: нужны images_dir/ results_dir/ METHOD_NAME gts_dir/\n";
            return 1;
        }
        std::string imgDir      = argv[2];
        std::string resultsDir  = argv[3];
        std::string methodName  = argv[4];
        std::string gtDir       = argv[5];

        std::vector<EvalResult> results;
        for (auto& entry : fs::directory_iterator(imgDir)) {
            std::string stem = entry.path().stem().string();
            std::string ext  = entry.path().extension().string();
            if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;

            std::string resultsPath = resultsDir + "/" + stem + ".json";
            std::string gtPath      = gtDir      + "/" + stem + ".json";
            if (!fs::exists(resultsPath) || !fs::exists(gtPath)) continue;

            EvalResult r = evaluate(entry.path().string(),
                                    resultsPath, methodName, gtPath);
            printResult(r);
            results.push_back(r);
        }
        if (!results.empty())
            printBatchSummary(results, methodName);
        return 0;
    }

    // ── Одно изображение ──────────────────────────────────
    if (argc < 5) {
        std::cerr << "Нужны: input.jpg results.json METHOD_NAME gt.json\n";
        return 1;
    }

    EvalResult r = evaluate(argv[1], argv[2], argv[3], argv[4]);
    printResult(r);
    return 0;
}