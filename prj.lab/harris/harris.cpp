//
// Shi-Tomasi Corner Detector с фильтрацией углов текста
// Использует только точки goodFeaturesToTrack, без findContours
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

// Фильтрация: оставляем 4 угла, наиболее близких к границам изображения
// Углы текста обычно находятся внутри, углы листа — у краёв
vector<Point2f> filterSheetCorners(const vector<Point2f>& corners,
                                   int imgWidth, int imgHeight,
                                   int maxResults = 4) {
    if (corners.size() <= maxResults) return corners;

    // Считаем "оценку экстремальности": расстояние до ближайшей границы
    vector<pair<Point2f, double>> scored;
    for (const auto& pt : corners) {
        double distToBorder = min({pt.x, pt.y, (imgWidth - pt.x), (imgHeight - pt.y)});
        scored.emplace_back(pt, distToBorder);
    }

    // Сортируем: чем меньше расстояние до границы — тем "лучше" угол листа
    sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Берём первые maxResults
    vector<Point2f> filtered;
    for (int i = 0; i < min(maxResults, static_cast<int>(scored.size())); i++) {
        filtered.push_back(scored[i].first);
    }

    return filtered;
}

// Сортировка 4 углов: порядок [верх-лев, верх-прав, низ-прав, низ-лев]
vector<Point2f> orderCorners(const vector<Point2f>& corners) {
    if (corners.size() != 4) return corners;

    vector<Point2f> ordered(4);

    // Сумма координат: мин = верх-лев, макс = низ-прав
    vector<pair<double, int>> sumIdx;
    for (int i = 0; i < 4; i++) {
        sumIdx.emplace_back(corners[i].x + corners[i].y, i);
    }
    sort(sumIdx.begin(), sumIdx.end());
    ordered[0] = corners[sumIdx[0].second];  // top-left
    ordered[2] = corners[sumIdx[3].second];  // bottom-right

    // Разность координат: мин = верх-прав, макс = низ-лев
    vector<pair<double, int>> diffIdx;
    for (int i = 0; i < 4; i++) {
        diffIdx.emplace_back(corners[i].x - corners[i].y, i);
    }
    sort(diffIdx.begin(), diffIdx.end());
    ordered[1] = corners[diffIdx[0].second];  // top-right
    ordered[3] = corners[diffIdx[3].second];  // bottom-left

    return ordered;
}

// Дополнительная фильтрация: проверка на прямоугольность
// Углы листа должны образовывать четырёхугольник, близкий к прямоугольнику
bool validateRectangle(const vector<Point2f>& corners, double aspectRatioTolerance = 0.5) {
    if (corners.size() != 4) return false;

    // Сортируем углы для правильного соединения
    vector<Point2f> ordered = orderCorners(corners);

    // Вычисляем длины сторон
    double top = norm(ordered[0] - ordered[1]);
    double right = norm(ordered[1] - ordered[2]);
    double bottom = norm(ordered[2] - ordered[3]);
    double left = norm(ordered[3] - ordered[0]);

    // Проверяем, что противоположные стороны примерно равны
    double hRatio = abs(top - bottom) / max(top, bottom);
    double vRatio = abs(left - right) / max(left, right);

    return (hRatio < aspectRatioTolerance && vRatio < aspectRatioTolerance);
}


// Альтернативная фильтрация: кластеризация по 4 квадрантам изображения
vector<Point2f> filterByQuadrants(const vector<Point2f>& corners,
                                  int imgWidth, int imgHeight) {
    if (corners.size() < 4) return corners;

    int centerX = imgWidth / 2;
    int centerY = imgHeight / 2;

    // 4 квадранта: [верх-лев, верх-прав, низ-прав, низ-лев]
    vector<vector<Point2f>> quadrants(4);

    for (const auto& pt : corners) {
        int quadrant;
        if (pt.x < centerX && pt.y < centerY) quadrant = 0;      // TL
        else if (pt.x >= centerX && pt.y < centerY) quadrant = 1; // TR
        else if (pt.x >= centerX && pt.y >= centerY) quadrant = 2; // BR
        else quadrant = 3;                                        // BL

        quadrants[quadrant].push_back(pt);
    }

    // Из каждого квадранта берём точку, наиболее близкую к углу изображения
    vector<Point2f> result;
    Point2f imgCorners[4] = {
        Point2f(0, 0),
        Point2f(imgWidth, 0),
        Point2f(imgWidth, imgHeight),
        Point2f(0, imgHeight)
    };

    for (int i = 0; i < 4; i++) {
        if (quadrants[i].empty()) continue;

        Point2f best = quadrants[i][0];
        double minDist = norm(quadrants[i][0] - imgCorners[i]);

        for (size_t j = 1; j < quadrants[i].size(); j++) {
            double dist = norm(quadrants[i][j] - imgCorners[i]);
            if (dist < minDist) {
                minDist = dist;
                best = quadrants[i][j];
            }
        }
        result.push_back(best);
    }

    return result;
}

// ===========================================================

int main(int argc, char** argv) {
    // ================= КОНФИГУРАЦИЯ =================
    string inputPath = "../images/img_002.jpg";
    string outputPath = "../images/shi_tomasi_result.jpg";

    // Параметры алгоритма Shi-Tomasi
    int maxCorners = 200;            // ⚠️ Увеличено для запаса (будет фильтрация)
    double qualityLevel = 0.08;     // ⚠️ Снижено чтобы поймать все углы включая текст
    double minDistance = 100;        // ⚠️ Снижено для плотной детекции
    int blockSize = 5;
    bool useHarris = false;
    double k = 0.04;

    // Параметры фильтрации
    string filterMethod = "border"; // "border" или "quadrant"
    int targetCorners = 4;          // Целевое количество углов листа

    // Параметры отрисовки
    int cornerRadius = 6;
    Scalar allCornersColor = Scalar(255, 0, 0);  // Серый для всех углов
    Scalar sheetCornersColor = Scalar(0, 0, 255);    // Красный для углов листа
    Scalar lineColor = Scalar(0, 255, 0);            // Зелёный для линий
    // =================================================

    // 1. Загрузка изображения
    Mat src = imread(inputPath, IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Ошибка: Не удалось загрузить изображение '" << inputPath << "'" << endl;
        return -1;
    }
    cout << "✓ Изображение загружено: " << src.cols << "x" << src.rows << endl;

    // 2. Предобработка
    Mat gray, blurred;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Инверсия для белого объекта на чёрном фоне
    Mat grayProcessed;
    threshold(gray, grayProcessed,  80, 255, THRESH_BINARY_INV);

    GaussianBlur(grayProcessed, blurred, Size(5, 5), 1.5);
    cout << "✓ Предобработка: Grayscale + Invert + GaussianBlur" << endl;

    // 3. Детекция углов алгоритмом Shi-Tomasi
    vector<Point2f> allCorners;

    goodFeaturesToTrack(
        blurred,
        allCorners,
        maxCorners,
        qualityLevel,
        minDistance,
        noArray(),
        blockSize,
        useHarris,
        k
    );

    cout << "✓ Найдено углов (Shi-Tomasi, все включая текст): " << allCorners.size() << endl;

    // 4. Фильтрация: оставляем только 4 угла листа
    vector<Point2f> sheetCorners;

    if (filterMethod == "border") {
        sheetCorners = filterSheetCorners(allCorners, src.cols, src.rows, targetCorners);
        cout << "✓ Фильтрация по границам: " << sheetCorners.size() << " углов" << endl;
    } else if (filterMethod == "quadrant") {
        sheetCorners = filterByQuadrants(allCorners, src.cols, src.rows);
        cout << "✓ Фильтрация по квадрантам: " << sheetCorners.size() << " углов" << endl;
    }

    // 5. Валидация прямоугольности
    if (sheetCorners.size() == 4) {
        bool isValid = validateRectangle(sheetCorners, 0.5);
        if (isValid) {
            cout << "✓ Углы образуют прямоугольник (валидация пройдена)" << endl;
        } else {
            cout << "⚠ Углы не образуют идеальный прямоугольник (возможно, перспектива)" << endl;
        }
        sheetCorners = orderCorners(sheetCorners);
    }

    // 6. Отрисовка результатов
    Mat result = src.clone();

    // Сначала рисуем ВСЕ найденные углы (серым) — для отладки
    for (const auto& pt : allCorners) {
        circle(result, pt, 10, allCornersColor, -1);
    }
    cout << "✓ Отрисовано всех углов: " << allCorners.size() << endl;

    // Затем рисуем 4 угла листа (красным, крупнее)
    for (size_t i = 0; i < sheetCorners.size(); i++) {
        Point pt = sheetCorners[i];
        circle(result, pt, cornerRadius, sheetCornersColor, FILLED);
        putText(result, to_string(i + 1),
                Point(pt.x - 8, pt.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    }

    // Соединяем углы листа линиями
    if (sheetCorners.size() == 4) {
        for (int i = 0; i < 4; i++) {
            line(result, sheetCorners[i], sheetCorners[(i + 1) % 4], lineColor, 3);
        }
        cout << "✓ Построен четырёхугольник из 4 углов листа" << endl;
    }

    // 7. Сохранение результата
    vector<int> compression_params = { IMWRITE_JPEG_QUALITY, 95 };
    bool saved = imwrite(outputPath, result, compression_params);

    if (saved) {
        cout << "✓ Результат сохранён в: " << outputPath << endl;
    } else {
        cerr << "✗ Ошибка сохранения!" << endl;
        return -1;
    }

    return 0;
}