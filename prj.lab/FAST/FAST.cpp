#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;

// Функция для упорядочивания 4 точек: Верх-Лев, Верх-Прав, Низ-Прав, Низ-Лев
vector<Point2f> orderPoints(vector<Point2f> pts) {
    vector<Point2f> ordered(4);

    // Сортировка по сумме координат (x + y)
    vector<pair<Point2f, float>> sums;
    for (const auto& p : pts) {
        sums.push_back({p, p.x + p.y});
    }
    sort(sums.begin(), sums.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    ordered[0] = sums[0].first;     // Top-Left
    ordered[2] = sums[3].first;     // Bottom-Right

    // Сортировка по разности координат (x - y)
    vector<pair<Point2f, float>> diffs;
    diffs.push_back({sums[1].first, sums[1].first.x - sums[1].first.y});
    diffs.push_back({sums[2].first, sums[2].first.x - sums[2].first.y});

    sort(diffs.begin(), diffs.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    ordered[1] = diffs[0].first;    // Top-Right
    ordered[3] = diffs[1].first;    // Bottom-Left

    return ordered;
}

int main(int argc, char** argv) {
    // 1. Загрузка изображения
    string filename = (argc > 1) ? argv[1] : "../src/img_easy_005.jpg";
    Mat img = imread(filename, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Ошибка: Не удалось загрузить изображение " << filename << endl;
        return -1;
    }

    // 2. Преобразование в оттенки серого
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 3. Детекция особенностей FAST
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(
        70,
        true,
        FastFeatureDetector::TYPE_9_16
    );
    vector<KeyPoint> keypoints;
    fast->detect(gray, keypoints);

    // Преобразование KeyPoint в Point2f
    vector<Point2f> featurePoints;
    for (const auto& kp : keypoints) {
        featurePoints.push_back(kp.pt);
    }

    cout << "Найдено точек FAST: " << featurePoints.size() << endl;

    if (featurePoints.size() < 4) {
        cerr << "Ошибка: Найдено меньше 4 точек для построения оболочки." << endl;
        return -1;
    }

    // 4. Построение выпуклой оболочки
    vector<Point2f> hull;
    convexHull(featurePoints, hull);

    cout << "Точек в выпуклой оболочке: " << hull.size() << endl;

    // 5. Аппроксимация оболочки до полигона
    vector<Point> hullInt;
    for (const auto& p : hull) {
        hullInt.push_back(Point(cvRound(p.x), cvRound(p.y)));
    }

    vector<Point> approx;
    double epsilon = 0.04 * arcLength(hullInt, true);
    approxPolyDP(hullInt, approx, epsilon, true);

    cout << "Точек после аппроксимации: " << approx.size() << endl;

    // 6. Проверка на 4 угла
    if (approx.size() == 4) {
        cout << "Успешно найдено 4 угла!" << endl;

        // Преобразование в Point2f и упорядочивание
        vector<Point2f> corners;
        for (const auto& p : approx) {
            corners.push_back(Point2f((float)p.x, (float)p.y));
        }
        vector<Point2f> orderedCorners = orderPoints(corners);

        // 7. Отрисовка результатов
        Mat output = img.clone();

        // Рисуем все точки FAST (мелкие зеленые)
        for (const auto& kp : keypoints) {
            circle(output, Point(kp.pt), 2, Scalar(0, 255, 0), -1);
        }

        // Рисуем выпуклую оболочку (синяя линия)
        vector<Point> hullDraw;
        for (const auto& p : hull) {
            hullDraw.push_back(Point(cvRound(p.x), cvRound(p.y)));
        }
        polylines(output, hullDraw, true, Scalar(255, 0, 0), 2);

        // Рисуем 4 угла (красные круги + подписи)
        vector<string> labels = {"TL", "TR", "BR", "BL"};
        for (int i = 0; i < 4; i++) {
            Point pt = Point(orderedCorners[i]);
            circle(output, pt, 10, Scalar(0, 0, 255), -1);
            putText(output, labels[i], pt, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2);
        }

        // Соединяем углы линиями (желтый четырехугольник)
        vector<Point> quad;
        for (const auto& p : orderedCorners) {
            quad.push_back(Point(cvRound(p.x), cvRound(p.y)));
        }
        polylines(output, quad, true, Scalar(0, 255, 255), 3);

        imwrite("../images/FAST_easy_005.jpg", output);
    }

    return 0;
}