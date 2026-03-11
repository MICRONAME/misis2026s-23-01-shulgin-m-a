//
// Created by Lenovo on 09.03.2026.
//
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>


/**
 * Находит все контуры на изображении и рисует их
 *
 * @param imagePath Путь к входному изображению
 * @param outputPath Путь для сохранения результата (опционально)
 * @return pair<Mat, vector<vector<Point>>> Изображение с контурами и сами контуры
 */
void findAndDrawContours(const std::string& imagePath,
                                                      const std::string& out, const std::string& out_binary) {
    // Загрузка изображения
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение: " + imagePath);
    }

    // 1. Конвертация в оттенки серого
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat denoised;
    cv::GaussianBlur(gray, denoised, cv::Size(5, 5), 0);

    // 3. Бинаризация с методом Оцу
    cv::Mat binary;
    double thresholdValue = cv::threshold(denoised, binary, 170, 255,
                                       cv::THRESH_BINARY);

    // Сохранение бинарного изображения
    cv::imwrite(out_binary, binary);
    std::cout << "✓ Порог бинаризации: " << thresholdValue << std::endl;

    // 4. Обнаружение краев (Canny)
    //cv::Mat edges;
    //cv::Canny(binary, edges, 50, 150);

    // 5. Поиск контуров
    std::vector<std::vector<cv::Point>> contours;
    std::vector< cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    // 6. Создание результата (копия бинарного изображения)
    cv::Mat result = image.clone();

    // 7. Проверка и закрашивание мелких контуров
    int minArea = 1000000;

    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        //std::cout << area << std::endl;

        if (area >= minArea)
        {
            cv::Rect bbox = boundingRect(contours[i]);
            cv::rectangle(result, bbox, cv::Scalar(0, 255, 0), 10);
            //cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 10);
        }
    }

    // 8. Сохранение результата
    cv::imwrite(out, result);

    /*
    //egor
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::RotatedRect detected_rect;
    cv::Point2f detected_center(-1, -1);
    double detected_solidity = 0.0;
    double contour_area = 0.0;
    cv::Mat final_viz;

    if (!contours.empty())
    {
        // Находим самый большой контур (на случай, если осталось >1)
        size_t best_idx = 0;
        double max_area = 0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                best_idx = i;
            }
        }

        contour_area = max_area;
        detected_rect = cv::minAreaRect(contours[best_idx]);
        detected_center = detected_rect.center;

        // Площадь RotatedRect
        double rect_area = detected_rect.size.width * detected_rect.size.height;
        detected_solidity = (rect_area > 0) ? (contour_area / rect_area) : 0.0;

        // Визуализация
        cv::cvtColor(binary, final_viz, cv::COLOR_GRAY2BGR);
        cv::Point2f vertices[4];
        detected_rect.points(vertices);
        for (int j = 0; j < 4; ++j) {
            cv::line(final_viz, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::circle(final_viz, detected_center, 5, cv::Scalar(255, 0, 0), -1); // синий центр

        cv::imwrite("out.jpg", final_viz);
    }
    */
}


int main() {
    // Пути к файлам
    std::string input_001 = "../src/img_easy_005.jpg";
    std::string input_002 = "../src/not_clear_bg_000.jpg";
    std::string input_003 = "../src/img_easy_004.jpg";
    std::string input_006 = "../src/img_easy_006.jpg";

    std::cout << "========================================" << std::endl;
    std::cout << "Обработка изображения" << std::endl;
    std::cout << "========================================" << std::endl;

    // Вызов функции
    findAndDrawContours(input_001, "../images/img_easy_005_out.jpg", "../images/img_easy_005_binary.jpg");
    findAndDrawContours(input_002, "../images/not_clear_bg_000_out.jpg", "../images/not_clear_bg_000_binary.jpg");
    findAndDrawContours(input_003, "../images/img_easy_004_out.jpg", "../images/img_easy_004_binary.jpg");
    findAndDrawContours(input_006, "../images/img_easy_006_out.jpg", "../images/img_easy_006_binary.jpg");
}