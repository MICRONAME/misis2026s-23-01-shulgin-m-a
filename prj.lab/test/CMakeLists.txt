add_executable(test-1 test_1.cpp)
add_executable(test-2 test_2.cpp)

target_link_libraries(test-1 ${OpenCV_LIBS})
target_link_libraries(test-2 ${OpenCV_LIBS})