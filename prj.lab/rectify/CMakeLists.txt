add_executable(rectify rectify.cpp)
add_executable(evaluate evaluate.cpp)

target_link_libraries(rectify ${OpenCV_LIBS})
target_link_libraries(evaluate ${OpenCV_LIBS})