![](images/img_001.jpg)  
<center><i>Рис. 1. img_001. </i></center>  

[json для img_001](images/img_001.json)

![](images/debug_canny_overlap.png)  
<center><i>Рис. 2. Визуализация работы Canny с отмеченными точками эталона (синие), линииями совпадения контура (желтые) и остальными найденными контурами (зеленые) для img_001. </i></center>  

Для оценки качества Canny была посчитана IoU метрика по площади  
| Metric | Value |
|--------|-------|
| IoU | 0.957 |
| Overlap with GT: | 99.3% |
