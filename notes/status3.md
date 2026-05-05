![](images/img_001.jpg)  
<center><i>Рис. 1. img_001. </i></center>  

[json для img_001](images/img_001.json)

![](images/debug_canny_overlap.png)  
<center><i>Рис. 2. Визуализация работы Canny с отмеченными точками эталона (синие), линииями совпадения контура (желтые) и остальными найденными контурами (зеленые) для img_001. </i></center>  

![](images/img_057.jpg)  
<center><i>Рис. 1. img_001. </i></center>  

[json для img_057](images/img_057.json)

![](images/debug_canny_overlap_057.png)  
<center><i>Рис. 2. Визуализация работы Canny с отмеченными точками эталона (синие), линииями совпадения контура (желтые) и остальными найденными контурами (зеленые) для img_057. </i></center>  

**Для оценки качества Canny была посчитана IoU метрика по площади  **

Для img_001:  
| Metric | Value |
|--------|-------|
| IoU | 0.957 |
| Overlap with GT: | 99.3% |

Для img_057:  
| Metric | Value |
|--------|-------|
| IoU | 0.615 |
| Overlap with GT: | 100.0% |
