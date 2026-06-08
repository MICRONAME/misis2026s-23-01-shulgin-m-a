![](../images/img_001.jpg)  
<center><i>Рис. 1. img_001. </i></center>  

[json для img_001](../src/img_001.json)

![](../images/debug_canny_overlap.png)  
<center><i>Рис. 2. Визуализация работы Canny с отмеченными точками эталона (синие), линииями совпадения контура (желтые) и остальными найденными контурами (зеленые) для img_001. </i></center>  

![](../images/img_057.jpg)  
<center><i>Рис. 3. img_057. </i></center>  

[json для img_057](../src/img_057.json)

![](../images/debug_canny_overlap_057.png)  
<center><i>Рис. 4. Визуализация работы Canny с отмеченными точками эталона (синие), линииями совпадения контура (желтые) и остальными найденными контурами (зеленые) для img_057. </i></center>  

![](../images/img_015.jpg)  
<center><i>Рис. 5. img_015. </i></center>  

[json для img_015](../src/img_015.json)

![](../images/debug_canny_overlap_015.png)  
<center><i>Рис. 6. Визуализация работы Canny с отмеченными точками эталона (синие), линииями совпадения контура (желтые) и остальными найденными контурами (зеленые) для img_015. </i></center>  

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

Для img_015:  
| Metric | Value |
|--------|-------|
| IoU | 0.759 |
| Overlap with GT: | 94.4% |
