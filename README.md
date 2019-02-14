# Image-Classification

[![N|Solid](http://www.vision.caltech.edu/Image_Datasets/Caltech256/intro_tight_crop.jpg)](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)

[Presentation Slides](https://docs.google.com/presentation/d/1Gqo_1wfaqPstNLpQPSDsFWu5RrrIWk0vB_t2JzLWw3U/edit?usp=sharing)
# Introduction
In the past, to do object recognition using computers, many classifiers have been developed, such as Support Vector Machines and K-Nearest Neighbors. But with the improvement of the computing power, recent years people start to use artificial neural networks to solve object recognition problems, such as Back Propagation Neural Network and Convolutional Neural Network.

In this project, our goal is to create object recognition systems which can identify which given object catagory is the best match. We built several supervised learning models with different algorithms and tried to analysis their performance.

We used [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) as our dataset, which is a challenging set of 256 object categories containing a total of 30607 images, taken at varying orientations, varying lighting conditions, and with different backgrounds.

To see detailed experiment and results, please refer to our report: [P13_final_report.pdf](https://drive.google.com/file/d/1eID5uI_tdAJrw1yVxVKFWrGlpt_nDO7G/view?usp=sharing)

# Conclusion
* KNN, BPNN and SVM are not suitable to image classification problems. Although we also tried these three algorithms on less categories of images, their performance are still even much worse than the result of using all categories in CNN. But they did some improvement, compared with random guessing.

* CNN is a very ideal algorithm to finish the image classification tasks if the dataset is valid and the parameters are properly set. To get a higher test accuracy, one should try to increase the amount of images in each category. More categories one has, more images in each category are needed. For example, if one has 10 categories, then 500 images for each category may be enough, but if one has 100 categories, then 500 images for each category will not perform as good as the previous one, and 2000 images for each category may be enough.

# Collaborators
[Dongyu Wang](https://github.com/wangdy25), [Tian Shi](https://github.com/tiansss), [Xiao Ma](https://github.ncsu.edu/xma21), [Xinyu Gong](https://github.com/XinyuGong)
