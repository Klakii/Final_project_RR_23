# Final project for the Reproducible Research Course
Natalia Miela | Krzysztof Kalisiak | Maciej Zdanowicz 
#
In the following project we have decided to take the report done for Unsupervised Learning course, and reproduce its results in Python. The original 
project concerned choosing the best cities for work and vaccation, and can be found here: https://rpubs.com/nmiela/PCA_clustering?fbclid=IwAR01neQh7_GXrdpepowCsHMshqLoOyMY2vmqVkjiZdEOzJK7U641n5sg34Y. 

Project consists of five main parts:
* codes folder with raw Python codes used to perform EDA, PCA, and Clustering,
* PCA_clustering_workation.Rmd file with the original report created in R,
* workation.csv file with data,
* Workation_report.ipynb report created in Jupyter Notebook containing reproduced codes together with comments and comparison of results obtained in Python and R,
* requirements.txt file that contains versions of packages using which we have performed our analysis, these requirements can be easily installed using following code: *$ pip install -r requirements.txt*
#
We found out that Python lacks the proper libraries and/or functions to perform simillar analysis as in R. What can be easily done in R using ready-to-use functions, is 
quite cumbersome in Python - for instance creating a circle plot for correlations or calculating the Hopkins statistics, where we had to resort to custom made function. On the other hand, 
data cleaning and preprocessing is more convenient to perform in Python. Obtained results differ from those obtained in R, mainly due to the lack of predefined random seeds and slightly different 
implementation of functions used for clustering and PCA in Python and R.
