# VisRec: A Hands-on Tutorial on Deep Learning for Visual Recommender Systems

This page hosts the material for the tutorial on **VisRec: A Hands-on Tutorial on Deep Learning for Visual Recommender Systems**,
presented at the 2021 ACM Conference on Intelligent User Interfaces (IUI 2021).

**Schedule**: Tuesday, April 13th 2021, starting at 1:30pm CDT 

## Citation

If you use this material or code and publish something thanks to it, please cite

Denis Parra, Antonio Ossa-Guerra, Manuel Cartagena, Patricio Cerda-Mardini, and Felipe del Rio. 2021. VisRec: A Hands-on Tutorial on Deep Learning for Visual Recommender Systems. In <i>26th International Conference on Intelligent User Interfaces</i> (<i>IUI '21</i>). Association for Computing Machinery, New York, NY, USA, 5–6. DOI:https://doi.org/10.1145/3397482.3450620


## Instructors

* Denis Parra, Associate Professor, PUC Chile
* Antonio Ossa-Guerra, MSc Student, PUC Chile
* Manuel Cartagena, MSc Student, PUC Chile
* Patricio Cerda-Mardini, MSc, PUC Chile & MindsDB
* Felipe del Río, PhD Student, PUC Chile

![speakers-visrec](https://user-images.githubusercontent.com/208111/114323807-f818ba80-9af4-11eb-84ef-428517a4fe60.jpg)

## Requisites

* Python 3.7+
* Pytorch 1.7
* Torchvision

## Program

* (40 mins) [Session 1](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021/blob/main/slides/Session%201%20VisRec%20Introduction%20.pdf): Introduction to Visual RecSys, datasets and feature extraction with CNNs in Pytorch
* (40 mins) [Session 2](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021/blob/main/slides/Session%202%20Pipeline%20%2B%20VisRank%20%2B%20VBPR.pdf): Pipeline for training and testing visual RecSys in Pytorch, application with VisRank and VBPR

(10 mins) [BREAK] 

* (25 mins) [Session 3](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021/blob/main/slides/Session%203%20DVBPR.pdf): Dynamic Visual Bayesian Personalized Ranking (DVBPR) in Pytorch
* (25 mins) [Session 4](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021/blob/main/slides/Session%204%20CuratorNet.pdf): CuratorNet in Pytorch
* (25 mins) [Session 5](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021/blob/main/slides/Session%205%20ACF.pdf): Attentive Collaborative Filtering (ACF) in Pytorch

(5 mins) [BREAK] 

* (10 mins) Conclusion

## Wikimedia Commons Dataset

Just like you, we have been looking for several years for some datasets to train our models. For instance, the <a href="#">RecSys dataset collection
by Prof. Julian McAuley at USCD </a> has datasets, but due to copyright issues he only shares embeddings as .npy and in some cases (such as the Amazon datasets) links to image URLS so you can doonload them on your own. We need images to test if our recommendations are making sense!

We acknowledge the support of [Diego Saez-Trumper](https://wikimediafoundation.org/profile/diego-saez-trumper/) from Wikimedia foundation to collect this dataset.

## Benchmark on Wikimedia Commons Dataset

|            | AUC     | RR      | R@20    | P@20    | nDCG@20 | R@100   | P@100   | nDCG@100 |
|------------|---------|---------|---------|---------|---------|---------|---------|----------|
| [1] CuratorNet | .66931 | .01955 | .03803 | .00190 | .02226 | .07884 | .00078 | .02943  |
| [2] VBPR       | .77846 | .02169 | .05565 | .00278 | .02684 | .13821 | .00138 | .04105  |
| [3] DVBPR      | .83168 | .04507 | .12152 | .00607 | .05814 | .25695 | .00256 | .08245  |
| [4] ACF        | .80409 | .01594 | .05473 | .00273 | .02127 | .14935 | .00149 | .03781  |

## References

[1] Messina, P., Cartagena, M., Cerda, P., del Rio, F., & Parra, D. (2020). CuratorNet: Visually-aware Recommendation of Art Images. arXiv preprint arXiv:2009.04426.

[2] He, R., & McAuley, J. (2016). VBPR: visual bayesian personalized ranking from implicit feedback. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 30, No. 1).

[3] Kang, W. C., Fang, C., Wang, Z., & McAuley, J. (2017). Visually-aware fashion recommendation and design with generative image models. In 2017 IEEE International Conference on Data Mining (ICDM) (pp. 207-216). IEEE.

[4] Chen, J., Zhang, H., He, X., Nie, L., Liu, W., & Chua, T. S. (2017). Attentive collaborative filtering: Multimedia recommendation with item-and component-level attention. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval (pp. 335-344).
