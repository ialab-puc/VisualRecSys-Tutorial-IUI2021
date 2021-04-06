# Tutorial on Visual Recommendation Systems

This page hosts the material for the tutorial on Visual Recommendation Systems which will be
presented at the 2021 ACM Conference on Intelligent User Interfaces (IUI 2021).

## Instructors

* Denis Parra, Associate Professor, PUC Chile
* Antonio Ossa-Guerra, MSc Student, PUC Chile
* Manuel Cartagena, MSc Student, PUC Chile
* Patricio Cerda-Mardini, MSc, PUC Chile & MindsDB
* Felipe del Río, PhD Student, PUC Chile

## Requisites

* Pytorch 1.7
* Python 3.7+

## Benchmark

<sub>
|            | AUC     | RR      | R@20    | P@20    | nDCG@20 | R@100   | P@100   | nDCG@100 | R@200   | P@200   | nDCG@200 |
|------------|---------|---------|---------|---------|---------|---------|---------|----------|---------|---------|----------|
| [1] CuratorNet | .689338 | .013876 | .029684 | .001484 | .016741 | .053803 | .000538 | .020919  | .075139 | .000375 | .023859  |
| [2] VBPR       | .778463 | .021693 | .055658 | .002782 | .026848 | .138218 | .001382 | .041056  | .197588 | .000987 | .049335  |
| [3] DVBPR      |         |         |         |         |         |         |         |          |         |         |          |
| [4] ACF        | .794756 | .020795 | .064007 | .003200 | .027615 | .156771 | .001567 | .043894  | .223562 | .001117 | .053251  |

</sub>

## Program


## Dataset

Just like you, we have been looking for several years some dataset to train our models. For instance, the <a href="#">RecSys dataset collection
by Prof. Julian McAuley at USCD </a> has datasets, but due to copyright issues he only shares embeddings as .npy, but we need images to test if our recommendations are making sense!

## References
[1]
[2]
[3]
[4]
