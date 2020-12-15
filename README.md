# UCLA Statistics 101C Fall 2020 Final Project: Predicting the growth rate of YouTube videos

### Instructor: Alan Vazquez

### Team Cybernetic Ducks: Oscar Monroy, Zoe Wang, Christine Marie Castle

## Competition

> In this project, we aim to predict the percentage change in views on a [YouTube] video between the second and sixth hour since its publishing.

[https://www.kaggle.com/c/stats101c-lec4-final-competition](https://www.kaggle.com/c/stats101c-lec4-final-competition)

## Presentation

[https://www.youtube.com/watch?v=ycBzyMqezXU](https://www.youtube.com/watch?v=ycBzyMqezXU)

## File descriptions

File | Source | Description
--- | --- | ---
[final_report.pdf](final_report.pdf) | Cybernetic Ducks | Final report w/code
[final_final.R](final_final.R) | Cybernetic Ducks | Compilation of code for all the preprocessing, variable selection, modeling
[latex](latex) | Christine Marie Castle | Files used to typeset/format the report including images, code
[training.csv](training.csv) | Ritvik Kharkar | 7242 observations (YouTube videos), 259 features + response (growth_2_6)
[test.csv](test.csv) | Ritvik Kharkar | 3105 observations (YouTube videos), 259 features
[Feature_Descriptions.xlsx](Feature_Descriptions.xlsx) | Ritvik Kharkar | Description of features in original data
[Final.Rmd](Final.Rmd) | Oscar Monroy | Script for cleaning data (used for final model)
[finding_best_corr_value.Rmd](finding_best_corr_value.Rmd) | Zoe Wang | Process of deciding correlation threshold
[actual_actual_final.Rmd](actual_actual_final.Rmd) | Zoe Wang | Variable selection and random forest models
[Uncorrelated_RF.Rmd](Uncorrelated_RF.Rmd) | Oscar Monroy | LASSO and ridge regression models
[clean_training.R](clean_training.R) | Christine Marie Castle | Script for cleaning data (uses `tidyverse` and has date-time feature)
[nontree.R](nontree.R) | Christine Marie Castle | Multiple linear regression, elastic net, PCR, PLS regression models
[training_clean.csv](training_clean.csv) | Cybernetic Ducks | Cleaned training data (produced from actual_actual_final.Rmd)
[test_clean.csv](test_clean.csv) | Cybernetic Ducks | Cleaned test data (produced from actual_actual_final.Rmd)
