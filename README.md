# Bayesian Conformal Prediction
This project presents a reproduction study of the Bayesian Conformal Prediction (BCP) method introduced in the Conformal Bayesian Computation paper. BCP combines conformal prediction with Bayesian posterior predictive modeling to provide valid uncertainty quantification, even under model misspecification. We reproduce two key experiments from the paper: sparse regression on the diabetes dataset and binary classification on the Wisconsin breast cancer dataset. In the regression setting, BCP achieved coverage of 81.11\% and average interval width of 1.807 with a misspecified prior ($c=0.02$), closely matching the target coverage of 80\% and outperforming the standard Bayesian method, which only achieved 59.4\% coverage under the same prior. In the classification task, our implementation matched the paper’s results with coverage of 81.2\% and average predictive set size of 0.814. These results confirm that BCP is effective at correcting for model misspecification and that the original findings are reproducible under the reported experimental setup.

## Report

[Click here to view the report (PDF)](Reproducibility_Report _FanyiWu.pdf)
