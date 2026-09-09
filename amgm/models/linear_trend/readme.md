# Linear trend prediction

This model fits a linear line to the training window, calculates 95% confidence interval (CI), and extrapolate it to the test window. Then, it estimates what is the likelihood of price staying within this CI during the testing window. That is to say:

$$Prob(x(t_2, t_3) \in C_{linear}) = NN(x(t_1, t_2), a, b, c, ..., \theta),$$

where, $C_{linear}$ represents the 95% CI of linear trend corridor based on the $x(t_1, t_2)$ which is extrapolated to $t_3$.

We need to decide the list of indicators $(a, b, c, ...)$ we want to feed as input to the NN.
- These features can be based on time periods older than $t_2$ 
- They can be based on different frequencies daily, weekly, … and then ensemble them in some way

# Understanding Confidence Intervals in a Simple Linear Model

A simple linear regression model establishes a relationship between an independent variable $X$ and a dependent variable $Y$ through the equation:

$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:
- $Y$ is the predicted outcome.
- $X$ is the independent variable.
- $\beta_0$ is the intercept.
- $\beta_1$ is the slope of the regression line.
- $\epsilon$ represents the error term (residual).

## Residual Calculation
Residuals measure the difference between observed and predicted values:

$$Residual = Y - \hat{Y}$$

For a given dataset, the standard deviation of residuals $σ_ε$ is computed as:

$$\sigma_{\epsilon} = \sqrt{\sum(Y - \hat{Y})² / (n - 2) }$$

Where $n$ is the number of observations.

## Confidence Interval Based on Residuals
To estimate the confidence interval, we use a multiplier of 1.96 (assuming normality) on the residual standard deviation:

$$CI = \hat{Y} ± 1.96 × \sigma_{\epsilon}$$

Where $\hat{Y}$ is the predicted value from the regression model.
