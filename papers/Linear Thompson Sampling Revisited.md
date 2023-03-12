# [Linear Thompson Sampling Revisited](https://arxiv.org/pdf/1611.06534.pdf)

## Introductions
* The major contributions of this paper are
  1. Following the intuition of [Thompson Sampling for contextual bandits with linear payoffs](Thompson%20Sampling%20for%20Contextual%20Bandits%20with%20Linear%20Payoffs.md), and we show that TS doesn't need to sample from an actual Bayesian posterior distribution
  2. We provide an alternative proof of TS. Regret is related to thegradient of objective function, that is ultimately controlled by the norm of the optimal arms. This shows that $\theta_t$ is chosen, then optimal arm $x_t=\argmax_xx^\top\theta_t$ represents *useful exploration* step
  3. Show howour proof can easily adapted to Generalized Linear Model

## PreLiminaries
### Setting
* We consider the stochastic linear bandit model, and reward is generated as $r(x)=x^\top\theta^*+\xi$
* An arm is evaluated due to its expected reward $x^\top\theta^*$ and for any $\theta$ we denote the optimal arm and its value by
$$x^*(\theta) = \argmax_{x\in\chi}x^\top\theta,\space J(\theta) = \sup_{x\in\chi}x^\top\theta $$
### Notations
* We impose the following assumptions on the problem structure and the noise $\xi_{t+1}$
  1. The arm set $\chi$ is a bounded closed(compact) subset of $\R^d$ such that $||x||\le1$ for all $x\in\chi$
  2. There exists $S\in\R^+$ such that $||\theta^*||\le S$ and S is known
  3. The noise process $\{\xi_t\}_t$ is a martingale differencne sequence given $F_t^x$ and it is conditionally R-subgaussian for some constant $R\ge 0$
* **Proposition 1)** For any $\theta\in(0,1)$ under assumption above, for any $F_t^x$ adapted sequence $(x_1,\dots,x_t)$, the RLS estimator $\hat{\theta}_t$ is such that for any fixed $t\ge 1$, and with probability $1-\delta$,
$$||\hat\theta_t-\theta^*||_{V_t}\le\beta_t(\delta)$$
$$\forall x\in\R^d, |x^\top(\hat\theta_t-\theta^*)|\le||x||_{V_t^{-1}}\beta_t(\delta)$$
$$\beta_t(\delta) = R\sqrt{2\log\frac{(\lambda+t)^{d/2}\lambda^{-d/2}}{\delta}}+\sqrt\lambda S$$
* At step t, we can define ellipsoid centered around $\hat\theta_t$ and radius $\beta_t(\delta/4T)$
* **Proposition 2)** Let $\lambda\ge 1$, for any arbitrary sequence $(x_1,\dots,x_t)\in\chi^t$ let $V_{t+1} = \lambda I+\sum_{s=1}^{t}x_sx_s^\top$, then
$$\sum_{s=1}^{t}||x_s||^2_{V_x^{-1}} \le 2\log\frac{\det(V_{t+1})}{\det(\lambda I)} \le 2d\log\left({1+\frac t \lambda}\right)$$

## Linear Thompson Sampling
* Algorithms
  ```
  Input = theta_1, V_1=lambda*I, delta, T
  Set theta_prime = theta/4T
  for t in range(1, T) do
    Sample n_t in distribution
    Compute parameter tilde(theta_t) = hat(theta_t) + beta_t(theta_prime)V_t^(-1/2)n_t
    Compute optimal arm x_t = argmax(x.T tilde(theta_t))
    Pull arm and observe reward r_(t+1)
    Compute V_(t+1) and hat(theta)_(t+1)
  endfor
  ```
* 