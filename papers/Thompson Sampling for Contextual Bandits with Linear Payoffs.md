# [Thompson Sampling for Contextual Bandits with Linear Payoffs](https://arxiv.org/pdf/1209.3352.pdf)

## Introduction
### General Structure for Thompson Sampling
1. a set $\Theta$ of parameters $\tilde\mu$
2. a prior distribution $P(\tilde\mu)$ on these parameters
3. past observations $D$ consisting for the past time msteps
4. a likelihood function $P(r|b, \tilde\mu)$ which gives the probability of reward given a context $b$ and parameter $\tilde\mu$
5. a posterior distribution $P(\tilde\mu|D)\propto$