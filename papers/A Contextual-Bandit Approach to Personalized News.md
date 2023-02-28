# A Contextual-Bandit Approach to
Personalized News Article Recommendation

날짜: 2023년 2월 28일 오후 3:42
링크: http://rob.schapire.net/papers/www10.pdf
상태: 완료
키워드: Contextual Bandit, Exploration/Exploitation dilemma, Personalization, Recommender System, Web Service

## Introduction

- 사용자에게 가장 적합한 컨텐츠를 보여주도록 하는 문제를 해결하고자 하는 논문임
- 전통적인 추천 시스템의 경우에는, 사용자의 과거 기록을 바탕으로 문제를 해결하는데 이제 컨텐츠는 계속 변화한다는 것이 문제가 될 수 있음
- 그래서 필요한 것이 $\epsilon$-greedy 알고리즘과 같은 exploration 알고리즘이며, 이를 얼마나 활용할 지에 대한 알고리즘인 LinUCB를 새롭게 제안하려 함

---

## Formulation & Related Work

### A Multi-armed Bandit Formulation

- 기존의 contextual bandit 알고리즘은 $t$번의 시도에서 다음과 같은 과정을 거친다.
    1. $u_t$와 arm의 집합인 $A_t$를 feature vector $x_{t,a}$로 구해낸다. 이 때 $a \in A_t$일 것이며, $u_t$와 $a$를 context라 할 것이다.
    2. 이전의 trial을 기준으로 알고리즘은 best arm을 구할 것이며 이를 통한 reward인 $r_{t, a_t}$를 얻어낼 것이다.
    3. 이후 지금까지 **선택되지 않은** arm중 하나를 선택하여 exploitation이 진행될 것이다.
- Optimal expected T-trial payoff
    
    $$
    E[\sum_{t=1}^{T} r_{t, a_{t^*}}]
    $$
    
- T-trial Regret of algorithm A
    
    $$
    R_A(T) \stackrel{\tiny{\text{def}}}{=} \space E[\sum_{t=1}^T r_{t, a_{t^*}}] - E[\sum_{t=1}^T r_{t, a_{t}}] 
    $$
    
- K-armed bandit이라고 하는, t에 상관없이 모든 feature vector이 일치하는 경우를 context-free bandit이라고 부른다.

### Existing Bandit Algorithms

- Regret을 최소화하기 위해서는 과거의 기록 중 가장 좋은 것을 이용하는 exploitation이 필요하지만, 더 많은 정보를 얻기 위해서는 exploration도 필요하다. 얘는 suboptimal arm을 선택하기 때문에 regret이 증가할 수 있고, tradeoff가 필요하다.
- 이제 trial에 따라 reward가 변하는 경우에서의 $\epsilon$-greedy algorithm을 생각해보자. 각각의 arm이 무한번 수행된다면 $\hat{\mu}_{t,a}$는 실제 value인 $\mu_a$에 근접할 것이며, $\epsilon$이 감소함에 따라 regret인 $R_A(T)/T$는 1의 확률로 0으로 converge할 것이다.
- 그리고 Upper Confidence Bound(UCB) 알고리즘의 경우는 실제 value를 모르기 때문에 $\hat{\mu}_{t,a}$를 가지고 confidence interval $c_{t,a}$를 추정하게 되는데, 이 때 $a_t = \argmax_{a}(\hat{\mu}_{t,a}+c{t,a})$가 된다.
- 하지만 이를 계산하기 위한 complexity가 exponential하기 때문에, regret을 $\tilde{O} (\sqrt{T})$만큼으로 계산함이 어려워진다. epoch-greedy나 linRel 알고리즘 역시도 그러한 문제를 가지고 있다.

---

## Algorithm

### LinUCB with Disjoint Linear Models

- **Disjoint** Model has coeffecient vector $\theta^*_a$ with $E[r_{t,a}|x_{t,a}]=x_{t,a}^{\top}\theta_a^*$
- Let $D_a$ be a design matrix at trial t and apply ridge regression to the training data $(D_a,c_a)$

$$
\hat\theta_a = (D_a^{\top}D_a+I_d)^{-1}D_a^{\top}c_a
$$

- This can be shown with probability at least $1-\delta$, and $\alpha = 1 + \sqrt{\ln(2/\delta)/2}$

$$
|x_{t,a}^{\top}\hat{\theta}_a - E[r_{t,a}|x_t,a]| \leq \alpha \sqrt{x_{t,a}^\top(D_a^\top D_a + I_d)^{-1}x_{t,a}}
$$

- 이를 반영하면, UCB의 arm selection에서는 다음 arm을 선택해야 할 것이다.

$$
a_t \stackrel{\text{def}}{=}\argmax_{a \in A_t} (x_{t,a}^\top \hat\theta_a + \alpha \sqrt {x_{t,a}^\top A_a^{-1}x_{t,a}})
$$

```python
def LinUCBdisjoint(alpha):
	for t in range(T):
		#observe features of all arms a, x
		for a in A_t:
			if a is new:
				A_a = I_d
				b_a = 0_d*1
			theta_a = inverse(A_a)b_a
			p(t, a) = theta_a
		a_t = argmax(p_t, a)
		A_a_t += x(t,a)x(t,a)^T
		b_a_t += r(t)x(t,a)
```

- Property of this algorithm
    1. Computational Complexity is linear
    2. Algorithm works well for dynamic arm set
    3. We can adapt this algorithm to show if arm set is fixed and contains K arms, then confidence interval decreases fast enough and prove strong regret bound

### LinUCB with Hybrid Linear Models

- **Hybrid** model에서는 disjoint에다가 linear term 하나를 더 추가해준다.

$$
E[r_{t,a} |x_{t,a}] = z_{t,a}^\top \beta^* + x_{t,a}^\top \theta_a^*
$$

- 이제 더 복잡한 알고리즘을 사용할 것이다

```python
def LinUCBHybrid(alpha):
	A_0 = I_k
	b_0 = 0_k
	for i in range(T)
		#observe features of all arms a
		beta_hat = inverse(A_0)b_0
		for a in A_t:
			if a is new:
				A_a = I_d
				B_a = 0_d*k
				b_a = 0_d*1
			theta_a = inverse(A_a)(b_a - B_a*beta_hat)
			s(t,a) = z(t,a).T inverse(A_0) z(t,a) - 2z(t,a).T inverse(A_0) B_a.T inverse(A_a) x(t,a) + x(t,a).T inverse(A_a)x(t,a) + x(t,a).T inverse(A_a) B_a inverse(A_0)B_a.T inverse(A_a) x(t,a)
			p(t,a) = z(t,a).T * beta_hat + x(t,a) * theta_a + alpha * sqrt(s(t,a))
		a_t = argmax(p(t, a))
		A_0 += B_a_t.T inverse(A_a_t) B_a_t
		b_0 += B_a_t.T inverse(A_a_t) B_a_t
		A_a_t += x_t,a x_t,a.T
		B_a_t += x_t,a x_t,a.T
		b_a_t += x_t,a x_t,a.T
		A_0 += z_t,a z_t,a.T - B_a_t.T inverse(A_a_t) B_a_t
		b_0 += z_t,a z_t,a.T - B_a_t.T inverse(A_a_t) B_a_t
```

- 이 경우의 computational complexity는 $O(d^2+k^2)$

---

## Evaluation Methodology

- Goal: measure the performance of bandit algorithm, that is, a rule for selecting an arm at each time step based on preceding interactions
- One way is build a simulator to model the bandit process from the logged data then evaluate our algorithm → this will introduce bias, so we need unbiased thing
- Policy evaluator

```python
def policyEvaluator(T, pi):
'''
pi is stream of event
'''
h_0 = set()
R_0 = 0
for t in range(T):
	while pi(h_t-a, (x_1,...x_k)) != a:
		get next event (x_1,..x_k,a,r_a)
	h_t = concat(h_(t-1), (x_1,...,x_k,a,r_a))
	R_t = R_(t-1)+r_a
return R_T/T
```

- Theorem: for all distribution D of contexts, all policies pi, all T and all sequences of events h_T where S is streams of events(iid.) The expected number of events obtained from the stream to gather a history of length T is KT

$$
\Pr_{\text{Policy Evaluator}(\pi, S)}(h_T) = \Pr_{\pi, D}(h_T)
$$
