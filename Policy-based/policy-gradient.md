# Policy-based
* value-based agent는 액션을 선택할 때 결정론적(deterministic)임. -> 각 상태에서 선택하는 액션이 항상 동일
* 또한 연속적 액션 공간에서는 액션의 수가 무한이기 때문에 value-based agent가 작동하기 어려움. 
* 반면 policy-based agent는 확률적 정책(stochastic policy)을 취할 수 있고, $\pi(s)$로 액션을 바로 선택할 수 있으므로 연속적 액션 공간에서도 동작함.  

## Policy gradient
* 목적함수 $J(\theta) = \displaystyle\sum_s d(s) * v_{\pi_\theta}(s)$, ($d(s)$는 시작 상태 $s$의 확률 분포)
* 목적함수 $J(\theta)$를 최대화하는 것이 목적.
* $\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)*Q_{\pi_\theta}(s,a)$

* REINFORCE 알고리즘  
> $\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)*G_t$
> >$Q_{\pi_\theta}(s,a)$ 대신 $G_t$  

>Update: $\theta$ <- $\theta + \alpha*\nabla_\theta log\pi_\theta(s,a)*G_t$
>> Return $G_t$에 따라 $log\pi_\theta(s,a)$를 증가 및 감소시키도록 update 됨. 즉, $\pi_\theta(s,a)$를 증가 및 감소시키도록 update 됨.

## Actor-Critic  
* Policy net과 value net을 함께 학습  
* $\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a)*Q_w(s,a)]$  
* Policity network $\pi_\theta$와 value network $Q_w$ 모두 학습  
> $\pi_\theta$가 actor, $Q_w$가 critic  
> $\theta$ Update: $\theta$ <- $\theta + \alpha\nabla_\theta log\pi_\theta(s,a)*Q_w(s,a)$  
> $w$ Update: $w$ <- $w + \beta(r+\gamma Q_w(s',a') - Q_w(s,a))\nabla_w Q_w(s,a)$

## TD Actor-Critic
* Gradient 추정치 변동성 줄임  
* Advantage actor-critic에서 필요한 $Q_w$ 불필요
* Advantage $A(s,a) = Q(s,a) - V(s)$의 불편추정량(unbiased estimate) $\delta = r+\gamma V(s')-V(s)$
> $Q_w$ 없이 $V$만으로 표현  
> $\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log\pi_\theta(s,a) * \delta] $  

* TD Actor-Critic pseudo code
> 1. Policy, value net의 parameter $\theta, \phi$ 초기화  
> 2. 액션 $a~\pi_\theta(a|s)$ 샘플링  
> 3. 스텝마다 A~E 반복
>> A - a를 실행해서 보상 r, 다음 상태 s' 얻음  
>> B - $\delta$ 계산: $delta$ <- $r+\gamma V_\phi(s')-V_\phi(s)$  
>> C - $\theta$ Update: $\theta$ <- $\theta+\alpha_1\nabla_\theta log\pi_\theta(s,a)*\delta$  
>> D - $\phi$ Update: $\phi$ <- $\phi + \alpha_2 \delta \nabla_\phi V_\phi(s)$  
>> E - $a$ <- $a'$, $s$ <- $s'$   
