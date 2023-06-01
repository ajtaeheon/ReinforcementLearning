# Value-based Agent_가치 기반 에이전트
* Value function으로 액션 결정  
* Model-free 상황에서는 $v(s)$만으로 액션 결정 불가 (reward, state transtion probability 모르기 때문)  
* action-value function $q(s,a)$로 액션 결정

### Loss 함수 정의
> * Value network $v_{\theta}(s)$ 정의 : 상태 $s$를 input으로 넣으면 $s$의 가치가 output으로 나옴. $\theta$는 parameter.  
> * $L(\theta)=E_\pi[(v_{true}(s)-v_{\theta}(s))^2]$  
> * $\nabla_{\theta} L(\theta)=-E_\pi[(v_{true}(s)-v_{\theta}(s))\nabla_{\theta} v_{\theta}(s)]$  
> * Update: $\theta' = \theta-\alpha\nabla_{\theta} L(\theta) =\theta+\alpha(v_{true}(s)-v_{\theta}(s))\nabla_{\theta} v_{\theta}(s)$
  
### $v_{true}(s)$의 대안
* $v_{true}(s)$는 model-free 환경에서 알 수 없음
> * Sol 1. MC return 사용: $\theta' = \theta+\alpha(G_t-v_{\theta}(s_t))\nabla_{\theta} v_{\theta}(s_t)$  
> * Sol 2. TD target 사용: $\theta' = \theta+\alpha(r_{t+1}+\gamma v_{\theta}(s_{t+1})-v_{\theta}(s_t))\nabla_{\theta} v_{\theta}(s_t)$   
>> TD target 사용할 때 $r_{t+1}+\gamma v_{\theta}(s_{t+1})$는 상수 취급해야 정답 값이 바뀌지 않고 안정적 학습 가능 -> tensor에 대해 detach 함수 호출하여 구현 가능  

### Q-Learning
* 벨만 최적방정식으로 $Q_*(s,a)$ 학습  
> * 정답: $r+\gamma \displaystyle\max_{a'} \textstyle Q(s',a')$  
> * Update: $\theta' = \theta+\alpha(r+\gamma \displaystyle\max_{a'} \textstyle Q_{\theta}(s',a')-Q_{\theta}(s,a))\nabla_{\theta} Q_{\theta}(s,a)$ 

### DQN  
* 추가 아이디어  
> 1. Experience Replay: Replay buffer를 둬서 buffer에 최근 데이터 n개를 저장해놓고, 학습할 때 임의로 데이터 뽑아서 사용.  
> -> 데이터 재사용 및 데이터 간 상관성 감소  
> 2. 별도의 target network: target network를 따로 둬서 학습 중인 Q network와 parameter를 공유하지 않도록 얼려놓고, 일정 주기마다 parameter를 Q network의 parameter로 업데이트.   
> -> 안정적 학습  
