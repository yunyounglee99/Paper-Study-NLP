# Titans: Learning to Memorize at Test Time


# 1. Introduction

- 트랜스포머는 attention기반 모델로써 sequence modeling에서 SOTA를 달성해오며 각종 downstream task에 적용되어왔음
- 문제는 quadratic한 time, memory complexity - 복잡한 real-world task에 적용시키기에는 큰 문제가 생김
- 이 문제를 해결하기 위해서는 소프트맥스를 커널함수로 대체한 linear-transformer도 등장함
    - 메모리 효율성, 계산 비용을 낮춤 → 긴 context 처리도 가능해짐
    - 반대로 긴 context를 처리하기 위해 데이터를 압축하는데 이로 인해 정보 손실이 생김
    - 그래서 긴 context를 처리하는데 있어 효율성과 계산 비용을 낮췄으나, 성능 개선은 정보손실 문제로 인해 이루어지지 않는 모순을 가지게 됨
- 그래서 이 논문에서는 기존의 모델들 (LSTM, transformer)등의 구조적인 문제를 비판함
    - 기존 모델들의 단점
        1. generalization : train data외의 데이터에서 성능을 발휘
        2. length extrapolation : 학습된 context의 길이보다 긴 context에서 안정적인 성능을 발휘하는지
        3. reasoning : 복잡한 문제를 논리를 통해 설명할 수 있는지
    - human brain에서 영감을 받았다기엔 부족한 점이 있음
        1. 학습과정에서 필요한 중요한 요소들의 부족 : short-term memory, long-term memory, meta-memory, attending to current memory 등
        2. 이런 구성요소들의 독립적이면서도 상호연결된 시스템
        3. 데이터를 능동적으로 학습하며, 과거는 추상화하여 기억하는 시스템

### Memory

- human learning에 있어서 memory는 아주 중요한 역할을 함.
- human memory는 단 하나의 single function이 아닌, 여러 시스템들(short-term memory, long-term memory등)이 결합하여 서로 상호작용하고 또 서로 독립적으로 작업을 수행함
- 이런 이유로 memory를 압축하거나, vector화 하여 memory module에서 retrieve를 수행하여 출력을 하도록 하는 많은 모델들이 생겨남
- 이러한 memory 측면의 관점에서 논문의 저자들은 의문점을 가지게 됨
    
    Q1. 메모리를 위한 좋은 구조가 무엇인가
    
    Q2. 적합한 memory update mechanism이 무엇인가
    
    Q3. good memory retrieval 구조가 무엇인가
    
    Q4. 어떻게하면 효율적인 구조를 디자인하고 서로 다른 메모리 시스템을 한데 통합할 수 있는가
    
    Q5. 장기 기억을 저장하고 기억하기 위해서는 deep memory module이 필요한가
    

### Contributions and Roadmap

**Neural memory :** human long-time memory에서 영감을 받아, test time에서 data를 memorize하는 방법을 parameter로 학습하는 방법을 제시함.

이 논문에서는 associative memory loss를 통해 ‘surprise’를 측정하여 more memorable한 event를 저장하는 memory 모듈을 만들고 이를 효율적으로 처리하기 위해 decaying mechanism을 도입함

**Titans Architechture :** 세 개의 hyper-head로 이루어진 모델 구조 디자인

1. Core : short-term memory module + 데이터 처리 프로세스의 main flow (attention with limited window size 사용)
2. Long-term memory : 지난 과거의 memory를 효율적으로 저장하고 기억하기 위한 모듈
3. Persistence memory : learnable but date-independent한 task에 대한 지식을 학습

context, layer, gated branch로 통합된 Titan 변형타입도 소개할거다

**Experimental results :** language modeling, commonsence reasoning, recall-intensive, needle in heystack, time series forecasting, DNA modeling task들에 대해 실험을 진행함

모든 테스크에서 기존 모델의 성능을 뛰어넘음 특히, same window size에서 transformer의 성능보다 우수함을 보였다고 한다.

# 2. Preliminaries

- 기존 연구는 메모리를 Key-Value 기반의 연상 메모리(Associative Memory)로 봄
    1. Hebbian Learning: 연관성을 기반으로 메모리 학습.
    2. Delta Rule: 과거 값을 제거하고 새로운 값을 추가.
- 하지만 이러한 규칙들을 문제들이 존재했음
    - 기존 모델은 ‘momentary surprise’ 에만 초점을 맞추며, 시퀀스 내 token flow를 제대로 고려하지 못함.
    - Forget Gate(망각 게이트)가 부족하여 메모리 관리가 비효율적.
- Titans는 위의 한계를 해결하기 위해 세가지 목표를 설정
    1. 효율적인 메모리 관리 ****: Forget Mechanism 및 Meta-Memory 설계로 메모리의 효율성 개선
    2. 데이터 추상화 및 일반화 ****: 메모리를 단순 저장 공간이 아닌 학습 가능한 구조로 설계
    3. 토큰 간 흐름 반영 ****: 시퀀스 내 정보 흐름을 학습하는 구조를 도입

# 3. Learning to Memorize at Test Time

## 3. 1. Long-term Memory

- 기존의 LLM처럼 neural long-term memory module을 디자인하기 위해서는 abstract한 past history를 encoding할 수 있어야함
    
    → 하지만 일반화 성능 부족, 보안문제, test-time에서의 성능 부족의 문제가 발생함
    
    → 그렇기 때문에 근본적으로 test time에서 **메모리를 어떻게 기억하고 망각할 것인지를 학습**하도록 함
    

### Learning Process and Surprise Metric

- neuropsychology에 따르면, 사람은 예상을 벗어나는 event에 대해 더 잘 기억하는 경향이 있음
- 이것에 영감을 받아서 ‘surprise’라는 것을 정의하고 이것의 gradient를 계산할 수 있게 함
    - gradient가 클수록 이전의 data와 많이 다른 input data임
    
    $\mathcal{M}_t=\mathcal{M}_{t-1}-\theta_t\nabla{\ell}(\mathcal{M}_{t-1};x_t)$
  
    
    - 여기서 $\nabla{\ell}(\mathcal{M}_{t-1};x_t)$가 surprise를 의미
    - 하지만 이런 형태는 big surprise(gradient 큼)이 오면 이전 data들은 missing되기 쉬움(local minima)
    - human memory 관점에서도 특정 event가 처음에는 big surprise였을지라도, 오랜 기간동안 지속적으로 놀라움을 주기는 힘듦
    - 그래서 surprise metric을 past surprise와 momentary surprise로 나눔
    
    $\mathcal{M}_t=\mathcal{M}_{t-1}+S_t$
    
    $S_t=\eta_t\ S_{t-1}-\theta_t\nabla{\ell}(\mathcal{M}_{t-1};x_t)$
    
    - 여기서 $S_{t-1}$이 past surprise, $\nabla{\ell}(\mathcal{M}_{t-1};x_t)$가 momentary surprise
    - $\eta_t$는 data-dependent surprise decay(a function of $x_t$)로 시간에 따른 surprise의 감소 비율을 나타냄
        - $\eta_t=0$일 때 : 현재 context와 연관이 없음 → surprise가 현재 단계에서 완전히 무시됨
        - $\eta_t = 1$일 때 : 현재 context가 과거 context와 강하게 연관이 있음 → surprise가 완전히 유지됨
    - $\theta_t$는 momentary surpirse를 final surprise에 얼마나 많은 양을 줄 것이냐를 결정함
- 흥미롭게도 이 formulation은 gradient with momentum과 굉장히 유사한 형태를 가지고 있음 다만 backward 과정에서 일어나는 것이 아님!
    
    
    |  | Titans | Momentum |
    | --- | --- | --- |
    | formulation | $\mathcal{M}_t=\mathcal{M}_{t-1}+S_t$
    $S_t=\eta_t\ S_{t-1}-\theta_t\nabla{\ell}(\mathcal{M}_{t-1};x_t)$ | $x_t = x_{t-1}-\alpha v_{t}$
    $v_t = \rho v_{t-1}+\nabla f(x_t)$ |
    | element | $S_t$ | $v_t$ |
    | role | as a memory of surprise across time(sequence) | as a memory of velocity across time(sequence) |

### Objective

- surprise metric은 loss function $\ell$을 사용해서 학습
- loss의 목표는 memory가 어떻게 작동하는지를 학습하기 → meta-memory 역할
    - **meta-memory** : 내가 무엇을 기억하고 있고 무엇을 잊었는지 알고, 어떻게 하면 기억할 수 있는지 스스로 아는 능력
- past data를 (key, value) pair로 저장하기 위해서 transformer의 방법을 차용함

$\mathbf{k}_t = x_tW_K$,  $\mathbf{v}_t=x_tW_V$  (where,  $W_K, W_V \in \mathbb{R}^{d_m \times d_m}$)

- transformer와 동일하게  $x_t$를 key와 정보를 담는 value로 변환함
- associated memory : 주어진 input에 대해 가장 유사한 past data를 검색하는데 사용

$\ell(\mathcal{M}_{t-1};x_t) = \|\mathcal{M_{t-1}}(\mathbf{k}_t)-\mathbf{v}_t\|^{2}_{2}$

- memory module의 값과 value값의 mse loss
- meta-memory module 안에서 돌아감
- test time에서 key와 value를 어떻게 mapping하면 좋을지를 학습
- inner loop에서 $\mathcal{M}$의 weight도 optimize함
- outer loop에서는 전체 구조의 나머지 파라미터 optimize

<transformer와 차이점>

|  | Titans | Transformer |
| --- | --- | --- |
| loss | MSE-Loss | Cross-Entropy Loss
(’attention is all you need’논문에 정확히 나와있는 건 아니지만 pytorch 공식 문서에서 확인해보면 CE-Loss를 사용한 것을 확인할 수 있음) |
| input | 현재 input $x_t$, memory state $\mathcal{M}_t$ | 이전 토큰 sequence $y_{<t}$ |
| purpose | 메모리를 저장하는 방법을 학습 | 확률 기반 다음 토큰 예측 |

### Forgetting Mechanism

- 대규모 sequence를 처리할 때 memory문제는 중요함
- 효율적인 memory 관리를 위해 adaptive forgetting mechanism 도입

$\mathcal{M}_t=(1-\alpha_t)\mathcal{M}_{t-1}+S_t$

$S_t=\eta_t\ S_{t-1}-\theta_t\nabla{\ell}(\mathcal{M}_{t-1};x_t)$

- $\alpha_t \in [0,1]$ : Gating mechanism, 얼마나 많은 정보를 잊어버릴건지를 결정
- $\alpha_t=0$이면 forgetting 없이 memory module 업데이트
- $\alpha_t = 1$이면 이전 memory 모두 망각
- RNN의 gating mechanism과 유사하나 동적으로 flexible하게 조정할 수 있다는 것이 Titans의 큰 장점

### Memory Architecture

- long-term memory에서는 단순 MLP 만을 사용
- **왜 MLP냐?**
    1. 기존 vector or metrix-valued memory는 input을 압축하여 linear regression로 optimize 해버림 → linear 의존성을 가질수 밖에
    2. non-linearity로 더 복잡한 패턴 학습 가능 → 더 높은 표현력을 가짐

### Retrieving a Memory

- 단순 forward pass(without weight update)를 사용해서 memory module에서 query retrieve 시행

$y_t=\mathcal{M}^*(\mathbf{q}_t)$

- $\mathbf{q}_t=x_tW_Q$

## 3. 2. How to Parallelize the Long-term Memory Training

- 이론적으로 long-term memory module은 $\mathcal{O}(N)$의 복잡성을 가짐
    
    → GPU, TPU를 활용할 수 있도록 병렬화 필요
    
- mini batch 안에서 gradient descent를 최적화하는 것을 목표로함
- sequence를 chunk size $b \ge 1$ 로 split하여 mini batch gradient descent 계산을 아래 수식과 같이 계산함

$\mathcal{M}_t = (1-\alpha_t)\mathcal{M}_{t-1} - \theta_t\nabla \ell(\mathcal{M}_{t-1};x_t) = \beta_t\mathcal{M}_0 - \sum_{i=1}^t\theta_t{\frac{\beta_t}{\beta_i}}\nabla \ell(\mathcal{M}_{t^{\prime}};x_i)$

- $t^{\prime}=t - \mathsf{mod}(t, b)$
- $\beta_t = \prod_{j=1}^i (1-\alpha_j)$
- 그리고 $t=b$ ($t^{\prime}=0)$에서 MLP 단일 layer($\mathcal{M}_t= W_t$ 인 linear progress)에서의 loss 계산을 통해 아래와 같은 식을 계산할 수 있음

$\nabla \ell(W_0;x_t) = (W_0x_t-x_t)x_t^{\top}\ \Rightarrow \  \sum_{i=1}^t\theta_t{\frac{\beta_t}{\beta_i}}\nabla \ell(W_0;x_i) = \Theta_b \mathbf{B}_b(W_0X-X)X^{\top}$

- $\Theta_b = \mathsf{diag}([\theta_1 \ \ \theta_2 \ \ ... \ \ \theta_b])$
- $\mathbf{B}_b$ : 각 청크에 대한 gradient 보정 행렬($\frac{\beta_t}{\beta_i}$와 유사)
- 따라서 surprise matric을 병렬적으로 한번에 계산이 가능해짐

### Parameters as the Function of Chunks

<img width="468" alt="스크린샷_2025-02-06_오후_7 14 46" src="https://github.com/user-attachments/assets/a06cdd47-31a7-4893-8880-78567be7139d" />


- input $x_t$에 dependent한 parameter $\alpha_t, \  \theta_t, \ \eta_t$ 를 사용하는 것이 아닌
- 각 청크 내에서는 constant(same value) $\alpha, \ \theta, \ \eta$를 사용
    
    → $\Theta$는 single scalar가 됨!
    
    → surprise계산을 청크 내에서 더 빠르게 계산이 가능하며, 각 파라미터들은 learable 하나 청크 내에서 time-invariant하게 됨
    
    → 시간에 대해 독립적인 linear time independent(LTI) system이 됨
    
    → global convolution 가능
    

## 3. 3. Persistent Memory

- long-term memory는 contextual memory로도 볼 수 있는데 이는 output이 fully depend on the context
- 추가적으로 learnable하지만 input-independent parameter를 사용해서 task-related memory로 작동하도록 함 → persistent or meta-memory

$x_{new} = [p_1 \ \ p_2 \ \ ... \ \ p_{N_p}] \ || \ x$

- $N_p \ge 1$
- learnable parameters $P = [p_1 \ \ p_2 \ \ ... \ \ p_{N_p}]$
- $||$ : concatenation

### Memory Perspective

- long-term memory는 contextual memory이고 parameter가 input-dependent함.
- 그러나 효과적인 memory system이기 위해서는 input-independent한 parameter가 필요 → task 지식의 abstraction이 필요
- task 별로 각 task가 어떻게 수행되어야하는지에 대한 지식의 memorization이 필요

### Feedforward Network Perspective

- transformer에서 attention 뒷단의 fully connected layer는 attention weight과 유사하나 data-independent함
- FFN에서 ReLU를 Softmax함수로 변환하여 parameter가 data-independent하면서 attention-like함을 가져감

$FFN(x) = W_V \  \mathsf{Softmax} \ (W_Kx)$

- attention-like하며 data-independent
    - 고정된 weight을 사용하기 때문

### Technical Perspective

- attention with casual mask는 sequence의 첫 토큰에 과하게 반응하는 bias 문제가 존재함
- 앞선 sequence 시작의 learnable parameter들은 이러한 문제를 완화를 하는데 도와줌

# 4. How to Incorporate Memory

- Transformer에서 key, value는 associative memory block으로 해석할 수도 있지만 또한 제한된 context window 때문에 current한 context에만 집중되는 short-term memory라고도 불릴수 있음
- 여러가지 Titans 모듈들을 소개하면서 long term memory가 어떻게 작동하는지를 알아보자

## 4. 1. Memory as a Context(MAC)

<img width="584" alt="스크린샷_2025-02-08_오후_1 20 59" src="https://github.com/user-attachments/assets/7983d3a4-6c42-4b34-9357-3b842aa260cb" />


- memory를 current information의 context로 간주함
- input : long sequence $x \ \in \ \mathbb{R}^{N \times d_{in}}$
- input을 고정된 사이즈의 segement $\mathsf{S}^{(i)} \ \ \ (i = 1,...,N/C)$로 나눔
    - 즉, 현재 시점의 segmented input $\mathsf{S}^{(t)}$이전의 long-term memory state은 $\mathcal{M}_{t-1}$

$h_t = \mathcal{M}^*_{t-1}(\mathbf{q}_t)$

- $\mathsf{S}^{(t)}$에 대한 쿼리 $\mathbf{q}_t$의 retrieve 수행

$\tilde{\mathsf{S}}^{(t)} = [p_1 \ \ p_2 \ \ ... \ \ p_{N_p}] \ || \ h_t \ || \ \mathsf{S}^{(t)}$

$y_t = \mathsf{Attn}(\tilde{\mathsf{S}}^{(t)})$

- persistency parameter : input-independent, fixed Task-specific information
    - 예를 들어 문장 분류, 변역, 물체 인식 등 여러가지 하위 task에 맞는 고정된 knowledge를 저장
- $h_t$ : 과거의 long-term memory에서 retrieve한 정보
- $\mathsf{S}^{(t)}$ : 현재 input segment
- 세 정보들을 concatenate하여 attention을 통해 어떤 정보를 저장할지를 결정
    - 즉, 더 중요한 정보에 집중할 수 있도록 하여 그 정보를 memory module에 저장할 수 있도록 함

<final output>

$\mathcal{M}_t = \mathcal{M}_{t-1}(y_t),$

$o_t = y_t \otimes\mathcal{M}^*_{t}(y_t)$

1. 현재 저장해야할 정보 $y_t$를 통해 memory module 업데이트
2. retrieve mode로 전환 후 현재 memory module에 $y_t$를 대입한 값과 $y_t$를 hadamart production(같은 차원의 두 벡터의 성분별 곱)을 통해 최종 output 도출

(개인적인 생각으로는 마지막 $o_t$의 과정이 self-attention과 유사하다고 생각함 다만 과거의 메모리 모듈에서 현재 저장해야할 정보와 자기 자신의 과거 메모리 정보값을 처리한다는 과정이 자기 자신 전체 context를 모두 계산하는 기존 과정보다 효율적이라고 생각함)

- 계속 생각해두면 좋은것은 memory module $\mathcal{M}$은 forward 과정만으로 메모리 업데이트가 진행된다는 것임(대박)
- 이러한 구조는 두가지의 주요한 장점이 있는데
    
    (1). historical, current context 두가지를 attention에 사용함으로써 현재 데이터가 주어졌을때, long-term memory 정보가 필요한지에 대해 결정할 수 있음
    
    (2). 이러한 구조가 long-term memory module이 중요한 정보만 현재 데이터에서 뽑아서 저장할 수 있는 이점을 가질 수 있음
    
    → 기존처럼 모든 토큰에 대해서 기억하는 것이 아니어서 효율적인 메모리 관리가 가능해짐
    
- test time 에서는…
    
    (1). persistence memory parameter는 task에 대한 지식을 가진채로 고정된 값을 가지고 있으며, 이는 변하지 않음
    
    (2). attention module이 in-context learner로 작동됨
    
    (3). 즉, long-term memory가 test time에서도 계속 학습을 함
    

<img width="415" alt="스크린샷_2025-02-08_오후_3 10 27" src="https://github.com/user-attachments/assets/2e3cf99a-4bf5-4083-95cd-96e325841875" />


- 위 사진은 각 sequence별로 참조하는 정보(토큰) 수인데
- long-term memory, persistent memory가 각각 추가될수록 sequence별 참조하는 정보의 양이 증가하는 것을 확인할 수 있음

## 4. 2. Gated Memory(MAG)

<img width="575" alt="스크린샷_2025-02-08_오후_3 26 00" src="https://github.com/user-attachments/assets/534ecb7f-96c5-4a55-96f7-eeacfba52901" />


- sliding window attention 사용 (CNN과 유사하게 window를 움직이며 단기적인 토큰에 집중함)

$\tilde{x} = [p_1 \ \ p_2 \ \ ... \ \ p_{N_p}] \ || \ x,$

$y = \mathsf{SW-Attn}^*(\tilde{x}),$

$o = y \otimes\mathcal{M}(\tilde{x})$

- $\mathsf{SW-Attn}^*$ : SWA + persistence memory
- MAC와 달리 input을 segment하지 않음
- $\otimes$ : non-linear gating
- 실험에서는 output을 learnable vector-calued weight를 통해 normalize한 후, Non-linearity $\sigma(.)$을 적용

<img width="401" alt="스크린샷_2025-02-08_오후_3 42 40" src="https://github.com/user-attachments/assets/9931fd47-8c35-4b55-9d5f-c08ec370b5b2" />


- 그림에서도 알 수 있듯이, sequence마다 long-term memory, persistent memory를 참조하는 MAC과 달리 처음부터 바로 두 메모리 모듈을 참조함
- SWA가 바로 shor-term memory 역할을 수행할 수 있음
- 특히 long-term memory는 시간이 지날 수록 중요하지 않는 정보는 까먹는 fading memory임
- 이러한 구조는 각 모듈 별로 서로 다른 메모리를 참조한다는 점에서 multi-head attention과 유사함 (왜?)

## 4. 3. Memory as a Layer(MAL)

<img width="556" alt="스크린샷_2025-02-08_오후_8 03 03" src="https://github.com/user-attachments/assets/a4e6946e-c58a-4080-b1ee-c010e1f608a1" />


- recurrent net, full or sliding window attention의 stack
    - 여기서 recurrent net은 논문의 long-term memory(LMM)을 사용
    - 즉 LMM + attention

$\tilde{x} = [p_1 \ \ p_2 \ \ ...p_{N_p}] \ || \ x,$

$y = \mathcal{M}(\tilde{x}),$

$o = \mathsf{SW-Attn}(y)$

- 앞서 논문에서 제시했던 모듈들과 달리 MAL은 각 layer에서 neural memory와 attention의 상호보완적인 장점을 가져가지는 못함(LMM과 attention의 상호보완적인 장점 비교 실험을 위한 모듈 설계인듯)

### Memory Without Attention

- Section 1에서도 언급했듯이 human brain에 가깝게 모방하기 위해서는 각 메모리 모듈들은 서로 상호보완적이면서도 **독립적이게** 작동해야함
    - 즉, attention이 없는 MAL(LMM)도 좋은 성능을 보여야함
    - 일종의 LMM을 하나의 독립적인 sequence 모델로 보는 것
- 그래서 논문에서는 결론적으로 LMM을 Titans라고 명명함

# 4. 4. Architectural Details

- 모든 block에서 residual connection 사용
- query, key, value 계산 위해 SiLU activation function 사용
- query, key normalize 위해 l2-norm 사용

### Convolution

- 1D depthwise-seperable conv layer 사용 (query, key, value projection 이후에 사용
    - 성능에 큰 영향을 미치지 않을 정도로 사용(오히려 성능 향상)
    - 계산 성능애도 효율적임

### Gating

- final output projection 이전에 normalization과 gating 사용

# 5. Experiments

- Titans의 performance를 보기 위해 5가지 실험 진행
    - language modeling, commonsense reasoning, needle in haystack, DNA modeling, time series forecasting tasks
- 실험들을 통해서 다음과 같은 질문들의 해답을 찾고자 함
    1. 다른 baseline models와 비교했을 때 downstream tasks에서 Titans는 성능이 어떠한가
    2. Titans에서 실제 context length가 얼마인가
    3. Titans는 context 길이에 관련하여 어떻게 확장이 가능한가
    4. memory depth가 성능과 효율성에 어떻게 영향을 미칠 수 있는가
    5. Titans의 각 요소와 성능은 어떠한 기여도를 가지는가

## 5. 1. Experimental Setup

### Models

- LMM, MAC, MAG, MAL 실험에 사용

### Baselines

- 여러가지 linear recurrent model들과 비교를 함
- 특히 눈여겨 볼 것은 transformer와 mamba도 비교대상 중 하나임

### Training

- Llama2 tokenizer -  vocabulary size : 32K / training length of 4K tokens
- AdamW optimizer 사용
- learning rate : 4e-4 + cosine annealing schedule
- batch size : 0.5M tokens
- weight decay : 0.1

## 5. 2. Language Modeling

<img width="591" alt="스크린샷_2025-02-10_오후_5 32 00" src="https://github.com/user-attachments/assets/e4f59cca-0637-434a-8dfc-5dda854af0ed" />


- SOTA 달성
- LMM vs TTT
    - weigh decay와 momentum의 중요성을 강조
    - TTT : gradeint기반의 recurrent model
- LMM vs other gating mechanism
    - suprise metric의 deep non-linear memory 구조가 이점을 가짐을 강조
- hybrid models of Titans(MAC, MAG, MAL) vs Samba, Gated DeltaNet-H2
    - attention과 architecture의 설계가 동일하기 때문에 memory module이 결정적인 성능 차이를 만듦
- MAG vs MAC
    - 성능이 서로 비슷하지만 데이터 내에서 좀 더 긴 dependency에 대해 좋은 성능을 보임
- MAG, MAC vs MAL
    - MAG, MAC가 성능이 더 좋음
    - 대부분의 기존 hybrid model이 MAL의 구조와 동일하기 때문에 이러한 실험결과는 중요함

## 5. 3. Needle in a Haystack

![a1166096-d995-431f-bfc4-deabce09ddd5](https://github.com/user-attachments/assets/ac0a583a-0d25-43c7-be25-b297182a134c)



- forgetting mechanism, momentum 덕분에 memory 용량을 더 잘 관리할 수 있음
- forgetting mechanism이 mamba와 달리 non-linearity를 가지기 때문에 더 우수한 성능을 보임

## 5. 4. BABILong Benchmark

<img width="475" alt="스크린샷_2025-02-11_오후_9 38 57" src="https://github.com/user-attachments/assets/cde48187-d01d-4a58-8416-6fa67701e63e" />


- 매우 긴 문서에서 분산된 여러 사실을 기반으로 **추론**
- 두 가지 실험과정을 세팅해서 실험 진행
    - Few-shot
        - 훨씬 적은 parameter로 모든 baseline 모델들 능가
    - Fine-tuning
        - RAG, 초대형 모델 등에서도 성능 능가

## 5. 5. The Effect of Deep Memory

<img width="653" alt="스크린샷_2025-02-11_오후_10 37 37" src="https://github.com/user-attachments/assets/eaff092e-3c91-4dd9-a89c-10f4ae861e49" />

- Mamba와 비교했을때, perplexity 정도가 훨씬 낫다는 것을 보여줌
- memory depth($L_{\mathcal{M}}$)이 증가할 수록 모든 sequence length에서 perplexity가 개선되는 것을 보여줌

<img width="212" alt="스크린샷_2025-02-11_오후_10 38 45" src="https://github.com/user-attachments/assets/6c240a4f-3029-4faa-b948-551af4ff5236" />


- sequence length에 상관없이 초당 처리하는 토큰 수가 일정함
- memory depth가 증가할수록 처리량이 선형적으로 감소
    
    → efficiency와 effectiveness가 trade-off 관계
    

### 5. 6. Time Series Forcasting

<img width="595" alt="스크린샷_2025-02-11_오후_10 51 14" src="https://github.com/user-attachments/assets/08f391b4-392f-4267-8da3-381d07eecbbf" />


### 5. 7. DNA Modeling

<img width="507" alt="스크린샷_2025-02-11_오후_10 50 20" src="https://github.com/user-attachments/assets/46e26b41-06bf-4075-8f8d-320b6ff31ad6" />


### 5. 8. Efficiency

<img width="223" alt="스크린샷_2025-02-11_오후_10 51 38" src="https://github.com/user-attachments/assets/f06565b6-e5b4-444f-892f-562142650a24" />


- 효율성 측면에서는 다른 모델들에 비해 다소 뒤떨어짐
- 저자들은 여러가지 이유(라고 하고 핑계라고 읽는다…)들을 서술함
    1. deep memory의 memory update으로 인한 복잡한 표현의 transition 때문
    2. mamba2 구현에서 고도로 최적화된 kernel을 사용했음 (구차한 변명을..)

### 5. 9. Ablation study

<img width="407" alt="스크린샷_2025-02-13_오전_9 14 30" src="https://github.com/user-attachments/assets/ada1410a-84eb-4205-9c02-503d7e2847cb" />

