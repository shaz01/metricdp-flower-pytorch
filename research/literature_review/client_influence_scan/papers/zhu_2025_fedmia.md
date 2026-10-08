FedMIA: An Effective Membership Inference Attack Exploiting
“All for One” Principle in Federated Learning
Gongxi Zhu1 Donghao Li2 Hanlin Gu2,3 * Yuan Yao2 Lixin Fan3 Yuxing Han1
1Tsinghua University 2The Hong Kong University of Science and Technology 3Webank
gx.zhu@foxmail.com, dlibf@connect.ust.hk, ghltsl123@gmail.com
Abstract
Federated Learning (FL) is a promising approach for train-
ing machine learning models on decentralized data while
preserving privacy. However, privacy risks, particularly
Membership Inference Attacks (MIAs), which aim to de-
termine whether a specific data point belongs to a target
client’s training set, remain a significant concern. Exist-
ing methods for implementing MIAs in FL primarily an-
alyze updates from the target client, focusing on metrics
such as loss, gradient norm, and gradient difference. How-
ever, these methods fail to leverage updates from non-target
clients, potentially underutilizing available information. In
this paper, we first formulate a one-tailed likelihood-ratio
hypothesis test based on the likelihood of updates from non-
target clients. Building upon this formulation, we introduce
a three-step Membership Inference Attack (MIA) method,
called FedMIA, which follows the ”all for one”—leveraging
updates from all clients across multiple communication
rounds to enhance MIA effectiveness. Both theoretical anal-
ysis and extensive experimental results demonstrate that
FedMIA outperforms existing MIAs in both classification
and generative tasks. Additionally, it can be integrated
as an extension to existing methods and is robust against
various defense strategies, Non-IID data, and different fed-
erated structures. Our code is available in https://
github.com/Liar-Mask/FedMIA.
1. Introduction
Federated learning (FL) [19, 26, 27] has emerged as a
promising approach for training machine learning models
on decentralized data sources while ensuring data privacy.
Despite its advantages, the privacy risks associated with the
information exchanged during FL have attracted significant
research attention. Membership Inference Attacks (MIAs)
in FL aim to determine whether a specific data point was
part of a particular client’s training dataset, typically per-
*Corresponding author.
1
Other MIAs
FedMIA (Ours)
𝜇𝜇𝑚𝑚𝑚𝑚𝑚𝑚 − 𝜇𝜇𝑛𝑛𝑛𝑛𝑛𝑛 = 0.027
 𝜇𝜇𝑚𝑚𝑚𝑚𝑚𝑚 − 𝜇𝜇𝑛𝑛𝑛𝑛𝑛𝑛 = 0.035
𝜇𝜇𝑚𝑚𝑚𝑚𝑚𝑚 − 𝜇𝜇𝑛𝑛𝑛𝑛𝑛𝑛 = 0.233
 𝜇𝜇𝑚𝑚𝑚𝑚𝑚𝑚 − 𝜇𝜇𝑛𝑛𝑛𝑛𝑛𝑛 = 0.167
Figure 1: The distributions of member and non-member
samples of FedMIA (the second row: FedMIA-I (ours) and
FedMIA-II (ours)) and other MIAs (the first row: Grad-
Cosine [24], Loss-Series [10]) on ResNet-CIFAR100. It
shows the obvious gap between the mean of the member
and non-member (µmem−µnon) for the proposed FedMIA
compared to other methods.
formed by adversaries positioned on the server side. In con-
trast to Gradient Inversion Attacks (GIAs) [7, 50], MIAs
[28] do not rely on strong assumptions, such as small batch
sizes or local training epochs, and thus remain significantly
underexplored within the FL context.
Most existing MIAs in FL [24, 28, 29, 39, 45, 48] focus
on inferring membership solely from the updates of the tar-
get clients, utilizing gradient norms, loss values, and gra-
dient differences. However, these methods overlook the
valuable information contained in updates uploaded by non-
target clients. Recent work [10, 14] attempts to enhance
the effectiveness of MIAs by incorporating shadow mod-
els into FL. While these methods make use of additional
information, they require an auxiliary dataset to train the
arXiv:2402.06289v3  [cs.LG]  27 Mar 2025

shadow model, which may not be feasible in the FL setting,
as the server does not have access to the private data of lo-
cal clients. Furthermore, training a shadow model incurs
additional computational costs for the adversary.
To address the limitations of existing approaches, this
paper proposes an alternative method that leverages updates
from non-target clients, denoted as Inon-tar, instead of rely-
ing on an auxiliary dataset for Membership Inference At-
tacks (MIAs). A key challenge in this approach is that
the server lacks knowledge of which updates correspond to
data trained on the target client’s dataset, making it diffi-
cult to estimate the distribution of updates trained on the
target data, denoted asQin. To overcome this challenge, we
demonstrate that it is possible to estimate the distribution of
updates that are not trained on the target data, denoted as
Qout, using updates uploaded by non-target clients Inon-tar.
Since each client’s data is disjoint, at least some of the up-
dates from non-target clients Inon-tar will not be trained on
the target data, i.e., ∃Inon-tar ∼ Qout. Based on this, we
formulate a one-tailed likelihood-ratio hypothesis test using
the distribution of updates that were not trained on the target
data,Qout, to perform the MIA.
Building on this one-tailed likelihood-ratio hypothesis
test, we introduce a three-step Membership Inference At-
tack (MIA) method called FedMIA, which follows the “all
for one”—leveraging updates from all clients across multi-
ple communication rounds. The first step involves comput-
ing a low-dimensional representation to simplify the dis-
tribution of updates. In the second step, we estimate the
distribution of updates not trained on the target data for
each communication round. Finally, we apply the one-tailed
likelihood-ratio test based on the estimated distribution of
updates not trained on the target data. This test is further
extended by incorporating updates from all communication
rounds. The proposed FedMIA has three advantages: 1)
FedMIA achieves superior performance compared to other
MIAs by utilizing the updates information from non-target
clients in Sect. 4; 2) FedMIA can be integrated into exist-
ing methods as an extension in Sect. 4.1; and 3) FedMIA
is robust against six defense methods, two federated struc-
tures, varying degrees of Non-IID data, and different client
counts, communication rounds, and local epochs. Our con-
tributions are summarized as the following:
• We first formulate the non-target updates as a one-tailed
likelihood-ratio hypothesis test to evaluate the perfor-
mance of the updates without being trained on target data.
Theorem 1 proves the validity of our formulation.
• Building on this hypothesis test, we introduce a three-step
Membership Inference Attack (MIA) method, called Fed-
MIA, which leverages updates from all clients and com-
munication rounds to enhance MIA effectiveness.
• Extensive results show that the proposed FedMIA: 1)
achieves superior performance compared to other MIAs
in both classification and generative tasks (see Fig. 1); 2)
can be integrated into existing methods as a plug-in; and
3) is robust against six defense methods, two federated
structures, varying degrees of Non-IID data, and different
client counts, communication rounds, and local epochs.
2. Related work
2.1. Federated Learning
Federated learning was originally proposed as a collabora-
tive approach for training machine learning models with-
out the need to share private data among multiple parties
[19, 26, 27, 43]. However, more recently, the concept of
“trustworthy federated learning” has been introduced by
[18]. This variant of federated learning places a heightened
emphasis on the preservation of privacy throughout the fed-
erated learning process. This shift in focus reflects the in-
creasing awareness of privacy concerns and the recognition
of the importance of robust security measures in federated
learning systems
2.2. Membership Inference Attack
MIA is a widely studied privacy attack in centralized learn-
ing scenarios. Depending on the information available to
the attacker, MIA can be categorized into black-box at-
tack (where only the output of the model can be obtained)
[4, 17, 33, 34, 36, 38, 41, 44] and white-box attack (where
the entire model is available) [29, 31].
In the context of federated learning, Nasr et al. [29] first
analyzed membership inference attacks in federated learn-
ing and proposed both passive and active attacks. In a pas-
sive attack, the attacker solely focuses on obtaining mem-
bership leaks based on accessible information without dis-
rupting or compromising the normal training process. Con-
versely, an active attack involves the ability to modify the
updates of federated learning, thereby increasing the vul-
nerability of the trained models to attacks. Zari et al. [45]
proposed a membership inference attack for federated learn-
ing that utilizes the probabilities of correct labels under lo-
cal models at different epochs for inference. However, this
approach requires member samples for auxiliary attacks. Li
et al. [24] proposed a passive membership inference attack
that does not require training on member samples. They
designed two metric features based on the orthogonality
of gradients to distinguish whether a sample is a member.
Hu et al.[16] designed an inference attack to facilitate an
honest-but-curious server to identify the training record’s
source client, which bases on but extends MIAs to source
inference. Moreover, inspired by work on worst-case pri-
vacy auditing, Aerni et al. [2] introduced an efficient as-
sessment method that accurately reflects the privacy of de-
fenders at their most vulnerable data points.

3. An Effective MIA in FL
In this section, we first present the setting of federated learn-
ing (FL) in Sect. 3.1. Subsequently, we formulate the Mem-
bership Inference Attacks (MIAs) in FL as a one-tailed like-
lihood ratio test in Sect. 3.2. Building upon this formula-
tion, we introduce an effective MIA in Sect. 3.3.
3.1. Setting
Horizontal Federated Learning. We consider ahorizontal
federated learning (HFL) [26, 43] setting consisting of one
server andK clients. We assumeK clients have their local
datasetDk ={(xk,i,y k,i)}nk
i=1,k = 1··· K, where xk,i is
the input data, yk,i is the label, and nk is the total number
of data points for kth client. Since we focus on evaluating
membership on each client, we further assume Dk are dis-
joint. We consider two commonly-used FL frameworks: 1)
FedAvg [27] that the server aggregates the models uploaded
by all clients; 2) FedEmbedding [25] that the server aggre-
gates the embeddings uploaded by all clients;
Threat Model. We assume the server are semi-honest and
do not collude with each other. The server faithfully exe-
cutes the training protocol but aims to infer the membership
information from the specific local clients.
Specifically, the server implement the attack A deter-
mine whether a specific sample (x,y ) belongs to the target
client’s datasetDtar based on a series of updates (models,
gradients or embeddings) among K clients andT commu-
nication rounds:I ={I t
k|t∈ [T ],k ∈ [K]}. However, the
server will not actively manipulate these information from
the local clients and thus without affecting the utilities of
the FL model. TheA can be represented as the following:
A(x,y,I) =
(
1, if (x,y )∈Dtar
0. otherwise (1)
3.2. One-tailed Likelihood-Ratio Test in FL
In FL, a membership inference attack is to determine the
sample (x,y ) belonging to target datasetDtar as follows:
H0 : (x,y ) /∈Dtar, H 1 : (x,y )∈Dtar. (2)
Moreover, for each communication round, the server ob-
servesK updates{I t
k}K
k=1. According to the updates of the
target clientI t
tar, it is thus natural to see a membership in-
ference attack as performing to guess whether updates I t
tar
is trained on the data (x,y ) or not. The Likelihood-ratio
Test [3] can be represented as:
Λ(Itar;x,y ) = pin(I =Itar|x,y )
pout(I =Itar|x,y ), (3)
where Qin(I|x,y ) and Qout(I|x,y ) denote the distribution
of updates trained on datasets with and without (x,y ), re-
spectively, andpin andpout represents the probability den-
sity function of Qin(I|x,y ) and Qout(I|x,y ).
However, estimating Qin(x,y ) in federated learning
(FL) presents a challenge, as the attacker (e.g., the server)
lacks knowledge of which updates are trained on (x,y ).
Therefore, we build the one-tailed Likelihood-Ratio Test us-
ing distribution Qout(I|x,y ) as:
ˆΛ(Itar,x,y ) =
X
I′<Itar
pout(I =I′|x,y ), (4)
which is the probability of observing a confidence as high as
the target updates under the null-hypothesis that the target
point (x,y ) is a non-member.
Remark 1. It is noted that, given the data sample (x,y ),
and considering that clients’ training datasets are disjoint,
at leastK− 2 updates (except the target updates) are guar-
anteed not to be trained on (x,y ). This enables the estima-
tion of Qout(x,y ).
Remark 2. Eq. (4) assumes that the member corresponds
to a large value ofI. If the member corresponds to a smaller
value of I, Eq. (4) is the probability of observing a confi-
dence as high as the target updates.
Furthermore, when the server observes theI ={I t
k|t∈
[T ],k ∈ [K]} duringT communication rounds, we can ex-
tend Eq. (4) as the following by utilizing the temporal in-
formation:
˜Λ({I t
tar}T
t=1,x,y ) = 1
T
TX
t=1
ˆΛ(I t
tar,x,y )
= 1
T
TX
t=1
X
I′<It
tar
pout(I =I′|x,y ),
(5)
where Qt
out(I|x,y ) are the distribution of updates trained
on datasets without (x,y ) in thetth communication round.
We establish the validity of Eq. (5) in Theorem 1, which
demonstrates that for all T communication rounds, any
member inferred by ˜Λ is also inferred at least once by
ˆΛt,t∈ [T ]. Furthermore, the worst-case membership leak-
age occurs when the target sample is inferred as a member
in at least one communication round (see proof in Appendix
C).
Theorem 1. Given the threshold δ, let Vt be the member
sets estimated by ˆΛt andδ in communication round t. Let
˜V be the member sets estimated by ˜Λ andδ. Then we have
˜V⊂ (V1∪···∪ VT ). (6)
3.3. The Proposed Method: FedMIA
Based on the one-tailed likelihood-ration test illustrated in
Sect. 3.2, we propose a three-step Membership inference
attacks by leveraging the spatial and temporal information.

Client 1
Client 2
Client k
Client tar
𝐼𝐼
1
𝑡𝑡
𝐼𝐼
2
𝑡𝑡
𝐼𝐼
k
𝑡𝑡
𝐼𝐼
tar
𝑡𝑡
Server
Attacker
Step1
Step2
�𝚲𝚲(𝑰𝑰𝒕𝒕𝒕𝒕𝒕𝒕
𝒕𝒕 , 𝒙𝒙, 𝒚𝒚)
ℚ𝑜𝑜𝑜𝑜𝑡𝑡(I|x, y)
Client 1
Client 2
Client k
Client tar
Server
Attacker 
Client 1
Client 2
Client k
Client tar
Server
Attacker
�𝚲𝚲(𝑰𝑰𝒕𝒕𝒕𝒕𝒕𝒕
𝟏𝟏 , 𝒙𝒙, 𝒚𝒚)Non-member
Confidences: �𝚲𝚲(𝑰𝑰𝒕𝒕𝒕𝒕𝒕𝒕
𝑻𝑻 , 𝒙𝒙, 𝒚𝒚)
Step2. Estimate Distribution
Step1. Low-dimensionalize
𝑀𝑀 𝐼𝐼 𝑥𝑥, 𝑦𝑦
𝐼𝐼
𝑘𝑘
𝑡𝑡
Cos-
simUpdates
 Measurement
∇𝑙𝑙𝑤𝑤(𝑥𝑥, 𝑦𝑦)
𝑀𝑀 of non-tar clients
Filter ℚ𝑜𝑜𝑜𝑜𝑡𝑡(I|x, y)
Step3. Infer Membership
② �𝚲𝚲 = Avg �𝚲𝚲𝟏𝟏, … , �𝚲𝚲𝑻𝑻 >  𝜹𝜹 ?
①Get Confidence �𝚲𝚲𝒕𝒕
ℚ𝑜𝑜𝑜𝑜𝑡𝑡(I|x, y)
Time 1 Time t Time T
Step3.① Step3.① Step3.①
Step3.② Non-Mem
Mem
or
Timeline of Model Training
 
Figure 2: Overview of FedMIA including three steps: 1) Computing the low-dimensional measurement; 2) Estimating the
distribution of updates without being trained on target data; 3) Building the one-tailed LRT test and Inferring the membership.
Step 1: Computing the low-dimensional measure-
mentM(I|(x,y )).
Since the updates are high-dimensional, directly estimat-
ing Qout(I|x,y ) is challenging. To address this, we utilize
gradient similarity [24] to map the updatesI(x,y ) to a low-
dimensional variableM(I|(x,y )), which allows for an es-
timation of the distribution as follows:
M(I|(x,y )) = ⟨I, ∂ℓ(ω,x,y)
∂ω ⟩
∥I∥∥ ∂ℓ(ω,x,y)
∂ω ∥
, (7)
Eq. (7) evaluates the similarity between the uploaded gra-
dient I (∇F ) and the model gradient on the target data
∂ℓ(ω,x,y)
∂ω , with similarity increasing when the target data is
a member. The modelω is the global model.
Remark 3. Eq. (7) represents one approach; other mea-
surements, such as loss or gradient norm [10, 28], can also
be used. This suggests that our method can be integrated
with existing methods that employ different measurements.
In Sect. 4, we also consider the model loss ℓ(ω,x,y ) on
target data [10] as one measurement.
Step 2: Estimating the Qout(M(I|x,y )) for each
communication roundt.
We first leverage the non-target clients’ updates{I t
k|k∈
[K],k ̸= tar} to estimate the distribution of M(I|(x,y )
without trained on (x,y ), i.e., ˜Qout(M(I|(x,y ))). We
assume Qout(M(I|x,y )) is a Gaussian distribution, i.e.,
Qout(M(I|x,y ))∼N (µout,σ 2
out).
If (x,y ) is trained on I t
k, then M(I t
k|x,y ) becomes
large. Therefore, if M(Ik|(x,y )) is exceptionally high for
all non-target clients’ measurements, the updates from the
non-target client k are likely trained on (x,y ) with high
probability. Consequently, we remove the extreme large
values of M(I t
k|x,y ), where k ̸= tar, to better estimate
˜Qout(M(I|(x,y ))). To filter the updates sets, we apply the
3-σ rule to the updates trained on (x,y ). Specifically, we
remove thek-th update if the following condition holds:
M(I t
k|x,y )>µ t + 3σt, (8)
where
(
µt = 1
K−1
P
j̸=tarM(I t
j|x,y )
σt =
q
1
K−1
P
j̸=tar(M(I t
j|x,y )−µt)2. (9)
After filtering, we obtain the update set Ut which ex-
cludes updates trained on the target data (x,y ) for commu-
nication round t with high probability. Therefore, we esti-
mate the distribution mean and variance of N (µt
out,v t
out)
as:
(
µt
out = 1
|Ut|
P
j∈UtM(I t
j|x,y ), and
vt
out = 1
|Ut|
P
j∈Ut(M(I t
j|x,y )−µt
out)2. (10)
Step 3: Inferring the membership based on
˜Λ({I t
tar}T
t=1,x,y ).
According the µt
out, and vt
out estimated in step 2, we can
calculate the ˆΛt(Itar,x,y ) of Eq. (4) as:
ˆΛ(I t
tar,x,y ) =
Z M (It
tar|x,y)
−∞
1p
2πv t
out
e
−
(x−µt
out )2
2vt
out dx,
(11)
which is the probability of observing a confidence as low
as the target updates under the null-hypothesis that the tar-
get point (x,y ) is a non-member. Specifically, the small
ˆΛ(I t
tar,x,y ) indicates the target data is non-member.

Moreover, we utilize the updates of all communication
rounds to obtain the ˜Λ(Itar,x,y ) of Eq. (5) as:
˜Λ({I t
tar}T
t=1,x,y ) = 1
T
TX
t=1
ˆΛ(I t
tar,x,y )
= 1
T
TX
t=1
Z M (It
tar|x,y)
−∞
1p
2πv t
out
e
−
(x−µt
out )2
2vt
out ,
(12)
Finally, given the thresholdδ, the member is determined if
˜Λ>δ , otherwise is non-member.
Algorithm 1 FedMIA
1: Input: Target data sample(x,y ) of clientk, communi-
cation rounds T , set of client models{I t
k|t∈ [T ],k ∈
[K]}, a thresholdδ.
2: Output: membership prediction (0 or 1)
3: ▷ Computing the low-dimensional measurement
4: Calculate M(I t
k|x,y ) according to Eq. (7) for all
k∈ [K],t∈ [T ];
5: ▷ Estimating the Qout(I|x,y )
6: ChooseUt according to Eq. (8);
7: µt
out = 1
|Ut|
P
j∈UtM(I t
j|x,y );
8: vt
out = 1
|Ut|
P
j∈Ut(M(I t
j|x,y )−µt
out)2;
9: ▷ Inferring the membership
10: Calculate ˆΛ(I t
tar,x,y ) according to Eq. (11);
11: Calculate ˜Λ({I t
tar}T
t=1,x,y ) according to Eq. (12);
12: if ˜Λ({I t
tar}T
t=1,x,y )>δ then
13: return 1
14: else
15: return 0
16: end if
4. Experimental Result
This section presents the empirical analysis of the proposed
FedMIA framework in terms of experimental setting, attack
effectiveness, robustness.
4.1. Experimental Setup
Dataset & Models. We employed three image datasets:
CIFAR-100 [20], DermNet [1], and Tiny-ImageNet [22].
Additionally, we implemented three models: ResNet18 [13]
and AlexNet [21] for the classification task, and the Latent
Diffusion Model [32] for the generative task.
Federated Setting. We consider horizontal federated learn-
ing with two typical structures: FedAvg [27], which trans-
fers models or gradients in classification tasks, and FedEm-
bedding [25], which transfers prompts in generative tasks.
We evaluate scenarios with 5–30 clients in FL, up to 300
communication rounds, and the data samples of each client
range from 1,000 to 10,000. Local training involves 1 to
9 epochs, and we explore different Non-IID extents us-
ing the Dirichlet distribution dir(β), with values of β =
0.1, 1, 10,∞ (IID). If there are no additional instructions,
each experiment has 10 clients, 300 synchronous commu-
nication rounds.
MIAs. We conducted a comprehensive comparison of our
methods, FedMIA-I and FedMIA-II, against six baseline
attack methods: Blackbox-Loss [44], Grad-Cosine [24],
Grad-Norm [28], Loss-Series [10], Avg-Cosine [24], and
Grad-Diff [24]. FedMIA-I is utilizes the model loss mea-
surement [44], while FedMIA-II employs the Grad-Cosine
measurement [24], as described in Eq. (7).
Defenses. We evaluate the robustness of FedMIA against
six defense methods, including Gradient Perturbation (Per-
turb) [8, 49], Gradient Sparsification (Sparse) [11], MixUp
[9, 47], Data Augmentation [37], Data Sampling [23], and
a combination of Data Augmentation + Sampling.
Evaluation metric. We use the metrics AUC and
TPR@FPR [3] to specifically assess the leakage of the most
vulnerable samples to attacks, where TPR@FPR refers to
the True Positive Rate (TPR) at a specific False Positive
Rate (FPR, a.k.a. Type-I Error Rate) in binary classifica-
tion. Specifically, we pay particular attention to the TPR
when the FPR is very low, such as FPR values of 0.1%.
Moreover, we test the attack effectiveness against different
defense methods. Since the attack effectiveness depends
on the parameters of defenses, e.g., for gradient perturba-
tion, if more noise is added, the attack effectiveness be-
comes weaker, but the model utility is largely influenced.
Therefore, we consider the attack effectiveness under dif-
ferent parameters of each defense method and obtain the
tradeoff between utility loss (the test error rate) and attack
effectiveness (TPR@FPR = 0.1%). Furthermore, the effec-
tiveness of the attack can be measured using hypervolume
(HV) [51], which, in our case, refers to the area between
the Pareto frontiers and the unit box. A larger hypervolume
indicates a better privacy-utility trade-off, representing that
the attack is less effective.
More experimental setup details are left in Appendix A.
4.2. FedMIA vs Others
The comparison results of all attacks are presented in the
Tab. 1. We can draw the following three conclusions:
• FedMIA generally outperforms other MIA meth-
ods across all experiments, as indicated by higher
TPR@FPR=0.1% and AUC metrics. For example,
FedMIA-II achieves a TPR@FPR=0.1% of (66.98 ±
1.74)%, which is significantly higher than the next best
method with (54.66± 1.22)% on AlexNet-CIFAR100.
• Blackbox-Loss [44] and Grad-Cosine [24] show rela-
tively low TPR and AUC values, significantly lagging

Table 1: Comparison of our attack with various MIAs methods on classification tasks and generative tasks. The larger
TPR(%)@FPR=0.1% and AUC indicates the better attack effectiveness.
MIA Methods Blackbox-Loss
[44]
Grad-Cosine
[24]
Grad-Norm
[28]
Loss-Series
[10]
Avg-Cosine
[24]
Grad-Diff
[24]
FedMIA-I
(Ours)
FedMIA-II
(Ours)
AlexNet
CIFAR100
TPR 0.18±0.05 7.26 ±0.25 0.14 ±0.03 25.3 ±0.88 54.66 ±1.22 20.52 ±0.45 53.78±1.24 66.98±1.74
AUC 0.58±0.01 0.78 ±0.02 0.51 ±0.01 0.82 ±0.01 0.85 ±0.01 0.61 ±0.02 0.90±0.01 0.89±0.02
AlexNet
DermNet
TPR 0.27±0.12 9.53 ±0.54 0.13 ±0.03 22.8 ±1.13 41 ±0.48 5.6 ±0.05 48.23±0.87 62.27±0.23
AUC 0.68±0.01 0.74 ±0.01 0.50 ±0.01 0.94±0.01 0.85±0.01 0.89 ±0.01 0.91±0.02 0.87 ±0.02
ResNet18
CIFAR100
TPR 0.36±0.11 5.48 ±0.27 0.26 ±0.06 16.82 ±2.12 44.02 ±1.58 15.06 ±1.78 57.36±2.12 68.74±1.84
AUC 0.67±0.01 0.80 ±0.02 0.55 ±0.01 0.73 ±0.01 0.85 ±0.02 0.65 ±0.01 0.84±0.01 0.89±0.01
ResNet18
DermNet
TPR 0.27±0.11 0.73 ±0.21 0.06 ±0.01 32.2 ±0.89 19.93 ±1.23 17.93 ±2.12 35.6±0.96 31.8±0.88
AUC 0.51±0.01 0.59 ±0.01 0.48 ±0.01 0.52 ±0.01 0.63 ±0.02 0.66±0.01 0.64±0.01 0.62 ±0.01
Diffusion Model
Tiny-ImageNet
TPR 1.10±0.50 2.40 ±0.60 0.25 ±0.02 1.5 ±0.20 1.80 ±0.30 1.20 ±0.40 3.20±0.30 4.50±0.20
AUC 0.51±0.02 0.54 ±0.01 0.49 ±0.01 0.54 ±0.01 0.53 ±0.01 0.51 ±0.01 0.58±0.01 0.59±0.01
Diffusion Model
CIFAR100
TPR 0.80±0.10 1.20 ±0.20 0.11 ±0.01 1.30 ±0.10 1.80 ±0.10 1.7 ±0.20 2.1±0.20 3.0±0.20
AUC 0.48±0.01 0.61 ±0.01 0.49 ±0.01 0.47 ±0.01 0.59 ±0.01 0.52 ±0.01 0.59±0.01 0.62±0.01
behind FedMIA. For example, in the AlexNet DermNet
task, Blackbox-Loss and Grad-Cosine both result in low
TPRs (around 0.27%), while FedMIA achieves a much
higher TPR of (62.27± 0.23)%.
• FedMIA performs well in both classification tasks and
generative task. Specifically, FedMIA also leads, but the
performance gap between FedMIA and other methods
like Blackbox-Loss or Grad-Cosine is more pronounced.
For example, FedMIA achieves a TPR of (3.0 ± 0.20)%
and an AUC of (0.62 ± 0.01)%, while Blackbox-Loss
and Grad-Cosine have much lower TPR and AUC values
(around (1.7± 0.20)% and (0.52± 0.01)%, respectively)
in CIFAR100 with diffusion model.
1
(a) Original training images
1
(b) Generated images
Figure 3: Original training images and generated images
based on uploaded embeddings via latent diffusion model.
Furthermore, we present the generated images based
on the embeddings alongside the original images, which
are considered as members by the proposed FedMIA-II, as
shown in Fig. 3. The results indicate that the two types of
images are highly similar, demonstrating that the target data
identified by FedMIA as a member is effectively trained on
the corresponding embedding. This observation highlights
the effectiveness of the FedMIA attack.
4.3. Robustness
This section illustrates the robustness of FedMIA against
six defense methods, varying degrees of Non-IID, different
client counts, communication rounds, and local epochs.
FedMIA against different defense methods. Tab. 2 and
Appendix B presents the hypervolume and privacy-utility
tradeoff against different defense methods. We can obtain:
1)FedMIA-I (ours) consistently performs better than other
MIA methods across all defense strategies for AlexNet: Ex-
ample: For AlexNet-CIFAR100, under Mixup, FedMIA-I
achieves a hypervolume of 0.3609, outperforming methods
like Blackbox-Loss (0.3328) and Loss-Series (0.3554); 2)
Combining data augmentation and data sampling combin-
ing in an appropriate manner may have the best defense ef-
fectiveness (achieves the largest HV volume); 3) Even the
strongest defense with combining data augmentation and
data sampling still exist privacy leakage when preserving
the model performance (see Appendix B).
Non-IID extent. We investigate the impact of non-IID on
MIA attacks. Following [15], the basic assumption of non-
iid simulation in this part is that the labels of each client’s
training data follow the Dirichlet distribution. β is the core
parameter controlling the distribution difference and the
smaller the β, the greater the degree of non-iid. We report
the performance of the FedMIA-II attack on the CIFAR-100
dataset in a table. We control the degree of non-IID by ad-
justing the parameter alpha, where a smaller β indicates a

Table 2: The hypervolume of various attack methods under defense strategies with AlexNet and ResNet on CIFAR100. The
smaller Hypervolume indicates the better attack effectiveness.
Perturb
[49]
Sparse
[11]
Mixup
[47]
Sampling
[23]
Data Aug
[37]
Data Aug
+ Sampling
AlexNet
CIFAR100
Blackbox-Loss [44] 0.3543 0.3533 0.3328 0.3491 0.3395 0.3422
Loss-Series [10] 0.3252 0.3377 0.3554 0.3464 0.3375 0.3427
Grad-Cosine [24] 0.304 0.3287 0.2845 0.3365 0.3247 0.3379
Avg-Cosine [24] 0.3421 0.3421 0.3334 0.3407 0.3248 0.3376
FedMIA-I (ours) 0.3611 0.3593 0.3609 0.3373 0.3202 0.3359
FedMIA-II (ours) 0.2702 0.3085 0.2588 0.3325 0.3175 0.3361
ResNet
CIFAR100
Blackbox-Loss [44] 0.4338 0.4307 0.4538 0.4568 0.5833 0.5815
Loss-Series [10] 0.4126 0.4092 0.458 0.4555 0.5823 0.5821
Grad-Cosine [24] 0.3261 0.3545 0.3905 0.4407 0.5751 0.5779
Avg-Cosine [24] 0.4064 0.4059 0.4533 0.4537 0.571 0.5745
FedMIA-I (ours) 0.4438 0.4392 0.4736 0.4446 0.5665 0.574
FedMIA-II (ours) 0.2969 0.3004 0.4302 0.4425 0.5693 0.5739
0
20
40
60
80
100TPR@FPR=0.1%
0.18
7.26
25.3
54.66 53.78
66.98
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II
(a) IID
0
20
40
60
80
100TPR@FPR=0.1%
0.16
5.9
27.04
54.08 55.09
66.76
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II (b) β = 10
0
20
40
60
80
100TPR@FPR=0.1%
0.67
7.98
59.57
68.97
78.55
83.14
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II (c) β = 1
0
20
40
60
80
100TPR@FPR=0.1%
1.49
5.99
96.2 94.23
99.3 98.4
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II (d) β = 0.1
0
20
40
60
80
100TPR@FPR=0.1%
0.36
5.48
16.82
44.02
57.36
68.74
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II
(e) IID
0
20
40
60
80
100TPR@FPR=0.1%
0.31
5.9
21.69
60.23 61.86
70.68
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II (f) β = 10
0
20
40
60
80
100TPR@FPR=0.1%
0.57
6.35
43.65
71.14
74.99
81.04
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II (g) β = 1
0
20
40
60
80
100TPR@FPR=0.1%
2.89
6.92
95.4 94.3
99.1 98.91
Blackbox-Loss
Grad-Cosine
Loss-Series
Avg-Cosine
FedMIA-I
FedMIA-II (h) β = 0.1
Figure 4: This set of figures shows the attack effects (TPR@FPR=0.1%) of various attacks (Blackbox-Loss[44], Grad-
Cosine[24], Loss-Series[10], Avg-Cosine[24], FedMIA-I and FedMIA-II) on AlexNet and ResNet18 (the first and second
row respectively) under IID and three Non-IID settings.
more severe Non-IID condition.
Based on Fig.4, we can observe the following: 1) the
proposed method is the strongest attacks with various Non-
IID extent, e.g., the TPR@FPR =0.1% for FedMIA-II
achieves the largest value 67% in AlexNet-CIFAR100. 2)
As non-IID increases, TPR shows a increasing trend. One
reason is when non-IID becomes severe, such as when a
client contains only a few classes, MIA attacks themselves
become easier. An extreme example is when a client con-
tains only one class, in which case we can achieve a strong
baseline by simply judging based on the sample labels.
Communication round. As for the benefits of synchronous
rounds to our scheme, it can be observed in Figure 5(a) and
(e) that the attack effect of our scheme increases rapidly in
most epochs as the synchronous communication progresses.
At epoch=200, the TPR@FPR=0.1% of Ours exceeds 0.4,
which is twice as high as that of the Avg-Cosine attack
and Loss-series attack. After that, the attack effect shows
a slight decrease (about 5% on TPR@FPR=0.1%) and a
similar trend can also be observed in the curve of the Avg-
Cosine attack. This phenomenon may be attributed to the
fact that the information obtained in the later epochs is not

0 100 200 300
Communication Epoch
0
20
40
60TPR@FPR=0.1%
Loss-Series
Avg-Cosine
FedMIA-II
(a) AlexNet-CIFAR100
5 10 15 20 25 30
Client Number
30
40
50
60
70TPR@FPR=0.1%
 (b) AlexNet-CIFAR100
2000 4000 6000 8000
Sample Number
20
30
40
50TPR@FPR=0.1%
 (c) AlexNet-CIFAR100
2 4 6 8
Local Epoch
30
40
50
60
70
80TPR@FPR=0.1%
 (d) AlexNet-CIFAR100
0 100 200 300
Communication Epoch
0
20
40
60
80TPR@FPR=0.1%
(e) ResNet18-CIFAR100
5 10 15 20 25 30
Client Number
20
30
40
50
60
70TPR@FPR=0.1%
 (f) ResNet18-CIFAR100
2000 4000 6000 8000
Sample Number
10
20
30
40
50TPR@FPR=0.1%
 (g) ResNet18-CIFAR100
2 4 6 8
Local Epoch
20
40
60
80
100TPR@FPR=0.1%
Loss-Series
Avg-Cosine
FedMIA-II (h) ResNet18-CIFAR100
Figure 5: This set of figures shows the attack effects (TPR@FPR=0.1%) of various attacks (blue line: Loss-series [10], green
line: Avg-Cosine [24] and red line: FedMIA-II) on AlexNet and ResNet18 (the first and second row respectively) under
four settings. The four columns of the graph group show the results of different communication rounds, client numbers, data
volumes and local epochs settings respectively.
as helpful for the membership leakage attack as the infor-
mation acquired in the previous epochs.
Number of Clients. Figure 5(b) and (f) illustrate the effects
of membership inference attacks on AlexNet and ResNet18,
while varying the number of clients from 2 to 20. As de-
picted in the figures, our two attacks outperform the base-
lines in most cases, indicating their significantly higher ef-
fectiveness. Furthermore, the gradually rising red and blue
curves indicates that as the number of clients increases, the
target model becomes more vulnerable to our MIA scheme.
Number of Samples. Figure 5(c) and (g) demonstrate the
impact of varying the number of samples (ranging from
500 to 5000) on the attack effects of MIAs on AlexNet
and ResNet18. Regardless of the increase in the number
of samples, the attack effect of our scheme remains con-
sistently high and even demonstrates notable improvement
on AlexNet. The TPR@FPR=0.1% of ours consistently ex-
ceeds twice that of the baseline in both subfigures. This
indicates that our attack scheme maintains a significant ad-
vantage as the training data increases.
Local Epoch. Figure 5(d) and (h) illustrate the effects of
MIAs on AlexNet and ResNet18 as the number of local
epochs varies from 1 to 9. As the number of local epochs
increases, the effectiveness of our attack method and the fed
loss attack significantly improve while the enhancement ef-
fect of the Avg-Cosine attack is not evident. This demon-
strates that an increase in the number of local epochs may
render the model more susceptible to MIA.
5. Discussion and Conclusion
While advantages brought by FL can be ascribed to, by and
large, the principle of “one for all and all for one”, this pa-
per shows that information shared by all clients through a
semi-honest server can actually be adversely exploited to
launch very effective membership inference attacks. Specif-
ically, this paper introduce FedMIA, a novel Membership
Inference Attack (MIA) method by leveraging updates from
non-target clients and applies a one-tailed likelihood-ratio
hypothesis test. This enables the inference of target data
membership without requiring access to auxiliary datasets
or making strong assumptions about the training process.
Through extensive experiments, we demonstrated that Fed-
MIA is highly effective across various federated learning
configurations, including both classification and generative
tasks, and remains robust against common defense methods,
Non-IID data, and different client setups.
Conventional FL privacy defenses (perturbation, sparsi-
fication, mixup) prove ineffective against FedMIA due to
attackers’ exploitation of cross-client information patterns
as shown in Sect. 4.3. While secure aggregation via MPC
[5][6]/HE [42][46] blocks FedMIA by encrypting individ-
ual updates, their computational/communication costs hin-
der practical deployment. This exposes an urgent need for
MIA defenses specifically to resist attackers from obtaining
valuable information from non-target updates.

References
[1] Amina Aboulmira, Hamid Hrimech, and Mohamed
Lachgar. Comparative study of multiple cnn models
for classification of 23 skin diseases. International
Journal of Online & Biomedical Engineering, 18(11),
2022. 5, 12
[2] Michael Aerni, Jie Zhang, and Florian Tram `er. Eval-
uations of machine learning privacy defenses are mis-
leading. arXiv preprint arXiv:2404.17399, 2024. 2
[3] Nicholas Carlini, Steve Chien, Milad Nasr, Shuang
Song, Andreas Terzis, and Florian Tramer. Member-
ship inference attacks from first principles. In 2022
IEEE Symposium on Security and Privacy (SP), pages
1897–1914. IEEE, 2022. 3, 5
[4] Christopher A Choquette Choo, Florian Tramer,
Nicholas Carlini, and Nicolas Papernot. Label-
only membership inference attacks. arXiv preprint
arXiv:2007.14321, 2020. 2
[5] Hossein Fereidooni, Samuel Marchal, Markus Mietti-
nen, Azalia Mirhoseini, Helen M ¨ollering, Thien Duc
Nguyen, Phillip Rieger, Ahmad-Reza Sadeghi,
Thomas Schneider, Hossein Yalame, et al. Safelearn:
Secure aggregation for private federated learning. In
2021 IEEE Security and Privacy Workshops (SPW) ,
pages 56–62. IEEE, 2021. 8
[6] Till Gehlhar, Felix Marx, Thomas Schneider, Ajith
Suresh, Tobias Wehrle, and Hossein Yalame. Safefl:
Mpc-friendly framework for private and robust fed-
erated learning. In 2023 IEEE Security and Privacy
Workshops (SPW), pages 69–76. IEEE, 2023. 8
[7] Jonas Geiping, Hartmut Bauermeister, Hannah Dr ¨oge,
and Michael Moeller. Inverting gradients-how easy is
it to break privacy in federated learning? Advances
in Neural Information Processing Systems, 33:16937–
16947, 2020. 1
[8] Robin C Geyer, Tassilo Klein, and Moin Nabi. Dif-
ferentially private federated learning: A client level
perspective. arXiv preprint arXiv:1712.07557, 2017.
5, 12, 13
[9] Hanlin Gu, Jiahuan Luo, Yan Kang, Lixin Fan, and
Qiang Yang. Fedpass: Privacy-preserving vertical fed-
erated deep learning with adaptive obfuscation. arXiv
e-prints, pages arXiv–2301, 2023. 5, 12
[10] Yuhao Gu, Yuebin Bai, and Shubin Xu. Cs-mia: Mem-
bership inference attack based on prediction confi-
dence series in federated learning. Journal of Infor-
mation Security and Applications , 67:103201, 2022.
1, 4, 5, 6, 7, 8, 13
[11] Otkrist Gupta and Ramesh Raskar. Distributed learn-
ing of deep neural network over multiple agents.Jour-
nal of Network and Computer Applications , 116:1–8,
2018. 5, 7, 12
[12] Farzin Haddadpour, Mohammad Mahdi Kamani,
Aryan Mokhtari, and Mehrdad Mahdavi. Federated
learning with compression: Unified analysis and sharp
guarantees. In International Conference on Artificial
Intelligence and Statistics, pages 2350–2358. PMLR,
2021. 12
[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. Deep residual learning for image recognition.
In Proceedings of the IEEE conference on Computer
Vision and Pattern Recognition (CVPR) , pages 770–
778, 2016. 5
[14] Xinlong He, Yang Xu, Sicong Zhang, Weida Xu, and
Jiale Yan. Enhance membership inference attacks
in federated learning. Computers & Security , 136:
103535, 2024. 1
[15] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown.
Measuring the effects of non-identical data distribu-
tion for federated visual classification. arXiv preprint
arXiv:1909.06335, 2019. 6, 12
[16] Hongsheng Hu, Xuyun Zhang, Zoran Salcic, Lichao
Sun, Kim-Kwang Raymond Choo, and Gillian Dob-
bie. Source inference attacks: Beyond membership
inference attacks in federated learning. IEEE Trans-
actions on Dependable and Secure Computing, 2023.
2
[17] Bo Hui, Yuchen Yang, Haolin Yuan, Philippe Burlina,
Neil Zhenqiang Gong, and Yinzhi Cao. Practical blind
membership inference attack via differential compar-
isons. arXiv preprint arXiv:2101.01341, 2021. 2
[18] Yan Kang, Hanlin Gu, Xingxing Tang, Yuanqin He,
Yuzhu Zhang, Jinnan He, Yuxing Han, Lixin Fan,
and Qiang Yang. Optimizing privacy, utility and effi-
ciency in constrained multi-objective federated learn-
ing. arXiv preprint arXiv:2305.00312, 2023. 2
[19] Jakub Kone ˇcn`y, H Brendan McMahan, Daniel Ram-
age, and Peter Richt´arik. Federated optimization: Dis-
tributed machine learning for on-device intelligence.
arXiv preprint arXiv:1610.02527, 2016. 1, 2
[20] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
Cifar-10 (canadian institute for advanced research). 5,
12
[21] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hin-
ton. Imagenet classification with deep convolutional
neural networks. Advances in neural information pro-
cessing systems, 25, 2012. 5
[22] Ya Le and Xuan Yang. Tiny imagenet visual recogni-
tion challenge. CS 231N, 7(7):3, 2015. 5, 12
[23] Anran Li, Lan Zhang, Juntao Tan, Yaxuan Qin, Jun-
hao Wang, and Xiang-Yang Li. Sample-level data
selection for federated learning. In IEEE INFO-
COM 2021-IEEE Conference on Computer Commu-
nications, pages 1–10. IEEE, 2021. 5, 7, 12, 13

[24] Jiacheng Li, Ninghui Li, and Bruno Ribeiro. Effec-
tive passive membership inference attacks in federated
learning against overparameterized models. In The
Eleventh International Conference on Learning Rep-
resentations, 2022. 1, 2, 4, 5, 6, 7, 8, 13
[25] Jinglin Liang, Jin Zhong, Hanlin Gu, Zhongqi Lu,
Xingxing Tang, Gang Dai, Shuangping Huang, Lixin
Fan, and Qiang Yang. Diffusion-driven data replay: A
novel approach to combat forgetting in federated class
continual learning. In European Conference on Com-
puter Vision, pages 303–319. Springer, 2025. 3, 5
[26] Brendan McMahan, Eider Moore, Daniel Ram-
age, Seth Hampson, and Blaise Aguera y Arcas.
Communication-efficient learning of deep networks
from decentralized data. In Proceedings of Artificial
Intelligence and Statistics (AISTATS) , pages 1273–
1282, 2017. 1, 2, 3
[27] H Brendan McMahan, Eider Moore, Daniel Ramage,
and Blaise Ag ¨uera y Arcas. Federated learning of
deep networks using model averaging. arXiv preprint
arXiv:1602.05629, 2016. 1, 2, 3, 5
[28] Milad Nasr, Reza Shokri, and Amir Houmansadr. Ma-
chine learning with membership privacy using adver-
sarial regularization. In Proceedings of the 2018 ACM
SIGSAC Conference on Computer and Communica-
tions Security (CCS), 2018. 1, 4, 5, 6
[29] Milad Nasr, Reza Shokri, and Amir Houmansadr.
Comprehensive privacy analysis of deep learning:
Passive and active white-box inference attacks against
centralized and federated learning. In 2019 IEEE sym-
posium on security and privacy (SP) , pages 739–753.
IEEE, 2019. 1, 2
[30] Amirhossein Reisizadeh, Aryan Mokhtari, Hamed
Hassani, Ali Jadbabaie, and Ramtin Pedarsani. Fed-
paq: A communication-efficient federated learning
method with periodic averaging and quantization.
In International Conference on Artificial Intelligence
and Statistics, pages 2021–2031. PMLR, 2020. 12
[31] Shahbaz Rezaei and Xin Liu. Towards the infeasibil-
ity of membership inference on deep models. arXiv
preprint arXiv:2005.13702, 2020. 2
[32] Robin Rombach, Andreas Blattmann, Dominik
Lorenz, Patrick Esser, and Bj ¨orn Ommer. High-
resolution image synthesis with latent diffusion mod-
els. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition , pages
10684–10695, 2022. 5
[33] Alexandre Sablayrolles, Matthijs Douze, Yann Ol-
livier, Cordelia Schmid, and Herv´e J´egou. White-box
vs black-box: Bayes optimal strategies for member-
ship inference. In International Conference on Ma-
chine Learning (ICML). PMLR, 2019. 2
[34] Ahmed Salem, Yang Zhang, Mathias Humbert, Mario
Fritz, and Michael Backes. Ml-leaks: Model and
data independent membership inference attacks and
defenses on machine learning models. In Annual
Network and Distributed System Security Symposium
(NDSS), 2019. 2
[35] Reza Shokri and Vitaly Shmatikov. Privacy-
preserving deep learning. In Proceedings of the 22nd
ACM SIGSAC conference on computer and communi-
cations security, pages 1310–1321, 2015. 12, 13
[36] Reza Shokri, Marco Stronati, Congzheng Song, and
Vitaly Shmatikov. Membership inference attacks
against machine learning models. In 2017 IEEE sym-
posium on security and privacy (SP) , pages 3–18.
IEEE, 2017. 2
[37] Connor Shorten and Taghi M Khoshgoftaar. A survey
on image data augmentation for deep learning. Jour-
nal of big data, 6(1):1–48, 2019. 5, 7, 12
[38] Liwei Song and Prateek Mittal. Systematic evaluation
of privacy risks of machine learning models. arXiv
preprint arXiv:2003.10595, 2020. 2
[39] Anshuman Suri, Pallika Kanani, Virendra J Marathe,
and Daniel W Peterson. Subject membership infer-
ence attacks in federated learning. arXiv preprint
arXiv:2206.03317, 2022. 1
[40] Chandra Thapa, Pathum Chamikara Mahawaga
Arachchige, Seyit Camtepe, and Lichao Sun. Splitfed:
When federated learning meets split learning. In Pro-
ceedings of the AAAI Conference on Artificial Intelli-
gence, pages 8485–8493, 2022. 12
[41] Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei
Yu, and Wenqi Wei. Demystifying membership infer-
ence attacks in machine learning as a service. IEEE
Transactions on Services Computing, 2019. 2
[42] Febrianti Wibawa, Ferhat Ozgur Catak, Murat Kuzlu,
Salih Sarp, and Umit Cali. Homomorphic encryption
and federated learning based privacy-preserving cnn
training: Covid-19 detection use-case. In Proceedings
of the 2022 European Interdisciplinary Cybersecurity
Conference, pages 85–90, 2022. 8
[43] Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin
Tong. Federated machine learning: Concept and ap-
plications. ACM Transactions on Intelligent Systems
and Technology (TIST), 10(2):1–19, 2019. 2, 3
[44] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and
Somesh Jha. Privacy risk in machine learning: An-
alyzing the connection to overfitting. In 2018 IEEE
31st computer security foundations symposium (CSF),
pages 268–282. IEEE, 2018. 2, 5, 6, 7, 13
[45] Oualid Zari, Chuan Xu, and Giovanni Neglia. Effi-
cient passive membership inference attack in federated
learning. arXiv preprint arXiv:2111.00430, 2021. 1,
2

[46] Chengliang Zhang, Suyi Li, Junzhe Xia, Wei Wang,
Feng Yan, and Yang Liu. {BatchCrypt}: Efficient
homomorphic encryption for {Cross-Silo} federated
learning. In 2020 USENIX annual technical confer-
ence (USENIX ATC 20), pages 493–506, 2020. 8
[47] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin,
and David Lopez-Paz. mixup: Beyond empirical
risk minimization. arXiv preprint arXiv:1710.09412,
2017. 5, 7, 12, 13
[48] Jingwen Zhang, Jiale Zhang, Junjun Chen, and Shui
Yu. Gan enhanced membership inference: A passive
local attack in federated learning. In ICC 2020-2020
IEEE International Conference on Communications
(ICC), pages 1–6. IEEE, 2020. 1
[49] Qinqing Zheng, Shuxiao Chen, Qi Long, and Wei-
jie Su. Federated f-differential privacy. In Interna-
tional Conference on Artificial Intelligence and Statis-
tics, pages 2251–2259. PMLR, 2021. 5, 7, 12
[50] Ligeng Zhu, Zhijian Liu, and Song Han. Deep leak-
age from gradients. Advances in neural information
processing systems, 32, 2019. 1
[51] Eckart Zitzler and Simon K ¨unzli. Indicator-based se-
lection in multiobjective search. In International con-
ference on parallel problem solving from nature, pages
832–842. Springer, 2004. 5, 13, 14

A. Appendix
A.1. Dataset and Training Details
The CIFAR-100 dataset [20] consists of 100 categories with
60,000 32 × 32 color images, where 50,000 images are
allocated for training and 10,000 images for testing. The
Dermnet dataset [1] includes 23 categories with a total of
19,500 images, where 15,500 images are allocated for train-
ing and 4,000 images for testing. Since the images have
varying sizes, we cropped them to a size of 64 ×64 pixels.
The Tiny Imagenet datasets [22] has 100000 images of 200
classes. Each class has 500 training images, 50 validation
images, and 50 test images. The member dataset we use
is the splited training datatset of target client and the non-
member datatset is consist of the one-tenth hold-out test
dataset and the sum of one-tenth training datset of the other
clients. In the IID setting, we uniformly and randomly dis-
tribute the samples of each class to each client. In the Non-
IID setting, we make the labels of each client’s training data
follow the Dirichlet distribution[15]. For image generative
task with diffusion model, we use 10 classes for training
model and attacking membership privacy. We run image
classification tasks on AlexNet and ResNet18 with NVIDIA
2060 GPU and run image generation task on laten diffusion
model with NVIDIA A100 GPU. The training parameters
details and dataset splited method of federated learning are
shown in Table 3.
A.2. Defense Methods
A.2.1. Gradient Perturbation
Client-level Differential Privacy. Differential Privacy
(DP) [8, 49] hides the membership of individual data by
clipping the gradients at the client level and adding Gaus-
sian noise. The magnitude of the noise controls the strength
of privacy protection: the larger the noise, the better the pri-
vacy protection, but the worse the model’s performance. In
the experiment, we set the DP noise standard deviation from
0.01 to 0.5 to achieve different levels of defense.
Gradient Quantization. Gradient quantization [12, 30] is
a technique used to reduce the precision of gradient updates
and mitigate information leakage. This algorithm quantizes
the values of gradients into discrete approximations, reduc-
ing the precision of the gradients. By reducing the detailed
information in the gradients, it lowers the sensitivity to in-
dividual data and improves privacy protection. The number
of bits used for quantization affects the privacy protection
effectiveness, where fewer bits introduce larger gradient er-
rors but provide better privacy protection. In the experi-
ment, we set the number of bits from 1 to 10 to achieve
different levels of defense.
Gradient Sparsification. The gradient sparsification algo-
rithm [11, 35, 40] reduces the risk of information leakage
by setting smaller absolute value elements in the gradient to
zero. The fewer non-zero elements in the gradient, the less
privacy leakage occurs. In the experiment, we set the rate
of gradient elements sparsified from 0.1 to 0.99 to achieve
different levels of defense.
A.2.2. Data Replacement
MixUp. MixUp method [9, 47] trains neural networks on
composite images created via linear combination of image
pairs. It has been shown to improve the generalization of the
neural network and stabilizes the training. The coefficient
of the linear combination is sampled from a Beta Distribu-
tion. We set the Beta Distribution parameter from 1e-5 to
1e5 to achieve different levels of defense.
Data Augmentation. Data Augmentation [37] includes
cropping, shifting, rotating, flipping, shearing, and color jit-
tering. We combine these augmentation schemes in differ-
ent amounts to achieve different levels of defense.
Data Sampling. In each local training epoch, clients may
choose to sample a portion of the training data instead of
using the entire dataset [23]. We set the portion from 0.1 to
1.0 to achieve different levels of defense.
A.3. Evaluation Metrics
Utility loss (Test error rate). In this paper, we quantify
the utility loss by using the test error as a metric. The test
error measures the accuracy of the model on a separate test
dataset, where a lower test error indicates better model util-
ity. The worst possible test error rate is 1, which means that
the model makes incorrect predictions for all instances in
the test dataset.
Privacy Leakage (AUC and attack TPR). We consider
attacks as a binary classification task, and the TPR@FPR of
the AUC can be used to measure the accuracy of the clas-
sification, which represents the effectiveness of the attack.
TPR@low FPR is a metric recently proposed for measuring
MIA (Membership Inference Attack). It focuses more on
the data that is most susceptible to attacks, and researchers
believe that using it as a metric can better characterize pri-
vacy protection in worst-case scenarios.
TPR (True Positive Rate) and FPR (False Positive Rate)
are two important metrics used to evaluate the performance
of binary classification models, such as machine learning
algorithms or diagnostic tests. They are calculated as fol-
lows:
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
where: TP (True Positives) represents the number of pos-
itive instances correctly classified as positive. FN (False
Negatives) represents the number of positive instances in-
correctly classified as negative. FP (False Positives) repre-
sents the number of negative instances incorrectly classified

Table 3: Training parameters for federated learning in this paper
Dataset CIFAR100 Dermnet Tiny ImageNet
Models AlexNet, ResNet18 AlexNet, ResNet18 Laten Diffusion Model
Communication epoch 300 300 20
Optimizer SGD SGD Adam
Initial learning rate 0.1 0.1 0.001
Learning rate decay 0.99 at each epoch 0.99 at each epoch Adaptive
Number of clients 10 10 10
Training set size for one client 5000 1500 1000
Testing set size 10000 4500 1000
(a) AlexNet-CIFAR100 Blackbox-Loss
 (b) AlexNet-CIFAR100 Grad-Cosine
 (c) ResNet18-CIFAR100 Blackbox-Loss
 (d) ResNet18-CIFAR100 Grad-Cosine
(e) AlexNet-CIFAR100 Loss-Series
 (f) AlexNet-CIFAR100 Avg-Cosine
 (g) ResNet18-CIFAR100 Loss-Series
 (h) ResNet18-CIFAR100 Avg-Cosine
(i) AlexNet-CIFAR100 FedMIA-I
 (j) AlexNet-CIFAR100 FedMIA-II
 (k) ResNet18-CIFAR100 FedMIA-I
 (l) ResNet18-CIFAR100 FedMIA-II
Figure 6: Figure (a)-(f) demonstrate the TPR@FPR=0.001 of various defence (including client-level differential privacy
(green line) [8], sparsification (blue line) [35], mixup (purple line) [47], data sampling (red line) [23], data augmentation (deep
blue line) and gradient, combination of data augmentation and sampling (yellow line)) under three attacks (Blackbox-Loss
[44], Loss-Series [10], FedMIA-I, Grad-Cosine, Avg-Cosine [24] and FedMIA-II are first, second and third row respectively).
A larger hypervolume (HV) [51] indicates a better Pareto front of privacy and utility.
as positive. TN (True Negatives) represents the number of
negative instances correctly classified as negative.
Hypervolume HV (). In order to compare Pareto fronts
achieved by different defense algorithms, we need to quan-

tify the quality of a Pareto front. To this end, we adopt the
hypervolume (HV) indicator [51] as the metric to evaluate
Pareto fronts. Definition 1 formally defines the hypervol-
ume.
Definition 1 (Hypervolume Indicator) . Let z =
{z1,··· ,z m} be a reference point that is an upper
bound of the objectives Y = {y1,...,y m}, such that
yi ≤ zi,∀i ∈ [m]. the hypervolume indicator HV z(Y )
measures the region betweenY andz and is formulated as:
HVz(Y ) = Λ
 (
q∈ Rmq∈
mY
i=1
[yi,z i]
)!
(13)
where Λ(·) refers to the Lebesgue measure.
We set the reference pointz of privacy leakage and utility
loss to be 1 and 100% respectively.
B. More experiment
Figure 6 demonstrates the privacy-utility tradeoff against
different attacks under various defense methods.
C. Proof of Theorem 1
Theorem 2. Given the threshold δ, let Vt be the member
sets estimated by ˆΛt andδ in communication round t. Let
˜V be the member sets estimated by ˜Λ andδ. Then we have
˜V⊂ (V1∪···∪ VT ). (14)
Proof. We employ proof by contradiction to establish the
theorem. Assume there exists an element v∈V such that
v /∈ (V1∪···∪V T ).
By definition, the setVt is defined as
Vt ={v|v∈V , ˆΛt(v)≥δ},
which represents the set of members determined by Eq.
(4) in the main text. Additionally, let V n
t denote the non-
member set determined by Eq. (4).
Now, consider an element v /∈ (V1∪···∪V T ). This
implies that ˆΛt(v)<δ for allt. Consequently, we have:
˜Λ(v) = 1
T
TX
t=1
ˆΛt(v)<δ.
This result indicates that v /∈V , which contradicts the as-
sumption thatv∈V .
Thus, the assumption leads to a contradiction, and the
proof is complete.