Published in Transactions on Machine Learning Research (09/2023)
Individual Privacy Accounting for
Differentially Private Stochastic Gradient Descent
Da Yu yuda3@mail2.sysu.edu.cn
Sun Yat-sen University
Gautam Kamath† * g@csail.mit.edu
Cheriton School of Computer Science
University of Waterloo
Janardhan Kulkarni * jakul@microsoft.com
Microsoft Research
Tie-Yan Liu* tyliu@microsoft.com
Microsoft Research
Jian Yin * issjyin@mail.sysu.edu.cn
Sun Yat-sen University
Huishuai Zhang * huzhang@microsoft.com
Microsoft Research
Reviewed on OpenReview: https: // openreview. net/ forum? id= l4Jcxs0fpC
Abstract
Differentially private stochastic gradient descent (DP-SGD) is the workhorse algorithm
for recent advances in private deep learning. It provides a single privacy guarantee to
all datapoints in the dataset. We proposeoutput-specific (ε,δ )-DP to characterize privacy
guarantees for individual examples when releasing models trained by DP-SGD. We also design
an efficient algorithm to investigate individual privacy across a number of datasets. We find
that most examples enjoy stronger privacy guarantees than the worst-case bound. We further
discover that the training loss and the privacy parameter of an example are well-correlated.
This implies groups that are underserved in terms of model utility simultaneously experience
weaker privacy guarantees. For example, on CIFAR-10, the averageε of the class with the
lowest test accuracy is 44.2% higher than that of the class with the highest accuracy. Our
code is available athttps://github.com/dayu11/individual_privacy_of_DPSGD.
1 Introduction
Differential privacy is a strong notion of data privacy, enabling rich forms of privacy-preserving data analysis
(Dwork et al., 2006; Dwork & Roth, 2014). Informally speaking, it quantitatively bounds the maximum
influence of any datapoint using a privacy parameterε, where smaller values ofε correspond to stronger
privacy guarantees. Training deep models with differential privacy is an active research area (Papernot et al.,
2017; Zhu et al., 2020; Anil et al., 2021; Yu et al., 2022; Li et al., 2022; Golatkar et al., 2022; Mehta et al.,
2022b; De et al., 2022; Bu et al., 2022; Mehta et al., 2022a). Models trained with differential privacy not only
provide theoretical privacy guarantees to their data owners but also are more robust against empirical attacks
(Rahman et al., 2018; Bernau et al., 2019; Carlini et al., 2019a; Jagielski et al., 2020; Nasr et al., 2021a).
Differentially private stochastic gradient descent (DP-SGD) is the most popular algorithm for differentially
private deep learning (Song et al., 2013; Bassily et al., 2014; Abadi et al., 2016). At each step, DP-SGD takes
†Supported by an NSERC Discovery Grant, an unrestricted gift from Google, and a University of Waterloo startup grant.
*Authors are listed in alphabetical order.
1
arXiv:2206.02617v7  [cs.LG]  25 Jul 2024

Published in Transactions on Machine Learning Research (09/2023)
0.8 1.6 2.4 3.2 4.0 4.8 5.6 6.4 7.2 8.0
0
2000
4000
6000
8000
10000
12000
14000
16000
18000
20000
22000
24000Count
CIFAR-10, test acc.=74.2, max i=7.8
0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4
0
1000
2000
3000
4000
5000
6000
7000
8000
9000
10000Count
MNIST, test acc.=97.1, max i=2.4
0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5
0
500
1000
1500
2000
2500
3000
3500
4000
4500
5000Count
UTKFace-Gender, test acc.=88.2, max i=4.5
0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5
0
500
1000
1500
2000
2500
3000
3500
4000
4500
5000Count
UTKFace-Age, test acc.=80.4, max i=4.2
Figure 1: Individual privacy parameters of models trained by DP-SGD. The value ofδ is 1× 10−5. The
dashed lines indicate30%, 50%, and 70% of datapoints. The black solid line shows the worst-case privacy
parameter.
the model from the previous step and the dataset as inputs. It adds isotropic Gaussian noise to the average
gradient of the current step. Models trained with DP-SGD satisfy(ε,δ )-differential privacy. The canonical
notion of differential privacy, including(ε,δ )-DP, considers theworst-case privacy over all possible inputs. In
the case of DP-SGD, this results in the privacy cost of all examples being computed with the largest possible
magnitude of individual gradients, i.e., the gradient clipping threshold.
In practice, we may care more about the privacy guarantees of the models that will be deployed, which
depend on the observed training trajectories. Broadly speaking, different examples may have very different
impacts on a trained model (Feldman & Zhang, 2020; Jiang et al., 2021). Some examples may be easier to
learn and hence the magnitudes of their individual gradients along the observed training trajectory could be
much smaller than the clipping threshold. Such a fine-grained privacy guarantee can not be inferred by the
canonical (ε,δ )-DP because it requires the privacy guarantee to hold for all possible datasets and training
trajectories. In this paper, we defineoutput-specific (ε,δ )-DP, which adapts to the training trajectory of the
model to analyze the individual privacy of DP-SGD. Our definition captures the impact of various factors,
such as the training set and algorithmic randomness, on individual privacy. We also develop algorithms to
efficiently and accurately estimate individual privacy.
It turns out that, unsurprisingly, for common benchmarks, many examples experience much stronger privacy
guarantees than implied by the worst-case DP analysis. To illustrate this, we plot the individual privacy
parameters of four benchmark datasets in Figure 1. Experimental details, as well as more results, are in
Sections 4 and 5. To the best of our knowledge, this paper is the first to reveal the disparity in individual
privacy when running DP-SGD.
Further, we demonstrate a strong correlation between the privacy parameter of an example and its final
training loss. That is, the examples with higher training loss also have higher privacy parameters in general.
This suggests that the examples that suffer unfairness in terms of worse privacy are also the ones that have
worse utility. See Figure 2 for an illustration. While prior works have shown that underrepresented groups
experience worse utility (Buolamwini & Gebru, 2018), and that these disparities are amplified when models
are trained privately (Bagdasaryan et al., 2019; Hansen et al., 2022; Noe et al., 2022; Lowy et al., 2022), we
are the first to show that the privacy guaranteeand utility are negatively impacted concurrently. In contrast,
prior work that takes a worst-case perspective for privacy accounting, results in a uniform privacy guarantee
for all training examples. For instance, when running gender classification on UTKFace, the averageε of the
race with the lowest test accuracy is 35.1% higher than that of the race with the highest accuracy.
1.1 Related Work
Several works have explored individual privacy analysis in differentially private learning. Jorgensen et al.
(2015); Mühl & Boenisch (2022), and the work subsequent to ours of Koskela et al. (2022); Boenisch et al.
(2023), design learning algorithms that satisfy prespecified individual privacy parameters. Those prespecified
parameters are independent of the learning algorithm, e.g., in some applications different users may have
different expectations of privacy. Wang (2019) definesPer-instance differential privacyto analyze individual
privacy when the target example is put in a fixed dataset. Redberg & Wang (2021) investigate per-instance
2

Published in Transactions on Machine Learning Research (09/2023)
Ship Auto. Truck Frog Horse Air. Dog Deer Bird Cat
Subgroup Name
60
70
80
90Accuracy (in %)
CIFAR-10
T est Acc.
Indian Black White Asian
Subgroup Name
80.0
82.5
85.0
87.5
90.0
92.5
95.0
97.5
100.0Accuracy (in %)
UTKFace-Gender
T est Acc.
4.5
5.0
5.5
6.0
6.5
7.0
7.5
Average 
Privacy Parameter 
 1.4
1.6
1.8
2.0
2.2
2.4
2.6
2.8
Average 
Privacy Parameter 
Figure 2: Accuracy and averageε of different groups on CIFAR-10 and UTK-Face. Groups with worse
accuracy also have worse privacy in general.
DP of the objective perturbation algorithm (Kifer et al., 2012). Golatkar et al. (2022) explore per-instance
DP of differentially private batch gradient descent. Redberg & Wang (2021); Golatkar et al. (2022) focus
on (strongly) convex objective functions because otherwise fixing the dataset is not sufficient to determine
the individual privacy parameters. In this work, we define output-specific(ε,δ )-DP that allows us to study
the individual privacy of non-convex models trained by DP-SGD. We also conduct experiments on several
datasets and demonstrate a strong correlation between privacy and utility.
Feldman & Zrnic (2021) design individualprivacy filtersto make use of the variation in individual sensitivity.
The filters allow examples with smaller per-step privacy costs to run for more steps until the accumulated
cost reaches a target budget. Additionally, Feldman & Zrnic (2021) study the individual privacy of DP-GD
and demonstrate that examples often experience stronger privacy than worst-case analysis suggests. However,
computing individual privacy requires calculating gradient norms for all training examples at every step. In
this work, we present an efficient algorithm for estimating the individual privacy of DP-SGD. Our algorithm
accurately estimates individual privacy while only occasionally computing the gradient norms of all examples.
Our output-specific (ε,δ )-DP and the notion ofex-post DP by Ligett et al. (2017) both tailor the privacy
guarantee to algorithm outcomes. Ex-post DP bounds the ratio between two probability/density functions at
a single outcome. It can be generalized to pure differential privacy ((ε, 0)-DP). In contrast, DP-SGD uses
Gaussian mechanisms and provides approximate differential privacy ((ε,δ )-DP). There is no clean conversion
between (ε,δ )-DP and ex-post DP (Meiser, 2018). Therefore, our notion is necessary for analyzing individual
privacy within the(ε,δ )-DP framework.
2 Preliminaries
We first give some background on DP-SGD and explain why the canonical(ε,δ )-DP is not suitable for
measuring individual privacy. Then we define output-specific(ε,δ )-DP. Finally, we give empirical evidence
showing that providing the same privacy to all examples is not ideal.
2.1 Background on Differentially Private SGD
The privacy guarantee of DP-SGD is measured by(ε,δ )-differential privacy.
Definition 1. [(ε,δ )-DP] An algorithmA :D→O satisfies (ε,δ )-differential privacy if for any pair of
neighboring datasets D, D′∈D and any subset of outputsS⊂O it holds that
Pr[A(D)∈ S]≤eε Pr[A(D′)∈ S] +δ.
Two datasets D, D′ are neighboring datasets if they only differ in one datapoint. DP-SGD uses Rényi
differential privacy (RDP) (Mironov, 2017) in privacy accounting to get a tighter composition bound (Abadi
et al., 2016). After training, the accumulated RDP is converted to(ε,δ )-DP. RDP measures the Rényi
3

Published in Transactions on Machine Learning Research (09/2023)
divergence at different orders. The Rényi divergence between two probability distributionsµ and ν at order
α is
Dα(µ||ν) = 1
α− 1 log
∫
(dµ
dν )αdν.
Let D↔
α (µ||ν) = max(Dα(µ||ν),Dα(ν||µ)) be the maximum divergence of two directions. The definition of
RDP is as follows.
Definition 2. [Rényi differential privacy (Mironov, 2017)] A randomized algorithmA :D→O satisfies
(α,ρ )-RDP if for any neighboring datasetsD, D′∈D it holds that
D↔
α (A(D)||A(D′))≤ρ.
WhenA is a deep learning algorithm, it is infeasible to directly measure the output distributions because of
the non-convex nature of neural networks. To address this, DP-SGD makes each gradient update differentially
private and uses the composition property of differential privacy to reason about the overall privacy cost.
Definition3. [Composition of RDP (Mironov, 2017)] LetA1 :D→O 1 be (α,ρ 1)-RDP andA2 :O1×D→O 2
be (α,ρ 2)-RDP, then the mechanism defined as(X,Y ), where X ∼A 1(D) and Y ∼A 2(X, D), satisfies
(α,ρ 1 +ρ2)-RDP.
The output of the composed algorithm is a tuple containing the outputs from all steps. Consequently, the
output of DP-SGD at stepT is a sequence of models(θ1,θ 2,...,θ T ).
The privacy cost of a target example depends on its gradients along the training trajectory. We formalize
the output distributions of DP-SGD at each step to illustrate this. DP-SGD uses Poisson sampling, i.e.,
each example is sampled independently with probabilityp. Letv =∑
i∈Mgi be the sum of the minibatch
gradients of D, where M is the set of sampled indices. Consider also a neighboring datasetD′ that has one
datapointd (with gradientg) added. Because of Poisson sampling, the output is exactlyv with probability
1−p (g is not sampled) and isv′ =v +g with probabilityp (g is sampled). After adding isotropic Gaussian
noise, the output distributions of two neighboring datasets are
A(D)∼N (v,σ 2I). (1)
A(D′)∼N (v,σ 2I) with prob. 1−p,
A(D′)∼N (v′,σ 2I) with prob.p. (2)
The RDP ofd at the current step is the Rényi divergences between Equation (1) and (2). For a givenσ
and p, the divergences are determined by theL2 norm ofg =v′−v. Therefore, a larger gradient would
result in a larger privacy cost. If we consider all possibleθt−1∈O t−1 as required by Definition 1, we would
have to compute the divergences with the largest possible magnitude ofg. Abadi et al. (2016) use the
gradient clipping threshold to compute the privacy cost, which results in a worst-case privacy analysis for
every example.
2.2 Output-specific (ε,δ )-Differential Privacy
We define output-specific individual(ε,δ )-differential privacy to provide a fine-grained analysis of individual
privacy. It makes the privacy parameterε a function of the outputs and the target datapoint.
Definition 4. [Output-specific individual(ε,δ )-DP] Fix a datapointd and a set of outcomesA⊂O . Let
D be an arbitrary dataset andD′ = D∪{d}, an algorithmA :D→O satisfies output-specific individual
(ε(A,d),δ )-DP ford at A if for anyS⊂ A
Pr[A(D)∈ S]≤eε(A,d) Pr[A(D′)∈ S] +δ,
Pr[A(D′)∈ S]≤eε(A,d) Pr[A(D)∈ S] +δ.
With ε being a function of the specified set of outcomesA, we can analyze the individual privacy of models
trained by DP-SGD. This is done by fixing the models from the firstT− 1 steps (θ1,...,θ T−1), which fully
specify the gradients along the training. We note that when the value ofδ is larger than the probability of
4

Published in Transactions on Machine Learning Research (09/2023)
0 50 100 150 200
Epoch
0
2
4
6
8
10
12
14
16Median Gradient Norm
DP-SGD with max = 7.5
Airplane
Automobile
Cat
Figure 3: Median of gradient norms of different classes when training a ResNet-20 model on CIFAR-10.
A, Definition 4 is trivially satisfied for anyε, thus it along does not provide meaningful privacy guarantee.
Consequently, this definition should not guide the design of new algorithms in such scenarios. In this work, we
adhere to the original implementation of DP-SGD and use Definition 4 to understand the individual privacy
of DP-SGD. The resulting privacy parameters provide insights into the model’s individual privacy and reflect
empirical risks against membership inference attacks (Shokri et al., 2017), as shown in Appendix D.
2.3 Gradients of Different Examples Vary Significantly
At each step of DP-SGD, the privacy cost of an example depends on its gradient at the current step (see
Section 2.1 for details). In this section, we empirically show gradients of different examples vary significantly to
demonstrate that different examples experience very different privacy costs. We train a ResNet-20 model with
DP-SGD on CIFAR-10. The maximum clipping threshold is the median of gradient norms at initialization.
More implementation details are in Section 4. We plot the median of gradient norms of three different
classes in Figure 3. The gradient norms of different classes show significant stratification. Such stratification
naturally leads to different privacy costs. This suggests that it is meaningful to further quantify individual
privacy parameters.
3 Individual Privacy of DP-SGD
Algorithm 1 shows the implementation of DP-SGD1 (Abadi et al., 2016). Theorem 3.1 gives the individual
privacy analysis of DP-SGD. Algorithm 2 gives the pseudocode for computing individual privacy parameters.
At each step, Algorithm 2 uses the (estimated) individual gradient norms to compute per-step Rényi differential
privacy (RDP) for every example. It also updates the individual gradient norms and the accumulated RDP.
We introduce two arguments in Algorithm 2 to reduce the computational cost of individual privacy accounting.
The first one is the frequencyK of computing batch gradient norms and the second one is whether to round
individual gradient norms with a small constantr. More discussion on these two arguments could be found
in Section 3.1 and 3.2.
Theorem 3.1. Let{θ1,...,θ t−1} be the observed models at stept. Suppose we run Algorithm 1 withK = 1
and without rounding, then Algorithm 1 satisfies(o(i)
α + log(1/δ)
α−1 ,δ )-output-specific individual DP for theith
example at A = (θ1,...,θ t−1,Ot), whereo(i)
α is the accumulated RDP at orderα andOt is the range ofAt.
Proof Sketch.At stept, given the observed models(θ1,...,θ t−1), the composited training algorithm is
ˆA(t) = (A1(D),A2(θ1, D),..., At(θ1,...,θ t−1, D)).
1Our implementation of DP-SGD follows the privacy analysis in Abadi et al. (2016) which uses Poisson sampling. We note
that many existing implementations of DP-SGD use shuffle data instead of Poisson sampling to enforce stochasticity. Shuffle
data is easier to implement but using it would create a mild discrepancy with the analysis in Abadi et al. (2016). Formal privacy
analysis of shuffle data requiresprivacy amplification by shuffling(Koskela et al., 2023; Wang, 2023; Feldman et al., 2023).
5

Published in Transactions on Machine Learning Research (09/2023)
Algorithm 1Differentially Private SGD
Input: Clipping thresholdC, noise varianceσ2, sampling probabilityp, number of stepsT.
Let{Z (i) =C}n
i=1 be the estimates of individual gradient norms, initialized asC.
Let{o(i) = 0}n
i=1 be the accumulated individual RDP.
for t = 0toT − 1 do
//Individual privacy accounting.
Run Algorithm 2 with{Z (i)}n
i=1 and{o(i)}n
i=1.
Update{Z (i)}n
i=1 and{o(i)}n
i=1 with the results of Algorithm 2.
//Run DP-SGD as usual.
Sample a minibatch of gradients{g(Ij )}|I|
j=1 with probabilityp , whereI is the sampled indices.
Clip gradients ¯g(Ij ) =clip(g(Ij ),C ).
Update modelθt =θt−1−η(∑¯g(Ij ) +z), wherez∼N (0,σ 2I).
end for
Algorithm 2Individual Privacy Accounting for DP-SGD
Input: Individual gradient norms{Z (i)}n
i=1, accumulated individual RDP{o(i)}n
i=1, frequency K of
updating{Z (i)}n
i=1, rounding precisionr, current iterationt.
if t modK = 0 then
//Update individual sensitivity.
Compute batch gradient norms{
‖‖g(i)‖‖
2}n
i=1.
Update Z (i) = min(
‖‖g(i)‖‖
2,C ).
if use roundingthen
//Reduce the number of different norms.
Update{Z (i) = arg minc∈C(|c−Z (i)|)}n
i=1, where C ={r, 2r,...,C } contains all possible norms.
end
end
//Compute the current step RDP.
Compute the Rényi divergences between Equation 1 and 2 numerically withZi, p, andσ2 and store the
result inρ(i).
//Update the accumulated RDP.
o(i) =o(i) +ρ(i).
return {Z (i)}n
i=1,{o(i)}n
i=1
We first use the composition theorem to show the accumulated RDP is the RDP ofˆA(t). Then we prove
the RDP bound on ˆA(t) gives an output-specific-(ε,δ )-DP bound on Algorithm 1. We relegate the proof to
Appendix A.
Remark 1. Algorithm 2 does not change the worst-case privacy guarantee (Definition 1) of DP-SGD because
it does not modify the update rule.
In Theorem 3.1, we run Algorithm 2 withK = 1 and without rounding individual gradient norms. This
configuration is computationally expensive for two reasons. Firstly, settingK = 1 requires computing batch
gradient norms at each SGD update. Secondly, the number of unique gradient norms is large without rounding.
Each unique gradient norm corresponds to a different single-step RDP that needs to be computed numerically.
In Section 3.1, we give more details on the computational challenges. In Section 3.2, we use a largerK and
round individual gradient norms to provide estimations of individual privacy. This greatly improves the
efficiency of Algorithm 2. In Section 3.3, we show the estimates of individual privacy parameters are accurate.
6

Published in Transactions on Machine Learning Research (09/2023)
0 2 4 6 8
Actual 
0
2
4
6
8Estimated 
Pearson's r = 0.998
Avg. | | = 0.09
Max | | = 0.54
CIFAR-10: = 0.5
y = x
y = 7.8
0 1 2 3
Actual 
0
1
2
3Estimated 
Pearson's r = 0.995
Avg. | | = 0.07
Max | | = 0.25
MNIST: = 0.5
y = x
y = 2.4
0 2 4
Actual 
0
1
2
3
4
5Estimated 
Pearson's r = 0.999
Avg. | | = 0.05
Max | | = 0.35
UTKFace-Gender: = 0.5
y = x
y = 4.5
0 2 4
Actual 
0
1
2
3
4
5Estimated 
Pearson's r = 0.999
Avg. | | = 0.05
Max | | = 0.29
UTKFace-Age: = 0.5
y = x
y = 4.2
0 2 4 6 8
Actual 
0
2
4
6
8Estimated 
Pearson's r = 0.999
Avg. | | = 0.06
Max | | = 0.35
CIFAR-10: = 1
y = x
y = 7.8
0 1 2 3
Actual 
0
1
2
3Estimated 
Pearson's r = 0.998
Avg. | | = 0.04
Max | | = 0.17
MNIST: = 1
y = x
y = 2.4
0 2 4
Actual 
0
1
2
3
4
5Estimated 
Pearson's r = 1.000
Avg. | | = 0.02
Max | | = 0.14
UTKFace-Gender: = 1
y = x
y = 4.5
0 2 4
Actual 
0
1
2
3
4
5Estimated 
Pearson's r = 1.000
Avg. | | = 0.02
Max | | = 0.19
UTKFace-Age: = 1
y = x
y = 4.2
0 2 4 6 8
Actual 
0
2
4
6
8Estimated 
Pearson's r = 1.000
Avg. | | = 0.04
Max | | = 0.32
CIFAR-10: = 3
y = x
y = 7.8
0 1 2 3
Actual 
0
1
2
3Estimated 
Pearson's r = 0.999
Avg. | | = 0.01
Max | | = 0.07
MNIST: = 3
y = x
y = 2.3
0 2 4
Actual 
0
1
2
3
4
5Estimated 
Pearson's r = 1.000
Avg. | | = 0.01
Max | | = 0.06
UTKFace-Gender: = 3
y = x
y = 4.5
0 2 4
Actual 
0
1
2
3
4
5Estimated 
Pearson's r = 1.000
Avg. | | = 0.01
Max | | = 0.17
UTKFace-Age: = 3
y = x
y = 4.2
Figure 4: Privacy parameters based on estimations of individual gradient norms (ε) versus those based on
exact ones (´ε). The value ofγ denotes the number of updates of full gradient norms per epoch. The horizontal
line shows the worst-case privacy guarantee.
3.1 Computational Challenges of Individual Privacy
The first challenge is that computing exact privacy costs at each step requires batch gradient norms, which
is impractical when running SGD. At each update, we need the gradient of every example to compute the
corresponding RDP between Equation 1 and 2. The worst-case privacy analysis of DP-SGD does not have
this problem because it simply assumes all examples have the maximum possible gradient, i.e., the clipping
threshold.
The second challenge is that the RDP of every example at each step needs to be computed numerically. It
has been shown that numerical computations are necessary to get tight bounds on the Rényi divergences
between Equation 1 and 2 (Abadi et al., 2016; Wang et al., 2019; Mironov et al., 2019; Gopi et al., 2021). In
Abadi et al. (2016), only one numerical computation is required because all examples are assumed to have the
worst-case privacy cost. However, when computing individual privacy parameters, the number of numerical
computations is the same as the number of different gradients that could be as large asn×T, wheren is the
dataset size andT is the number of iterations.
3.2 Improving the Efficiency of Individual Privacy Accounting
We only compute the batch gradient norms everyK iteration to reduce the computational overhead. The
norms are then used to estimate the privacy costs for the subsequent iterations. We note that providing
estimates of privacy costs is inevitable when the computational budget is limited. This is because, by the
nature of SGD, one does not have the exact individual gradient norms at every iteration. In Appendix B.2,
we explore another design choice which is to clipg(i) withZ (i) in Algorithm 1. Although this slightly changes
the implementation of DP-SGD, Algorithm 2 would return the exact privacy costs. We run experiments with
7

Published in Transactions on Machine Learning Research (09/2023)
Table 1: Computational costs of computing individual privacy parameters for CIFAR-10.
w/ rounding w/o rounding
# of computations 1 × 102 1 × 107
Time (in seconds) < 3 ∼ 2.6 × 104
this design choice and report the results in Appendix B.2. Our observations in the main text still hold in
Appendix B.2.
To reduce the number of numerical computations, we round individual gradient norms with a small constant
r. Because the maximum clipping thresholdC is a constant, then, by the pigeonhole principle, there are at
most⌈C/r⌉ different values of gradient norms, and hence there are at most⌈C/r⌉ different values of RDP
between Equation (1) and (2). Note thatr should be small enough to avoid underestimation of RDP. We set
r = 0.01C throughout this paper.
We compare the computational costs with/without rounding in Table 1. We run the numerical method
in Mironov et al. (2019) once for every different value of RDP (with the default setup in the Opacus library
(Yousefpour et al., 2021a)). We run DP-SGD on CIFAR-10 for 200 epochs. The full gradient norms are
updated once per epoch. All results in Table 1 use multiprocessing with 5 cores of an AMD EPYC™
7V13 CPU. With rounding, the overhead of computing individual privacy parameters is negligible. The
computational cost without rounding is more than 7 hours.
3.3 Estimates of Individual Privacy Are Accurate
We run Algorithm 2 with the setup in Section 3.2 and compare the results with ground-truth values. To
compute the ground-truth individual privacy, we randomly sample 1000 examples before training. During
training, we compute the exact privacy costs for the same 1000 examples at every iteration.
We compute the Pearson correlation coefficient between the estimations and the ground-truth values. We also
compute the average and the worst absolute errors. We report results on MNIST, CIFAR-10, and UTKFace.
Details about the experiments are in Section 4. We plot the results in Figure 4. The estimations ofε are close
to those ground-truth values (Pearson’sr> 0.99) even when we only update the gradient norms every two
epochs (γ = 0.5). Updating batch gradient norms more frequently further improves the estimation, though
doing so would increase the computational overhead.
It is worth noting that the maximum clipping thresholdC affects the computed privacy parameters. LargeC
increases the variation of gradient norms (and hence the variation of privacy parameters) but leads to large
noise variance while smallC suppresses the variation and leads to large gradient bias. Large noise variance
and gradient bias are both harmful to learning (Chen et al., 2020; Song et al., 2021). In Appendix C, we
show the influence of using differentC on both accuracy and privacy.
3.4 What Can We Do with Individual Privacy Parameters?
Individual privacy parameters depend on the private data and are thus sensitive. They can not be released
publicly without care. We describe some approaches to safely make use of individual privacy parameters.
The first approach is to releaseεi to the owner ofdi. This approach does not incur additional privacy cost
for two reasons. First, it is safe fordi because only the rightful owner seesεi. Second, releasingεi does
not increase the privacy cost of any other exampledj̸=di. This is because computingεi can be seen as a
post-processing of (θ1,...,θ t−1), which is reported in a privacy-preserving manner. We prove the claim in
Theorem 3.2.
Theorem 3.2. LetA :D → Obe an algorithm that is (εj,δ )-output-specific individual DP fordj at
A⊂O . Let f (·,di) :O→R×O be a post-processing function that returns the privacy parameter ofdi
(̸=dj) and the training trajectory. We havef (·,di) is (εj,δ )-output-specific DP fordj at F⊂R×O where
F ={f (a,di) :a∈ A} is all possible post-processing results.
8

Published in Transactions on Machine Learning Research (09/2023)
Table 2: Statistics of individual privacy parameters can be accurately released with minor privacy costs. The
average estimation error rate is 1.13% for MNIST and 0.91% for CIFAR-10. The value ofδ is 1× 10−5.
MNIST A verage 0.1-quantile 0.3-quantile Median 0.7-quantile 0.9-quantile
Non-private 0.686 0.236 0.318 0.431 0.697 1.682
ε = 0.1 0.681 0.238 0.317 0.436 0.708 1.647
CIF AR-10 A verage 0.1-quantile 0.3-quantile Median 0.7-quantile 0.9-quantile
Non-private 5.942 2.713 4.892 6.730 7.692 7.815
ε = 0.1 5.939 2.801 4.876 6.744 7.672 7.923
Proof. First note that the construction off (·,di) does not increase the privacy cost ofdj because it is
independent ofdj. Without loss of generality, letD, D′∈D be the neighboring datasets whereD′ = D∪{dj}.
Let S⊂ F be an arbitrary event andT ={a∈ A :f (a,di)∈ S}. Becausef is a bijective function, we have
Pr [f (A(D),di)∈ S] = Pr [A(D)∈ T] (3)
≤eεj Pr [A(D′)∈ T] +δ (4)
=eεj Pr [f (A(D′),di)∈ S] +δ, (5)
which completes the proof. Using a bijective post-processing function is necessary for Theorem 3.2 to hold.
Otherwise, there may be someo /∈ A and a∈ A that have the same processed output, which invalids the
derivation from Equation 4 to Equation 5.
The second approach is to privately release aggregate statistics of the population, e.g., the average or quantiles
of theε values. Recent works have demonstrated such statistics can be published accurately with a minor
privacy cost Andrew et al. (2021). Specifically, we privately release the average and quantiles of theε values.
We report the results on CIFAR-10 and MNIST. For releasing the average value ofε, we use the Gaussian
Mechanism. For releasing the quantiles, we use 20 steps of batch gradient descent to solve the objective
function in Andrew et al. (2021) with the default setup. The results are in Table 2. The released statistics
are close to the actual values under(0.1, 10−5)-DP.
Finally, individual privacy parameters can also serve as a powerful tool for a trusted data curator to improve
the model quality. By analyzing the individual privacy parameters of a dataset, a trusted curator can focus
on collecting more data representative of the groups that have higher privacy risks to mitigate the disparity
in privacy.
4 Individual Privacy Parameters on Different Datasets
In Section 4.1, we first show the distribution of individual privacy parameters on four tasks. Then we study
how individual privacy parameters correlate with training loss in Section 4.2. The experimental setup is as
follows.
Datasets. We use two benchmark datasets MNIST (n = 60000) and CIFAR-10 (n = 50000) (LeCun et al.,
1998; Krizhevsky, 2009) as well as the UTKFace dataset (n≃ 15000) (Zhang et al., 2017) that contains the
face images of four different races (White,n≃ 7000; Black,n≃ 3500; Asian,n≃ 2000; Indian,n≃ 2800).
We construct two classification tasks on UTKFace: predicting gender, and predicting whether the age is
under 30.2 We slightly modify the dataset between these two tasks by randomly removing a few examples to
ensure each race has balanced positive and negative labels.
Models and hyperparameters.For CIFAR-10, we use the WRN16-4 model in De et al. (2022), which
achieves advanced performance in private setting. We follow the implementation details in De et al. (2022)
expect their data augmentation method to reduce computational cost. For MNIST and UTKFace, we use
2We acknowledge that predicting gender and age from images may be problematic. Nonetheless, as facial images have
previously been highlighted as a setting where machine learning has disparate accuracy on different groups, we revisit this
domain through a related lens. The labels are provided by the dataset curators.
9

Published in Transactions on Machine Learning Research (09/2023)
Figure 5: Privacy parameters and final training losses. Each point shows the final training loss and privacy
parameter of one example. Pearson’sr is computed between privacy parameters and log loss values.
ResNet20 models with batch normalization layers replaced by group normalization layers. For UTKFace, we
initialize the model with weights pre-trained on ImageNet.
We setC = 1 on CIFAR-10, following De et al. (2022). For MNIST and UTKFace, we setC as the median of
gradient norms at initialization, following the suggestion in Abadi et al. (2016). The privacy cost of using the
median is not taken care of. However, the median of gradient norms could be released accurately with a
small privacy cost3. The batchsize is 4096 for CIFAR-10 and 1024 for MNIST and UTKFace. The training
epoch is 300 for CIFAR-10 and 100 for MNIST and UTKFace. For a target maximumε, we use the package
in the Opacus library to find the corresponding noise variance Yousefpour et al. (2021b). We update the
batch gradient norms three times per epoch for all experiments in this section (the case ofγ = 3 in Figure 4).
All experiments are run on single Tesla V100 GPUs with 32G memory. Our source code will be publicly
available.
4.1 Individual Privacy Parameters Vary Significantly
Figure 1 shows the individual privacy parameters on all datasets. The privacy parameters vary across a large
range on all four tasks. On the CIFAR-10 dataset, the maximumεi is 7.8 while the minimumεi is 1.0. When
running gender classification on the UTKFace dataset, the maximumεi is 4.5 while the minimumεi is only
0.1.
We also observe that, for easier tasks, more examples enjoy stronger privacy guarantees. For example,∼35%
of examples reach the worst-caseε on CIFAR-10 while only∼3% do so on MNIST. This may be because the
loss decreases quickly when the task is easy, resulting in gradient norms also decreasing and thus stronger
privacy guarantees.
4.2 Privacy Parameters and Loss Are Positively Correlated
We study how individual privacy parameters correlate with final training loss values. The privacy parameter
of one example depends on its gradient norms along the training. In strongly convex optimization, the loss
value of an example is reflected in the norm of its gradient. However, for non-convex deep models, there is
no clear relation between the final training loss of one example and its gradient norms. Therefore, we run
experiments to reveal the empirical correlation between privacy and utility.
We visualize individual privacy parameters and the final training loss values in Figure 5. The individual
privacy parameters increase with loss until they reach the maximumε. To quantify the order of correlation,
we further fit the points with one-dimensional logarithmic functions and compute the Pearson correlation
coefficients between the privacy parameters and log loss values. The Pearson correlation coefficients are larger
than 0.9 on all datasets, showing a logarithmic correlation between the privacy parameter of a datapoint
and its final training loss. In Appendix E, we experiment withC = 5 and C = 10 on CIFAR-10 to study
the correlation between training loss and privacy under various clipping thresholds. The Pearson correlation
coefficients are 0.89 and 0.9 for C = 5 and C = 10, respectively, suggesting that there is still a positive
logarithmic correlation.
3For instance, if we use the algorithm in Andrew et al. (2021) to privatize the median gradient norm of the UTKFace-Gender
dataset with (ε = 0.1,δ = 1 × 10−5). The non-private median is 15.73 and the privatized median is 15.82.
10

Published in Transactions on Machine Learning Research (09/2023)
1 7 6 4 0 2 3 9 5 8
Subgroup Name
93
94
95
96
97
98
99
100Accuracy (in %)
MNIST
T est Acc.
Asian White Indian Black
Subgroup Name
65
70
75
80
85
90
95Accuracy (in %)
UTKFace-Age
T est Acc.
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
Average 
Privacy Parameter 
 1.50
1.75
2.00
2.25
2.50
2.75
3.00
3.25
Average 
Privacy Parameter 
Figure 6: Accuracy and averageε of different groups on MNIST and UTK-Age. Groups with worse accuracy
also have worse privacy in general.
5 Groups Are Simultaneously Underserved in Both Accuracy and Privacy
It is well-documented that the accuracy of machine learning models may be unfair for different subpopulations
(Buolamwini & Gebru, 2018; Bagdasaryan et al., 2019; Suriyakumar et al., 2021). Our finding demonstrates
that this disparity may be simultaneous in terms of both accuracyand privacy. We empirically verify this by
plotting the averageε and test accuracy of different groups. The experiment setup is the same as Section 4.
For CIFAR-10 and MNIST, the groups are the data from different classes, while for UTKFace, the groups are
the data from different races.
We plot the results in Figure 2 and 6. The groups are sorted based on the averageε. The test accuracy of
different groups correlates well with the averageε values. Groups with worse accuracy do have worse privacy
guarantees in general. On CIFAR-10, the averageε of the ‘Cat’ class (which has the worst test accuracy)
is 44.2% higher than the averageε of the ‘Automobile’ class (which has the highest test accuracy). On
UTKFace-Gender, the averageε of the group with the lowest test accuracy (‘Asian’) is 35.1% higher than the
averageε of the group with the highest accuracy (‘Indian’). Similar observation also holds on other tasks. To
the best of our knowledge, our work is the first to reveal this simultaneous disparity. In Appendix D, we run
membership inference attacks to show the disparity in privacy parameters reflects the disparity in empirical
privacy risks.
6 Conclusion
We define output-specific individual(ε,δ )-differential privacy to characterize the individual privacy guarantees
of models trained by DP-SGD. We also design an efficient algorithm to accurately estimate the individual
privacy parameters. We use this new algorithm to examine individual privacy guarantees on several datasets.
Significantly, we find that groups with worse utility also suffer from worse privacy. This new finding reveals
the complex while interesting relation among utility, privacy, and fairness. It suggests that mitigating the
utility fairness under differential privacy is more tricky than doing so in the non-private case. This is because
classic methods such as upweighting underserved examples would exacerbate the disparity in privacy. We
hope that our work sheds new light on this timely topic.
Broader Impact
One way to utilize individual privacy parameters is to release them to corresponding users (see Section 3.4
for details). This provides a more accurate, and hence more responsible, privacy report. However, releasing
individual privacy parameters may pose new challenges when a machine learning system enables data deletion,
also known as machine unlearning (Ginart et al., 2019; Bourtoule et al., 2019). Data deletions may worsen
the privacy guarantees of the remaining examples (Carlini et al., 2022). Moreover, groups with largerε may
send deletion requests more frequently than others, which could further deteriorate their privacy guarantees
(Hashimoto et al., 2018). It’s important to keep these considerations in mind when implementing both data
deletion and individual privacy accounting in real-world settings.
11

Published in Transactions on Machine Learning Research (09/2023)
Acknowledgments
The authors express their gratitude to Yu-Xiang Wang and Saeed Mahloujifar for their valuable comments
on an earlier version of this paper, and to the anonymous reviewers for their insightful feedback.
References
Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang.
Deep learning with differential privacy. InProceedings of ACM Conference on Computer and Communica-
tions Security, 2016.
Galen Andrew, Om Thakkar, Brendan McMahan, and Swaroop Ramaswamy. Differentially private learning
with adaptive clipping. InAdvances in Neural Information Processing Systems, 2021.
Rohan Anil, Badih Ghazi, Vineet Gupta, Ravi Kumar, and Pasin Manurangsi. Large-scale differentially
private BERT.arXiv preprint arXiv:2108.01624, 2021.
Eugene Bagdasaryan, Omid Poursaeed, and Vitaly Shmatikov. Differential privacy has disparate impact on
model accuracy. InAdvances in Neural Information Processing Systems, 2019.
Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efficient algorithms
and tight error bounds. InProceedings of the 55th Annual IEEE Symposium on Foundations of Computer
Science, 2014.
Daniel Bernau, Philip-William Grassal, Jonas Robl, and Florian Kerschbaum. Assessing differentially private
deep learning with membership inference.arXiv preprint arXiv:1912.11328, 2019.
Franziska Boenisch, Christopher Mühl, Adam Dziedzic, Roy Rinberg, and Nicolas Papernot. Have it your
way: Individualized privacy assignment for dp-sgd.arXiv preprint arXiv:2303.17046, 2023.
Lucas Bourtoule, Varun Chandrasekaran, Christopher Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu
Zhang, David Lie, and Nicolas Papernot. Machine unlearning.arXiv preprint arXiv:1912.03817, 2019.
Zhiqi Bu, Jialin Mao, and Shiyun Xu. Scalable and efficient training of large convolutional neural networks
with differential privacy.arXiv preprint arXiv:2205.10683, 2022.
Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender
classification. In Conference on Fairness, Accountability, and Transparency, 2018.
Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating
and testing unintended memorization in neural networks. In28th USENIX Security Symposium, 2019a.
Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating
and testing unintended memorization in neural networks. InUSENIX Security Symposium, 2019b.
Nicholas Carlini, Matthew Jagielski, Chiyuan Zhang, Nicolas Papernot, Andreas Terzis, and Florian Tramer.
The privacy onion effect: Memorization is relative.Advances in Neural Information Processing Systems,
35:13263–13276, 2022.
Xiangyi Chen, Steven Z Wu, and Mingyi Hong. Understanding gradient clipping in private sgd: A geometric
perspective. Advances in Neural Information Processing Systems, 2020.
Soham De, Leonard Berrada, Jamie Hayes, Samuel L Smith, and Borja Balle. Unlocking high-accuracy
differentially private image classification through scale.arXiv preprint arXiv:2204.13650, 2022.
Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy.Foundations and
Trends® in Theoretical Computer Science, 2014.
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private
data analysis. InProceedings of the 3rd Conference on Theory of Cryptography, TCC ’06, pp. 265–284,
Berlin, Heidelberg, 2006. Springer.
12

Published in Transactions on Machine Learning Research (09/2023)
Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via
influence estimation. InAdvances in Neural Information Processing Systems 33, 2020.
Vitaly Feldman and Tijana Zrnic. Individual privacy accounting via a Renyi filter. InAdvances in Neural
Information Processing Systems, 2021.
Vitaly Feldman, Audra McMillan, and Kunal Talwar. Stronger privacy amplification by shuffling for rényi
and approximate differential privacy. InAnnual ACM-SIAM Symposium on Discrete Algorithms, 2023.
Antonio Ginart, Melody Guan, Gregory Valiant, and James Y Zou. Making ai forget you: Data deletion in
machine learning. InAdvances in Neural Information Processing Systems, 2019.
Aditya Golatkar, Alessandro Achille, Yu-Xiang Wang, Aaron Roth, Michael Kearns, and Stefano Soatto.
Mixed differential privacy in computer vision. InProceedings of IEEE Computer Society Conference on
Computer Vision and Pattern Recognition, 2022.
Sivakanth Gopi, Yin Tat Lee, and Lukas Wutschitz. Numerical composition of differential privacy. In
Advances in Neural Information Processing Systems, 2021.
Victor Petrén Bach Hansen, Atula Tejaswi Neerkaje, Ramit Sawhney, Lucie Flek, and Anders Søgaard. The
impact of differential privacy on group disparity mitigation.arXiv preprint arXiv:2203.02745, 2022.
Tatsunori Hashimoto, Megha Srivastava, Hongseok Namkoong, and Percy Liang. Fairness without demo-
graphics in repeated loss minimization. InInternational Conference on Machine Learning, pp. 1929–1938.
PMLR, 2018.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In
Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2016.
Matthew Jagielski, Jonathan Ullman, and Alina Oprea. Auditing differentially private machine learning:
How private is private sgd? InAdvances in Neural Information Processing Systems, 2020.
Ziheng Jiang, Chiyuan Zhang, Kunal Talwar, and Michael C Mozer. Characterizing structural regularities of
labeled data in overparameterized models. InInternational Conference on Machine Learning, pp. 5034–5044.
PMLR, 2021.
Zach Jorgensen, Ting Yu, and Graham Cormode. Conservative or liberal? personalized differential privacy.
In International Conference on Data Engineering (ICDE), 2015.
Daniel Kifer, Adam Smith, and Abhradeep Thakurta. Private convex empirical risk minimization and
high-dimensional regression. InProceedings of the 25th Annual Conference on Learning Theory, COLT ’12,
pp. 25.1–25.40, 2012.
Antti Koskela, Marlon Tobaben, and Antti Honkela. Individual privacy accounting with gaussian differential
privacy. arXiv preprint arXiv:2209.15596, 2022.
Antti Koskela, Mikko A Heikkilä, and Antti Honkela. Numerical accounting in the shuffle model of differential
privacy. Transactions on Machine Learning Research, 2023.
Alex Krizhevsky. Learning multiple layers of features from tiny images, 2009.
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 1998.
Mathias Lécuyer. Practical privacy filters and odometers with rényi differential privacy and applications to
differentially private deep learning.arXiv preprint arXiv:2103.01379, 2021.
Xuechen Li, Florian Tramèr, Percy Liang, and Tatsunori Hashimoto. Large language models can be
strong differentially private learners. InProceedings of the 10th International Conference on Learning
Representations, 2022.
13

Published in Transactions on Machine Learning Research (09/2023)
Katrina Ligett, Seth Neel, Aaron Roth, Bo Waggoner, and Steven Z Wu. Accuracy first: Selecting a differential
privacy level for accuracy constrained erm.Advances in Neural Information Processing Systems, 2017.
Andrew Lowy, Devansh Gupta, and Meisam Razaviyayn. Stochastic differentially private and fair learning.
arXiv preprint arXiv:2210.08781, 2022.
Harsh Mehta, Walid Krichene, Abhradeep Thakurta, Alexey Kurakin, and Ashok Cutkosky. Differentially
private image classification from features.arXiv preprint arXiv:2211.13403, 2022a.
Harsh Mehta, Abhradeep Thakurta, Alexey Kurakin, and Ashok Cutkosky. Large scale transfer learning for
differentially private image classification.arXiv preprint arXiv:, 2022b.
Sebastian Meiser. Approximate and probabilistic differential privacy definitions.Cryptology ePrint Archive,
2018.
Ilya Mironov. Rényi differential privacy. InProceedings of the 30th IEEE Computer Security Foundations
Symposium, 2017.
Ilya Mironov, Kunal Talwar, and Li Zhang. Rényi differential privacy of the sampled gaussian mechanism.
arXiv preprint arXiv:1908.10530, 2019.
Christopher Mühl and Franziska Boenisch. Personalized PATE: Differential privacy for machine learning
with individual privacy guarantees.arXiv preprint arXiv:2202.10517, 2022.
Milad Nasr, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, and Nicholas Carlini. Adversary in-
stantiation: Lower bounds for differentially private machine learning.arXiv preprint arXiv:2101.04535,
2021a.
Milad Nasr, Shuang Songi, Abhradeep Thakurta, Nicolas Papemoti, and Nicholas Carlin. Adversary
instantiation: Lower bounds for differentially private machine learning. In2021 IEEE Symposium on
Security and Privacy (SP), 2021b.
Frederik Noe, Rasmus Herskind, and Anders Søgaard. Exploring the unfairness of dp-sgd across settings.
arXiv preprint arXiv:2202.12058, 2022.
Nicolas Papernot, Martín Abadi, Ulfar Erlingsson, Ian Goodfellow, and Kunal Talwar. Semi-supervised
knowledge transfer for deep learning from private training data. InProceedings of the 5th International
Conference on Learning Representations, 2017.
Nicolas Papernot, Abhradeep Thakurta, Shuang Song, Steve Chien, and Ulfar Erlingsson. Tempered sigmoid
activations for deep learning with differential privacy.arXiv preprint arXiv:2007.14191, 2020.
Md Atiqur Rahman, Tanzila Rahman, Robert Laganiere, Noman Mohammed, and Yang Wang. Membership
inference attack against differentially private deep learning model.Transactions on Data Privacy, 2018.
Rachel Redberg and Yu-Xiang Wang. Privately publishable per-instance privacy. InAdvances in Neural
Information Processing Systems, 2021.
Alexandre Sablayrolles, Matthijs Douze, Yann Ollivier, Cordelia Schmid, and Hervé Jégou. White-box
vs black-box: Bayes optimal strategies for membership inference.International Conference on Machine
Learning, 2019.
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against
machine learning models. InIEEE Symposium on Security and Privacy (SP), 2017.
Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially
private updates. InProceedings of IEEE Global Conference on Signal and Information Processing, 2013.
Shuang Song, Thomas Steinke, Om Thakkar, and Abhradeep Thakurta. Evading the curse of dimensionality
in unconstrained private glms. InInternational Conference on Artificial Intelligence and Statistics, 2021.
14

Published in Transactions on Machine Learning Research (09/2023)
Vinith M Suriyakumar, Nicolas Papernot, Anna Goldenberg, and Marzyeh Ghassemi. Chasing your long tails:
Differentially private prediction in health care settings. InProceedings of ACM Conference on Fairness,
Accountability, and Transparency, 2021.
Shaowei Wang. Privacy amplification via shuffling: Unified, simplified, and tightened. arXiv preprint
arXiv:2304.05007, 2023.
Yu-Xiang Wang. Per-instance differential privacy.The Journal of Privacy and Confidentiality, 2019.
Yu-Xiang Wang, Borja Balle, and Shiva Prasad Kasiviswanathan. Subsampled rényi differential privacy
and analytical moments accountant. InProceedings of the 22nd International Conference on Artificial
Intelligence and Statistics, 2019.
Justin Whitehouse, Aaditya Ramdas, Ryan Rogers, and Zhiwei Steven Wu. Fully adaptive composition in
differential privacy.arXiv preprint arXiv:2203.05481, 2022.
Ashkan Yousefpour, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek,
John Nguyen, Sayan Ghosh, Akash Bharadwaj, Jessica Zhao, Graham Cormode, and Ilya Mironov. Opacus:
User-friendly differential privacy library in PyTorch.arXiv preprint arXiv:2109.12298, 2021a.
Ashkan Yousefpour, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek,
John Nguyen, Sayan Gosh, Akash Bharadwaj, Jessica Zhao, Graham Cormode, and Ilya Mironov. Opacus:
User-friendly differential privacy library in PyTorch.arXiv preprint arXiv:2109.12298, 2021b.
Da Yu, Huishuai Zhang, Wei Chen, Jian Yin, and Tie-Yan Liu. Large scale private learning via low-rank
reparametrization. In International Conference on Machine Learning, 2021.
DaYu, SaurabhNaik, ArtursBackurs, SivakanthGopi, HuseyinAInan, GautamKamath, JanardhanKulkarni,
Yin Tat Lee, Andre Manoel, Lukas Wutschitz, Sergey Yekhanin, and Huishuai Zhang. Differentially
private fine-tuning of language models. InProceedings of the 10th International Conference on Learning
Representations, 2022.
Zhifei Zhang, Yang Song, and Hairong Qi. Age progression/regression by conditional adversarial autoencoder.
In Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2017.
Yuqing Zhu, Xiang Yu, Manmohan Chandraker, and Yu-Xiang Wang. Private-knn: Practical differential
privacy for computer vision. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 11854–11862, 2020.
15

Published in Transactions on Machine Learning Research (09/2023)
A Proof of Theorem 3.1
Theorem 3.1. Let{θ1,...,θ t−1} be the observed models at stept. Suppose we run Algorithm 1 withK = 1
and without rounding, then Algorithm 1 satisfies(o(i)
α + log(1/δ)
α−1 ,δ )-output-specific individual DP for theith
example at A = (θ1,...,θ t−1,Ot), whereo(i)
α is the accumulated RDP at orderα andOt is the range ofAt.
Here we give the proof of Theorem 3.1. Let(A1,..., At−1) be a sequence of randomized algorithms and
(θ1,...,θ t−1) be some fixed outcomes, we define
ˆA(t)(θ1,...,θ t−1, D) = (A1(D),A2(θ1, D),..., At(θ1,...,θ t−1, D)).
Noting that the individual RDP parameters of each individual mechanism inˆA(t)(θ1,...,θ t−1, D) are constants.
Further let
A(t)(D) = (A1(D),A2(A1(D), D),..., At(A1(D),..., D))
be the adaptive composition. In Lemma A.1, we show an RDP bound onˆA(t) gives an output-specific DP
bound onA(t). We comment that the individual RDP parameters of each individual mechanism inA(t)(D)
are random variables. The composition of random privacy parameters requires additional care because the
standard composition theorem requires the privacy parameters to be constants (Feldman & Zrnic, 2021;
Lécuyer, 2021; Whitehouse et al., 2022).
Lemma A.1. Let A = (θ1,...,θ t−1,Ot)⊂O (t) whereθ1,...,θ t−1 are some arbitrary fixed outcomes and
O(t) is the domain ofA(t)(D) and ˆA(t)(D). If ˆA(t)(·) satisfies oα RDP at orderα, thenA(t)(D) satisfies
(oα + log(1/δ)
α−1 ,δ )-output-specific differential privacy atA.
Proof. For a given outcomeθ(t) = (θ1,θ 2,...,θ t−1,θt)∈ A, we haveP
[
A(t)(D) =θ(t)]
=
P
[
A(t−1)(D) =θ(t−1)
]
P
[
At(A1(D),..., At−1(D), D) =θt|A(t−1)(D) =θ(t−1)
]
, (6)
= P
[
A(t−1)(D) =θ(t−1)
]
P [At(θ1,...,θ t−1, D) =θt], (7)
by the product rule of conditional probability. Apply the product rule recurrently onP
[
A(t−1)(D) =θ(t−1)]
,
we haveP
[
A(t)(D) =θ(t)]
=
P
[
A(t−2)(D) =θ(t−2)
]
P [At−1(θ1,...,θ t−2, D) =θt−1] P [At(θ1,...,θ t−1, D) =θt], (8)
= P [A1(D) =θ1] P [A2(θ1, D) =θ2]... P [At(θ1,...,θ t−1, D) =θt], (9)
= P
[
ˆA(t)(θ1,...,θ t−1, D) =θ(t)
]
. (10)
In words,A(t) and ˆA(t) are identical inA. Therefore,A(t) satisfies (ε,δ )-DP at anyS⊂ A if ˆA(t) satisfies
(ε,δ )-DP. Converting the RDP bound onˆA(t)(D) into a(ε,δ )-DP bound with Lemma A.2 then completes
the proof.
Lemma A.2(Conversion from RDP to(ε,δ )-DP Mironov (2017)). IfA satisfies (α,ρ )-RDP, thenA satisfies
(ρ + log(1/δ)
α−1 ,δ )-DP for all0<δ < 1.
16

Published in Transactions on Machine Learning Research (09/2023)
Algorithm 3Differentially Private SGD with Individual Clipping
Input: Clipping thresholdC, noise varianceσ2, sampling probabilityp, number of stepsT.
for t = 0toT − 1 do
Call Algorithm 2 for individual privacy accounting and get estimates of individual gradient
norms{Z (i)}n
i=1.
Sample a minibatch of gradients{g(Ij )}|I|
j=1 with probabilityp , whereI is the sampled indices.
Clip gradients ¯g(Ij ) =clip(g(Ij ),Z (Ij )).
Update modelθt =θt−1−η(∑¯g(Ij ) +z), wherez∼N (0,σ 2I).
end for
B Individual Privacy Accounting with Individual Clipping
We setK >1 in Algorithm 2 to reduce the computational cost of individual privacy accounting (see Section 3
for details). In this case, the computed privacy costs are estimates of the exact ones. In Section 3.3 we
demonstrate the estimates are accurate. In this section, we give another design choice that slightly modifies
the original DP-SGD to give exact privacy accounting. More specifically, we clip the individual gradients with
the estimates of gradient norms{Z (i)} from Algorithm 2. We refer to this design choice asindividual clipping.
We give the implementation in Algorithm 3 and highlight the changes in bold font. We run experiments with
individual clipping and report the results in Appendix B.1 and B.2. The experimental setup is the same as
that in Section 4.
B.1 Individual Clipping Does Not Affect Accuracy
Algorithm 3 uses individual clipping thresholds to ensure the computed privacy parameters are strict privacy
guarantees. If the clipping thresholds are close to the actual gradient norms, then the clipped results are
close to those of using a single maximum clipping threshold. However, if the estimations of gradient norms
are not accurate, individual thresholds would clip more signal than using a single maximum threshold.
Table 3: Comparison between the test accuracy of using individual clipping thresholds and that of using a
single maximum clipping threshold. The maximumε is 7.8 for CIFAR-10 and 2.4 for MNIST.
CIFAR-10 MNIST
Individual 74.0 ( ±0.19) 97.17 ( ±0.12)
Maximum 74.1 ( ±0.24) 97.26 ( ±0.11)
We compare the accuracy of two different clipping methods in Table 3. The individual clipping thresholds
are updated once per epoch. We repeat the experiment four times with different random seeds. The results
suggest that using individual clipping thresholds in Algorithm 1 has a negligible effect on accuracy.
B.2 Individual Clipping Does Not Change The Observations
Here we show running DP-SGD with individual clipping does not change our observations in Section 4 and 5.
Privacy parameters have a strong correlation with individual training loss.In Figure 7, we show
privacy parameters computed with individual clipping are still positively correlated with training losses. The
Pearson correlation coefficient between privacy parameters and log losses is larger than0.9 for all datasets.
Groups are simultaneously underserved in both accuracy and privacyWe show our observation
in Section 5, i.e., low-accuracy groups have worse privacy parameters, still holds in Figure 8. We also make
a direct comparison with privacy parameters computed without individual clipping. We find that privacy
parameters computed with individual clipping are close to those computed without individual clipping. We
also find that the order of groups, sorted by the averageε, is exactly the same for both cases.
17

Published in Transactions on Machine Learning Research (09/2023)
Figure 7: Privacy parameters and final training losses. The experiments are run with individual clipping
(Algorithm 3). The Pearson correlation coefficient is computed between privacy parameters and log losses.
Ship Auto. Truck Frog Horse Air. Deer Dog Bird Cat
Subgroup Name
60
70
80
90
100Accuracy (in %)
CIFAR-10
T est Acc. (Without I.C.)
1 7 6 4 0 3 2 9 5 8
Subgroup Name
93
94
95
96
97
98
99
100Accuracy (in %)
MNIST
T est Acc. (Without I.C.)
Indian Black White Asian
Subgroup Name
80.0
82.5
85.0
87.5
90.0
92.5
95.0
97.5
100.0Accuracy (in %)
UTKFace-Gender
T est Acc. (Without I.C.)
Asian White Indian Black
Subgroup Name
65
70
75
80
85
90
95Accuracy (in %)
UTKFace-Age
T est Acc. (Without I.C.)
4.5
5.0
5.5
6.0
6.5
7.0
7.5
Average 
Privacy Parameter  (Without I.C.)
Privacy Parameter  (With I.C.)
 0.2
0.4
0.6
0.8
1.0
1.2
1.4
Average 
Privacy Parameter  (Without I.C.)
Privacy Parameter  (With I.C.)
1.8
2.0
2.2
2.4
2.6
2.8
3.0
3.2
Average 
Privacy Parameter  (Without I.C.)
Privacy Parameter  (With I.C.)
 1.75
2.00
2.25
2.50
2.75
3.00
3.25
3.50
Average 
Privacy Parameter  (Without I.C.)
Privacy Parameter  (With I.C.)
Figure 8: Test accuracy and privacy parameters computed with/without individual clipping (I.C.). Groups
with worse test accuracy also have worse privacy in general.
18

Published in Transactions on Machine Learning Research (09/2023)
0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 6.4 7.2 8.0
0
4000
8000
12000
16000
20000
24000
28000
32000
36000
40000Count
C=0.1M, test acc.=60.7, max i=7.5, min i=1.6
0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 6.4 7.2 8.0
0
3500
7000
10500
14000
17500
21000
24500
28000
31500
35000Count
C=0.3M, test acc.=64.9, max i=7.5, min i=1.0
0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 6.4 7.2 8.0
0
3000
6000
9000
12000
15000
18000
21000
24000
27000
30000Count
C=0.5M, test acc.=65.6, max i=7.5, min i=0.9
0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 6.4 7.2 8.0
0
1000
2000
3000
4000
5000
6000
7000
8000
9000
10000Count
C=1.5M, test acc.=63.2, max i=7.5, min i=0.5
Figure 9: Distributions of individual privacy parameters on CIFAR-10 with different maximum clipping
thresholds. The dashed line indicates the average of privacy parameters.
Ship Frog Auto. Truck Air. Horse Dog Bird Deer Cat
Subgroup Name
40
50
60
70
80Accuracy (in %)
 Clip=0.1 Median
Training Acc. (Max.-Min.=34.7%)
Test Acc. (Max.-Min.=34.3%)
Ship Auto. Frog TruckHorse Air. Dog Bird Deer Cat
Subgroup Name
40
50
60
70
80
90Accuracy (in %)
 Clip=0.3 Median
Training Acc. (Max.-Min.=40.7%)
Test Acc. (Max.-Min.=41.0%)
Ship Auto. Truck Frog Horse Air. Dog Bird Deer Cat
Subgroup Name
50
60
70
80
90Accuracy (in %)
 Clip=0.5 Median
Training Acc. (Max.-Min.=35.5%)
Test Acc. (Max.-Min.=36.5%)
Ship Auto. Truck Air. Frog Dog Horse Cat Deer Bird
Subgroup Name
50
60
70
80Accuracy (in %)
 Clip=1.5 Median
Training Acc. (Max.-Min.=33.9%)
Test Acc. (Max.-Min.=33.6%)
6.0
6.5
7.0
7.5
8.0
Average 
Privacy Parameter  (Max-Min=1.33)
 5.5
6.0
6.5
7.0
7.5
8.0
Average 
Privacy Parameter  (Max-Min=1.78)
5.5
6.0
6.5
7.0
7.5
Average 
Privacy Parameter  (Max-Min=1.87)
3.75
4.00
4.25
4.50
4.75
5.00
5.25
5.50
Average 
Privacy Parameter  (Max-Min=0.91)
Figure 10: Accuracy and average ε of different groups on CIFAR-10 with different maximum clipping
thresholds.
19

Published in Transactions on Machine Learning Research (09/2023)
C The Influence of Different Maximum Clipping Thresholds
The value of the maximum clipping thresholdC would affect individual privacy parameters. A large value
of C would increase the stratification in gradient norms but also increase the noise variance for a fixed
privacy budget. A small value ofC would suppress the stratification but also increase the gradient bias. Here
we run experiments with different values ofC on CIFAR-10. We use a small ResNet20 model in He et al.
(2016), which only has∼0.2M parameters, to reduce the computation cost. All batch normalization layers
are replaced with group normalization layers. LetM be the median of gradient norms at initialization, we
choose C from the list[0.1M, 0.3M, 0.5M, 1.5M].
The histograms of individual privacy parameters are in Figure 9. In terms of accuracy, using clipping
thresholds near the median gives better test accuracy. In terms of privacy, using smaller clipping thresholds
increases privacy parameters in general. The number of datapoints that reaches the worst privacy decreases
with the value ofC. WhenC = 0.1M, nearly 70% datapoints reach the worst privacy parameter while only
∼2% datapoints reach the worst parameter whenC = 1.5M.
The correlation between accuracy and privacy is in Figure 10. The disparity in averageε is clear for all choices
ofC. Another important observation is that when decreasingC, the privacy parameters of underserved groups
increase quicker than other groups. When changingC = 1.5M to 0.5M, the averageε of ‘Cat’ increases from
4.8 to 7.4, almost reaching the worst-case bound. In comparison, the increment inε of the ‘Ship’ class is only
1.3 (from 4.2 to 5.5).
D Privacy Parameters Reflect Empirical Privacy Risks in Non-Private Learning
We run membership inference (MI) attacks to verify whether examples with larger privacy parameters have
higher privacy risks in practice. We use a simple loss-threshold attack that predicts an example is a member
if its loss value is smaller than a prespecified threshold (Sablayrolles et al., 2019). Previous works show that
even large privacy parameters are sufficient to defend against such attacks (Carlini et al., 2019b; Yu et al.,
2021). In order to better observe the difference in privacy risks, we also include models trained without
differential privacy as target models. For each data subgroup, we use its whole test set and a random subset
of the training set so the numbers of training and test loss values are balanced. We further split the data into
two subsets evenly to find the optimal threshold on one and report the success rate on another.
Ship Auto. Truck Frog Horse Air. Dog Deer Bird Cat
Subgroup Name
2
3
4
5
6
7Average 
CIFAR-10
Average Epsilon
Indian Black White Asian
Subgroup Name
1.00
1.25
1.50
1.75
2.00
2.25
2.50
2.75
3.00Average 
UTKFace-Gender
Average Epsilon
50
55
60
65
70
75
80
85
90
95
MI Success Rate
61.4
59.2
62.6
66.4 65.8
67.9
71.3 70.9
73.1
79.7
52.0 52.4 51.4 51.7 52.5 52.1 52.2 52.4 52.8 52.8
Success Rate (without DP)
Success Rate (with DP)
50.0
52.5
55.0
57.5
60.0
62.5
65.0
67.5
70.0
MI Success Rate
57.1
57.8 57.8
60.4
50.7
51.8
50.4
51.2
Success Rate (without DP)
Success Rate (with DP)
Figure 11: Averageϵ and membership inference success rates on different subgroups.
The results on CIFAR-10 and UTKFace-Gender are in Figure 11. The subgroups are sorted based on their
average ϵ. When the models are trained with DP, all attack success rates are close to random guessing
(50%). Although the attack we use can not show the disparity in this case, we note that there are more
powerful attacks whose success rates are closer to the lower bound that DP offers (Nasr et al., 2021b). On
the other hand, the difference in privacy risks is clear when models are trained without DP. On CIFAR-10,
the MI success rate is 79.7% on the Cat class (which has the worst averageϵ when trained with DP) while is
20

Published in Transactions on Machine Learning Research (09/2023)
only 61.4% on the Ship class (which has the best averageϵ). These results suggest that theϵ values reflect
empirical privacy risks which could vary significantly in different subgroups.
E The Correlation Between Privacy Parameters and Loss Holds under Different
Clipping Thresholds
In Section 4.2, we show that there is a positive logarithmic correlation between privacy parameters and
training loss. In this section, we run experiments on CIFAR-10 withC = 5 andC = 10 to show the correlation
still holds under different clipping thresholds. The experiment setup, except the value ofC, is the same as
Section 4.2. The results are in Figure 12. Although changing the clipping threshold changes the slope and
intercept, the logarithmic correlation is still strong. The Pearson correlation coefficients are0.89 and 0.9 for
C = 5 and C = 10, respectively.
Figure 12: Privacy parameters and final training losses. Each point shows the final training loss and privacy
parameter of one example. Pearson’sr is computed between privacy parameters and log loss values.
F Individual Privacy Accounting in More Settings
In this section, we study individual privacy accounting in more experimental settings. The dataset in this
section is CIFAR-10. We first study the influence of loss function. We replace the cross-entropy loss with the
multi-class hinge loss. Other settings are the same as those in Section 4. The results are in Figure 13. We
also study the influence of model architectures. We replace WRN16-4 with the two-layer convolutional neural
networks in Papernot et al. (2020) and still use the cross-entropy loss. The learning rate is set as1.0 and
other settings are the same as those in Section 4. The results are in Figure 14.
Our main observations in the main text still hold in the new settings. The Pearson’s correlation coefficients
between the estimated and actual privacy parameters are larger than0.99 in both cases. Examples that
are underserved by the model also suffer from higher privacy costs. Although the main observations do not
change, we observe some differences in the privacy parameters. When using the two-layer neural network
instead of WRN16-4, the privacy parameters become larger. For instance, the averageε of Cat increases
from 7.3 to 7.7.
21

Published in Transactions on Machine Learning Research (09/2023)
0 2 4 6 8
Actual 
0
2
4
6
8Estimated 
Pearson's r = 0.999
Avg. | | = 0.06
Max | | = 0.37
CIFAR-10: = 1
y = x
y = 7.9
(a) Estimated ε versus exact ε.
Ship Auto. Truck Frog Horse Air. Deer Bird Dog Cat
Subgroup Name
60
70
80
90Accuracy (in %)
CIFAR-10
T est Acc.
4.0
4.5
5.0
5.5
6.0
6.5
7.0
Average 
Privacy Parameter 
 (b) Accuracy and group-wise ε.
Figure 13: The results of using multi-class hinge loss instead of cross-entropy loss.
0 2 4 6 8
Actual 
0
2
4
6
8Estimated 
Pearson's r = 0.999
Avg. | | = 0.05
Max | | = 0.28
CIFAR-10: = 1
y = x
y = 7.9
(a) Estimated ε versus exact ε.
Auto. Ship Truck Horse Frog Air. Dog Deer Bird Cat
Subgroup Name
50
60
70
80
90Accuracy (in %)
CIFAR-10
T est Acc.
5.5
6.0
6.5
7.0
7.5
8.0
Average 
Privacy Parameter 
 (b) Accuracy and group-wise ε.
Figure 14: The results of using the small convolutional neural network in Papernot et al. (2020).
22