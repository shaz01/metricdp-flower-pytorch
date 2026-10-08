Towards Understanding the Inﬂuence of Individual Clients in Federated Learning
Yihao Xue 1, Chaoyue Niu 1, Zhenzhe Zheng 1,
Shaojie Tang 2, Chengfei Lyu 3, Fan Wu 1, and Guihai Chen 1
1 Shanghai Jiao Tong University,
2 The University of Texas at Dallas,
3 Alibaba Group
yh xue@outlook.com, rvince@sjtu.edu.cn, zhengzhenzhe@sjtu.edu.cn,
shaojie.tang@utdallas.edu, chengfei.lcf@alibaba-inc.com, fwu@cs.sjtu.edu.cn, gchen@cs.sjtu.edu.cn
Abstract
Federated learning allows mobile clients to jointly train a
global model without sending their private data to a central
server. Extensive works have studied the performance guar-
antee of the global model, however, it is still unclear how
each individual client inﬂuences the collaborative training
process. In this work, we deﬁned a new notion, called Fed-
Inﬂuence, to quantify this inﬂuence over the model parame-
ters, and proposed an effective and efﬁcient algorithm to es-
timate this metric. In particular, our design satisﬁes several
desirable properties: (1) it requires neither retraining nor re-
tracing, adding only linear computational overhead to clients
and the server; (2) it strictly maintains the tenets of federated
learning, without revealing any client’s local private data; and
(3) it works well on both convex and non-convex loss func-
tions, and does not require the ﬁnal model to be optimal.
Empirical results on a synthetic dataset and the FEMNIST
dataset demonstrate that our estimation method can approxi-
mate Fed-Inﬂuence with small bias. Further, we show an ap-
plication of Fed-Inﬂuence in model debugging.
Introduction
Federated learning ingeniously leverages a large amount of
valuable data in a distributed manner, while mitigating sys-
temic privacy risks (McMahan et al. 2017; Kairouz et al.
2019). In such a setting, the training data are stored on multi-
ple decentralized clients, who share a common global model
and train it collaboratively. There is a central server that
orchestrates the whole training process. In each round, the
server collects local models from some eligible clients and
use them to update the global model.
In this paper, we consider a signiﬁcantly important but
less considered problem in federated learning: how does
each client inﬂuence the global model? Finding the answer
to this question is meaningful in several aspects. On one
hand, it helps us understand where the model comes from
by explaining the prediction of the current global model
in terms of clients. On the other hand, it provides quanti-
tative insights into the roles of individual clients in feder-
ated learning, and informs us whether the existence of a
certain client beneﬁts the global model. This is critical to
Copyright © 2021, Association for the Advancement of Artiﬁcial
Intelligence (www.aaai.org). All rights reserved.
a fair credit/reward allocation, and, more importantly, de-
bugging for federated learning. Thus, a ﬁne-grained under-
standing of clients’ inﬂuence further facilitates the exclusion
of negative-inﬂuence clients or dynamically requires them
to check their local data, thereby improving model perfor-
mance. All of these are important to the explanation and ro-
bustness of federated learning, and also help sustain long-
term user participation.
There already exists a classical statistics notion of “inﬂu-
ence” in centralized learning, which evaluates the effect that
the absence of an individual sample has on a model. To mea-
sure this inﬂuence, one should conduct a leave-one-out test
(Cook 1977): retrain the model over the training set with one
certain sample removed, and compare this model with that
trained on the full dataset. A notion from robust statistics,
called “inﬂuence function” (Jaeckel 1972; Hampel 1974;
Cook and Weisberg 1980), was introduced to avoid retrain-
ing by measuring the change in the model caused by slightly
changing the weight of one sample and using quadratic ap-
proximation combined with a Newton step (Cook and Weis-
berg 1982). Koh and Liang (2017) leveraged inﬂuence func-
tion in modern machine learning settings and developed an
efﬁcient and simple implementation using second-order op-
timization techniques. Hara, Nitanda, and Maehara (2019)
went beyond convexity and optimality, which are two impor-
tant assumptions in (Koh and Liang 2017). Koh et al. (2019)
studied the effects of removing a group of data points, which
is analogous to removing a client that holds a subset of train-
ing data in federated learning, except that their study is still
in a centralized setting. Khanna et al. (2019) applied Fisher
kernels along with sequential Bayesian quadrature to iden-
tify a subset of training examples that are most responsible
for a given set of predictions and recovered (Koh and Liang
2017) as a special case.
The existing works above focused on centralized learning,
and we are the ﬁrst to consider a similar problem, the inﬂu-
ence of individual clients, under a brand new framework,
namely federated learning. There are some essential differ-
ences between centralized learning and federated learning
which raise several design challenges: (1) The server in cen-
tralized learning, as the inﬂuence evaluator, has the full con-
trol over the considered sampling data, while in federated
learning, the server would not be able to access clients’ raw
TheThi rty-Fif thAAA ICon fe renceon A rtif icial Intellig ence(AAAI-21)
10560

data because of the privacy requirement; (2) the clients in
federated learning may not always be available, due to the
unreliable network connection. This implies that the server
cannot communicate with a certain client at any desired
time; and (3) the computing resources of mobile clients are
limited, for which reason we should not bring too many ad-
ditional computational burdens to clients when measuring
their inﬂuence.
Due to the ﬁrst difference, sample-level inﬂuence in cen-
tralized learning cannot be applied to measuring the inﬂu-
ence of individual clients in federated learning. In this work,
we turn to client-level inﬂuence measurement by investigat-
ing the effect of removing a client. In addition, works in cen-
tralized learning mainly focus on the inﬂuence on a model’s
testing loss, while we consider the inﬂuence on the parame-
ter of the global model for the following two reasons: (1) In
centralized learning, to cut down the time complexity, some
techniques can be applied to obtain inﬂuence on loss without
computing that on parameter (Hara, Nitanda, and Maehara
2019). However, in federated learning, computing inﬂuence
on parameter is unavoidable because of the second and third
differences mentioned earlier. Please refer to the supplement
1 for detailed reasoning; and (2) inﬂuence on parameter is
more fundamental and powerful than that on loss. With the
knowledge of inﬂuence on parameter, we can easily derive
the inﬂuence of individual clients in terms of various metrics
for evaluating models, such as loss, accuracy, precision, etc.
Our contributions. (1) To the best of our knowledge, we
are the ﬁrst to consider client-level inﬂuence in federated
learning; (2) we propose a basic estimator for individual
clients’ inﬂuence on model parameter by leveraging the re-
lationship between the global models in two consecutive
communication rounds. Guided by the error analysis, we ex-
tend the basic design to support both convex and non-convex
loss functions. We also develop an efﬁcient implementation,
bringing only slight communication and computation over-
head to the server and the clients (in fact no extra compu-
tation overhead for the client); and (3) empirical studies on
a synthetic dataset and the FEMNIST dataset (Caldas et al.
2018) demonstrate the effectiveness of our method. The esti-
mation error observed in experiments indicates both the ne-
cessity and accuracy of our method. Based on the inﬂuence
on parameter, we further derive the inﬂuence on model per-
formance and observe that it was well estimated. In particu-
lar, the Pearson correlation coefﬁcient between the estimated
inﬂuence on loss and the ground truth achieves 0.62 in the
most difﬁcult setting. We also leverage inﬂuence on model
performance for client valuation and client cleansing.
Problem Formulation
Federated Learning
We ﬁrst introduce some necessary notations. Let C denote
the set of all the clients, and Dk denote the local dataset of
clientk2C withnk samples. The setD =S
k2CDk is the
full training set. For any set of clients C0, we use N(C0) =
1The supplement is available from https://drive.google.com/ﬁle
/d/1zFefHCAYiv5DPJ6nDVjgLUeO4yOoAFlV/view?usp=sharing.
P
k2C0nk to denote the total size of these clients’ datasets.
L(w;z ) denotes the loss function over a model w and a
samplez. In addition,L(w;Dk) = 1
nk
P
z2DkL(w;z ) de-
notes the empirical loss over a model w and a datasetDk.
Then, we consider the following optimization task of feder-
ated learning:
min
w2Rp
(
L(w;D) =
X
k2C
nk
N(C)L(w;Dk)
)
; (1)
where the global loss functionL(w;D) is the weighted av-
erage of the local functionsL(w;Dk) with the weight pro-
portioning to the size of each client’s local dataset. In this
work, we consider a standard algorithm, federated averaging
(FedAvg) (McMahan et al. 2017), to solve the optimization
problem in (1). Although there are some other variants, such
as FedBoost (Hamer, Mohri, and Suresh 2020), FedNova
(Wang et al. 2020), FetchSGD (Rothchild et al. 2020), Fed-
Prox (Li et al. 2020a), and SCAFFOLD (Karimireddy et al.
2019), FedAvg is the ﬁrst and the most widely used one. As a
result, we see FedAvg as our basic block, which executes as
follows. In the initial stage, the server randomly initializes a
global model w0. Then, the training process is orchestrated
by repeating the following two steps within each communi-
cation roundt from 1 toT :
• Local training. The server selects a random set of clients
Ct as participants in this round. Each participant k2C t
then downloads from the server the latest global model
wt 1, i.e., the output model from the previous roundt 1.
Then, the client k performs local updates for each local
iterationi from 1 tom:
wk
t;i wk
t;i 1 rwL
 
wk
t;i 1;Dk
; (2)
with the starting local model wk
t;0 initialized as wt 1. In
addition, denotes the learning rate.
• Model aggregation. Participants in round t upload their
updated local models. The server aggregates (takes a
weighted average of) the local models to a new global
model wt:
wt 
X
k2Ct
nk
N(Ct) wk
t;m; (3)
where the weight of clientk is the size of her datasetnk.
Fed-Inﬂuence
To express the client-level inﬂuence on federated learn-
ing clearly, we introduce a new notation wt (C0!C0nfcg),
which represents the aggregated model in round t with
a client set C0 replaced with C0nfcg, where C0 2
fC1;C2;C3;:::; CT;Cg. For example, w10 (C5!C 5nfcg)
represents the resulting model we get at the end of round
10 if we remove the client c in round 5. One special case is
thatC0 =Ct, i.e.,
wt (Ct!C tnfcg) =
X
k2Ctnfcg
nk
N (Ctnfcg) wk
t;m: (4)
Another special case is wt (C!Cnf cg), where we perma-
nently remove the clientc, i.e., replaceCt withCtnfcg for all
the roundst2 [T ].
10561

With these notations, we next give the following deﬁnition
of Fed-Inﬂuence on Parameter (FIP).
Deﬁnition 1. We refer to the change in model parameters
due to removing a client c fromC as Fed-Inﬂuence on Pa-
rameter (FIP) of clientc, denoted by c;
t :
 c;
t
def
= wt (C!Cnf cg)  wt: (5)
Based on FIP, we can extend the notion to measure the
inﬂuence on model performance, such as the metrics of ac-
curacy, cross-entropy loss, precision, recall, mean squared
error (MSE), and etc. We can derive the inﬂuence on any of
these metrics from FIP. SupposeF is the loss function for a
certain metric over a test setDtest, we can compute
F
 
wt + c;
t ;Dtest

 F (wt;Dtest) (6)
as the Fed-Inﬂuence over the metric. In Sections and , we
will focus on two most widely used metrics, loss and accu-
racy, which corresponds to Fed-Inﬂuence on Loss (FIL) and
Fed-Inﬂuence on Accuracy (FIA), respectively.
The exact value of FIP for a client can only be obtained
by conducting leave-one-out test: retrain the model by re-
moving the client, and compare the retrained model with
the model trained on the full client set. However, it is pro-
hibitively inefﬁcient to rerun the whole federated learning
process, especially when we intend to measure the inﬂuence
of each client, implying the number of rerunning being the
total number of clients.
Basic Estimator
We now derive an estimator for FIP c;
t to avoid retraining.
We start by rewriting the expression of c;
t as follows
 c;
t
def
= wt (C!Cnf cg)  wt
= wt (C!Cnf cg)  wt (Ct!C tnfcg)
+ wt (Ct!C tnfcg)  wt
=
X
k2Ctnfcg
nk
N(Ctnfcg)
 
wk
t;m (C!Cnf cg)  wk
t;m

| {z }
local sequential inﬂuence
| {z }
sequential inﬂuence
+ wt (Ct!C tnfcg)  wt| {z }
combinatorial inﬂuence
(7)
Equation (7) shows that c;
t comprises three parts: (1) Lo-
cal sequential inﬂuence: the inﬂuence that removing clientc
from the client set C on the local model of any other par-
ticipating client k 2 Ctnfcg in round t. We regard it as
“sequential” inﬂuence because it can be derived from c;
t 1 ,
the inﬂuence in the previous round; (2)Sequential inﬂuence:
the weighted average of local sequential inﬂuence; and (3)
Combinatorial inﬂuence: the inﬂuence of removingc merely
fromCt in the roundt. The combinatorial inﬂuence is “com-
binatorial” as it is independent of c;
t 1 .
We next dissect how to compute local sequential inﬂu-
ence and combinatorial inﬂuence. The combinatorial one
can be easily obtained using (3) and (4). The local sequen-
tial inﬂuence is more complicated. Given that the reasons
behind the difference between wk
t;m (C!Cnf cg) and wk
t;m
is that they are locally updated from different initial models,
wt 1 (C!Cnf cg) and wt 1, respectively. We take a look
at the ﬁrst local iteration at round t. we estimate the term
by applying ﬁrst-order Taylor approximation and the chain
rule:
wk
t;m (C!Cnf cg)  wk
t;m
 @wk
t;m
@wk
t;m 1
@wk
t;m 1
@wk
t;m 2
::: @wk
t;1
@wk
t;0
wk
t;0 (8)
where wk
t;0 = wk
t;0 (C!Cnf cg) wk
t;0 = c;
t 1 . Accord-
ing to the update rule in Equation 2, with the assumption that
L(w;Dk) is twice differentiable, we obtain
@wk
t;i
@wk
t;i 1
= I Hk
t;i 1; (9)
where Hk
t;i
def
=r2
wL(wk
t;i;Dk). By combining Equations 7,
8 and 9, we get an estimator of c;
t , denoted by c
t
 c
t
def
= M c
t  c
t 1 + wt (Ct!C tnfcg)  wt; (10)
where
M c
t
def
=
X
k2Ctnfcg
nk
N(Ctnfcg)
m 1Y
i=0
(I Hk
t;i): (11)
By recalling that the initial modelw0 is randomly initialized
by the server, we have c
0 = 0. Then the estimator c
t can
be computed iteratively using Equation 10.
We ﬁnally take a close look at the relation between the
estimation error and the iterationt. We give a uniform bound
on the error in both convex and non-convex cases under the
following assumptions.
Assumption 1. There exists and  such thatI 4r2L 4
I.
Assumption 2. The norm of c;
t is bounded byC, fort2
[T ] andc2C .
Note that in Assumption 1, there is no constraint on the
values of and  except , which means the loss func-
tion is not necessarily convex. Next we give Theorem 1, the
proof of which is provided in the supplement.
Theorem 1. With Assumptions 1 and 2, the error of the es-
timator is bounded by
 c
t
def
=k c;
t   c
t k 1 t
1  o(C); 8t> 0; (12)
whereo is the little-o notation, and =m; = maxfj1 
j;j1 jg.
From (12), we can observe that the bound is in the format
of the sum of geometric series. An intuitive explanation is
that each time we use (10), the error in the previous round is
scaled by M c
t , and added a newly introduced error in this
round, similar to summing a geometric series. Finally, there
are three different cases depending on the relation between
 and 1:
10562

• Case 1 ( < 1): In this case,  > 0 and 0 <  < 2
,
where the loss function is strongly-convex and the learn-
ing rate is small enough. This is the most ideal case, where
the bound can be further scaled to 1
1 o(C), which is in-
dependent oft.
• Case 2 ( = 1): In this case, either  = 0 and 2
 or
 0 and = 2
. The former situation is more common,
with a convex loss function and an appropriate learning
rate. Then the error iso(C)t, linear witht.
• Case 3 ( > 1): In this case,  < 0 or  > 2
, where
either the loss function is non-convex or the learning rate
is too large. Then the error bound is exponential with t,
making estimation ineffective.
Improving Robustness and Efﬁciency
In this section, we improve the basic estimator from two as-
pects: one is to improve the robustness of the method in non-
convex case, and the other is to cut down the high cost due
to computing Hessian matrix.
Layer-Wise Examination and Truncation
The analysis in Section reveals that the basic estimator can
have a large error when the loss function is non-convex. The
non-convex case is quite common in federated learning for
deep learning tasks (Yu, Yang, and Zhu 2019; Haddadpour
et al. 2019). We thus propose layer-wise examination and
truncation (LWET for short).
Truncation. Our primary goal is to avoid an exponen-
tial error in Case 3, which results from the estimation
of sequential inﬂuence. We consider a counterpart, where
the sequential inﬂuence is completely omitted, i.e.,  c
t =
wt (Ct!C tnfcg)  wt (we call it the truncated estimator
for simplicity), and ﬁnd that the error becomes independent
oft, as shown in the following theorem.
Theorem 2. With the sequential inﬂuence omitted, we get
the bound of error as follows:
 c
t C +o(C):
Layer-wise Operation. However, after truncation, a new
problem arises: the truncated estimator will always be 0
at round t as long as c is not one of the participants. For
example, in a setting where 5 clients are selected each
round with 100 clients in total, there will be 95 clients
with FIP being 0 each round, which does not make sense.
To ensure accuracy while retaining as much information as
possible, it is necessary to ﬁnd a happy medium between
the basic estimator and the truncated estimator, which in-
stead partially omits the sequential inﬂuence. To achieve
this, we ﬁrst introduce layer-wise operation. We calculate
only parts of the Hessian matrix, with the interaction be-
tween different layers ignored. We take a convolutional neu-
ral network (CNN) for example. Supposing that the pa-
rameter w is composed of w(j), j = 1 ; 2;:::; 8, corre-
sponding to conv-layer 1, bias of conv-layer 1, conv-layer
2, bias of conv-layer 2, dense-layer 1, bias of dense-layer
1, dense-layer 2, bias of dense-layer 2, respectively, i.e.,
0 50 100
Round
0.0
0.1
0.2Layer-wise error
layer 1
layer 2
layer 3
layer 4
layer 5
layer 6
layer 7
layer 8
(a)
0 50 100
Round
0.0
0.1
0.2
0.3Norm of layer-wise FIP
150 200
0
1
2
1e11 (b)
Figure 1: The experimental result on a toy model (setting 2
in Table 2), a CNN without activation function. (a) The error
and (b) the norm of estimated FIP both have a tendency to
increase exponentially on layer 5, the ﬁrst fully-connected
layer. Inset: the norm of estimated FIP from round 135 to
200.
w = [ wT
(1); wT
(2); wT
(3); wT
(4); wT
(5); wT
(6); wT
(7); wT
(8)]T . For
each layer j, we seeL(w) as function of w(j) denoted by
L(j)(w(j)). Rather than computing the entire Hessian matrix
H =r2
wL(w), we only calculate H(j) =r2
w(j)L(j)(w(j))
for each j, some smaller matrices on the diagonal of H.
Then the estimator in each layerj is updated independently:
 c
t;(j) M c
t;(j) c
t 1;(j) + wt;(j) (Ct!C tnfcg)  wt;(j);
where M c
t;(j) =P
k2Ctnfcg
nk
N(Ctnfcg)
Qm 1
i=0 (I Hk
t;i;(j)).
We can examine separately the property of the loss function
L(j)(w(j)) in each layer j, and use a layer-wise truncated
estimator shown below only for layers in Case 3:
 c
t;(j) wt;(j) (Ct!C tnfcg)  wt;(j): (13)
Then we put together layer-wise FIPs to get the complete
FIP, i.e., c
t = [( c
t;(1))T; ( c
t;(2))T::: ]T .
Because the convexity and continuity ofL(j)(w(j)) vary
among different layers, there are layers in Case 1 or Case
2 which still remain the sequential inﬂuence. Therefore the
complete FIP is non-zero and contains much information
even when c =2 Ct. In an experiment conducted on a toy
model (CNN 2 in Table 2), we found the exponential error
only exists in the ﬁrst fully-connected layer, which is the
only layer where we need to apply a layer-wise truncated
estimator, as shown in Fig. 1(a).
Examination. The next problem is how we can examine
the property of loss function in each layer? We propose a
mechanism which does not require any prior knowledge: in
particular, we comparekM c
t;(j) c
t 1;(j)k withk c
t 1;(j)k. If
kM c
t;(j) c
t 1;(j)k is larger at a certain roundr, then, in all of
the following rounds, i.e., for all t r, a layer-wise trun-
cated estimator in Equation 13 will be used in layerj.
This mechanism is aimed at examining a sufﬁcient (but
not necessary) condition for  >1. Please refer to the sup-
plement for the reason. In addition, this examination also
avoids the overﬂow of the estimator itself. The upper bound
of  c
t 1 is also exponential with t in Case 3, which means
10563

there is always a risk for it to overﬂow as t increases. This
phenomenon can be observed in the experiment with the
aforementioned toy model. As shown in Fig. 1(b), the norm
of estimated FIP on the ﬁrst fully-connected layer increases
sharply around the 125-th round, and then achieves an order
of 1011 at round 200, indicating the failure of the algorithm.
Combining the three strategies, we get LWET. Because
real-world applications rarely satisfy the Identically and In-
dependently Distributed (IID) assumption and are likely
to be non-IID in many ways (Zhao et al. 2018; Li et al.
2020a,b; Yan et al. 2020), where clients vary in the data
distribution and the property of the local loss function,
a more ﬁne-grained version of LWET is recommended
in these cases. We can examine the relationship between
k
Qm 1
i=0 (I Hk
t;i;(j))

 c
t;(j)k andk c
t;(j)k. If the former
is larger at one round, then we drop the local sequential inﬂu-
ence onk from that round on. Please refer to this algorithm
in the supplement.
Low-Cost Hessian Approximation
Fisher information. Because the cross entropy loss is
a negative log-likelihood, it is not difﬁcult to obtain
that Ez2D0

rwL(w;z )rwL(w;z )T
, where w is the
true parameter (i.e., the model distribution under w
equals to the underlying distribution), is at the form of
Fisher information. And according to one of the alter-
native deﬁnition of Fisher information (Friedman, Hastie,
and Tibshirani 2001; Ly et al. 2017), it can also be
written as Ez2D0 [r2
wL(w;z )]. This means we can use
rwL(w;z )rwL(w;z )T as an asymptotically unbiased es-
timation ofr2
wL(w;z ), since w gradually converges to w
during the training.
Leveraging the fact that clients holds their own datasets,
we let each client randomly select a given number, denoted
byNs, of gradients and use the empirical expectation of the
outer product as an approximation to the Hessian, denoted
by ~Hk
t;i;(j):
Hk
t;i;(j) ~Hk
t;i;(j)
def
= 1
Ns
X
z2S k
t;i
g(wk
t;i;(j);z )g(wk
t;i;(j);z )T
where Sk
t;i is the set of Ns samples randomly selected
by client k at the i-th local iteration in round t, and
g(wk
t;i;(j);z ) =rw(j)L(j)(wk
t;i;(j);z ).
Recursive computation. Simply introducing Fisher infor-
mation cannot help cut down the cost, because the outer
product is O(p2), and the O(p2) matrix-vector or O(p3)
matrix-matrix multiplication still exists, where p is the
size of the model. However, we can actually circumvent
these needless calculations. With Hk
t;i;(j) approximated by
~Hk
t;i;(j), the estimator of local sequential inﬂuence is
0
@
m 1Y
i=0
0
@I  1
Ns
X
z2S k
t;i
g(wk
t;i;(j);z )g(wk
t;i;(j);z )T
1
A
1
A c
t 1;(j):
(14)
Server Client Comm.
Naive 1 K2p2 m(np2 +p3) p2
Naive 2 K2mp2 mnp2 mp2
Our method K2mNsp 0 mNsp
Table 1: Extra time complexity for the server, extra time
complexity for each client, and extra communication com-
plexity for each client. n is the size of the local dataset and
K =jCj. In a naive implementation, operations where the
Hessian matrix is involved, can be taken either on the clients
or the server, corresponding to naive implementations 1 and
2, respectively. For naive implementation 1, each clientk has
to computeQm 1
i=0 (I Hk
t;i) locally, which requires matrix-
matrix multiplications, and upload the result to the server.
For naive implementation 2, each client k uploads Hk
t;i for
alli, and then the server computes M c
t  c
t 1, which can be
implemented with only matrix-vector multiplications.
Instead of ﬁrst calculating the cumprod on the left and then
multiplying it by  c
t 1;(j), the linear-cost method is based
on a recursive computation: we initialize a vector k
(j) by
k
(j)  c
t 1;(j), and then we updatek
(j) using Equation 15
repeatedly fori from 0 tom  1:
k
(j) k
(j)  
Ns
X
z2S k
t;i
g(wk
t;i;(j);z )

g(wk
t;i;(j);z )Tk
(j)

:
(15)
k
(j) produced in the ﬁnal iteration is the value of Equa-
tion 14. Note that this method requires the computation to
take place on the server, because only the server has c
t 1;(j),
and the clients need to do nothing other than randomly select
and upload a certain number of local gradients. We show the
efﬁciency of our method by comparing it with naive imple-
mentations in Table 1.
Fundamental Experiments
In this section, we demonstrate two properties of our
method: (1) LWET plays a vital role; and (2) Hessian ap-
proximation causes only a slight drop in the accuracy. Ex-
periments are conducted on 64bit Ubuntu 18.04 LTS with
four Intel i9-9900K CPUs and two NVIDIA RTX-2080TI
GPUs, 200GB storage. We take “Leaf” (Caldas et al. 2018),
a benchmarking framework for federated learning based on
tensorﬂow. We evaluated our method on three settings, as
shown in Table 2. We used the softmax function at the out-
put layer and adopted the cross entropy as the loss function.
In setting 1, the loss function is convex but not strongly con-
vex, and therefore it is in Case 2 ( = 1 ). In setting 2,
although the toy model has no activation function, which
makes it equivalent to a single-layer perceptron with convex
loss function, results show that it is still in Case 3 ( > 1)
because the learning rate is too large. And in setting 3, the
loss function is non-convex and is therefore in Case 3, too.
10564

Model Dataset Distribution  jCj jC tj m T N s
Setting 1 LogReg Synthetic Non-IID, Unbalanced 0.003 1000 10 5 1000 50
Setting 2 CNN 1 FEMNIST IID, Balanced 0.03 50 5 2 500 50
Setting 3 CNN 2 FEMNIST Non-IID, Unbalanced 0.02 100 10 2 2000 50
Table 2: Detailed conﬁguration of the three different settings. The two datasets are described in Caldas et al. (2018). The logistic
regression model is the original one in “Leaf”. We made a little adjustment to the original CNN to create models with certain
properties and scales; see details in the supplement.
0 200 400
Round
0.0
0.2
0.4
0.6
0.8Distance
LWET & H.A.
Basic estimator
Only LWET
Figure 2: The distance between exact and estimated FIP in
setting 2.
Inﬂuence on Parameter
We use four different methods to obtain  c
t : (1) the basic
estimator, (2) the estimator with only LWET, (3) the esti-
mator with only Hessian approximation, and (4) the estima-
tor with both LWET and Hessian approximation. We do not
demonstrate results of all methods in each setting. In set-
ting 1, we had tested all of the four methods, but found that
the result produced with LWET is completely the same with
that produced without LWET, which further validates that
setting 1 is in Case 2. Therefore we only show two different
results, the result based on exact Hessian and that based on
approximated Hessian. In setting 2, we do not demonstrate
the method with only Hessian approximation, because the
basic estimator has already caused a terrible error, and the
introduction of Hessian approximation will, no doubt, cre-
ate an even larger error. In setting 3, we can only test the
two methods that contain a Hessian approximation because
the large model size and the limited resources on our device
do not allow us to compute the exact Hessian.
We examine the error of the proposed method k c;
t  
 c
t k, which is the distance between exact FIP and estimated
FIP underL2 norm. The exact FIP is obtained by conducting
leave-one-out tests. We track the error of one randomly se-
lected client’s FIP and show the result in setting 2 in ﬁgure
2; results in the other two settings are provided as supple-
mentary materials. Here we can see the necessity of LWET:
without LWET there will be a sharp increase in the error at
about the 120-th round (the error is caused by the ﬁrst fully-
connected layer as mentioned earlier in Fig. 1). Further tak-
ing a Hessian approximation only causes a slight increase in
the error.
Inﬂuence on Loss
In this section, we give a more intuitive demonstration of the
experimental results by mapping the high-dimensional FIP
to a scalar, FIL. The exact FIL is obtained from the result of
leave-one-out test.
By adding estimated FIP to the original global model, we
get a estimation of the model trained with one client re-
moved wt + c
t . Then we test it onDtest and subtract from
the result the loss of the original model to get the estimated
FIL, i.e.,L(wt + c
t ;Dtest) L (wt;Dtest).
We compare the estimated FIL with the exact FIL and
show the results in Fig. 3. The correlation between the es-
timated and exact FIL is measured by Pearson’s correlation
coefﬁcient, and we plot its variation with time in Figs 3(a),
3(b), and 3(c). We also visualize the correlation in the last
round in Figs 3(d), 3(e), and 3(f). In particular, the proposed
method achieves a Pearson correlation coefﬁcient of 0.6200
at the last round in setting 3, the most difﬁcult setting; in set-
ting 1 Pearson correlation coefﬁcient is 0.9857 and in setting
2 it is 0.7957.
Extended Experiments
In this section, we use FIL and FIA for client valuation and
client cleansing, despite that we believe there are more po-
tential applications based on other types of inﬂuence given
by FIP, for example, the client-level inﬂuence on the predic-
tion results.
Inﬂuence on model performance can be used as a reason-
able metric to determine a client’s value. We conduct the
following experiment: we make an observation at clients’
inﬂuence at a certain round, remove some of the clients with
highest/lowest inﬂuence, from the training set, and then con-
tinue the learning process. The experiment is repeated with
different fractions of clients removed and the performance
of the ﬁnal global model is recorded each time. As can be
seen from Fig. 4, removing valuable clients (those with high
FIL or low FIA) greatly degrades the model performance,
which indicates the importance of these clients. In contrast,
removing least valuable clients (those with low FIL or high
FIA) improves the model performance. And the curve of
randomly removing falls between the other two curves.
The results of removing least valuable clients elicits an
application, which we call client cleansing. It is a little dif-
ferent from the data cleansing in Hara, Nitanda, and Mae-
hara (2019). Data cleansing is conducted by retraining the
model with a subset of data removed. In the setting of fed-
erated learning, as we mentioned before, it does not make
10565

0 250 500 750 1000
Round
0.92
0.94
0.96
0.98
1.00Pearson's R H.A.
No H.A.
(a)
0 200 400
Round
−0.25
0.00
0.25
0.50
0.75
1.00
Pearson's R
LWET & H.A.
Only LWET
Basic estimator (b)
0 500 1000 1500 2000
Round
0.0
0.2
0.4
0.6
0.8Pearson's R
LWET & H.A.
Only H.A. (c)
−0.001 0.000 0.001 0.002
Actual FIL
−1.0
−0.5
0.0
0.5
1.0
1.5
2.0
Estimated FIL
1e−3
(d)
−0.02 0.00 0.02 0.04
Actual FIL
−0.02
0.00
0.02
0.04
Estimated FIL (e)
−0.01 0.00 0.01 0.02
Actual FIL
−0.01
0.00
0.01
0.02
Estimated FIL
 −0.01 0.00 0.01
0.0
0.5
 (f)
Figure 3: (a)(b)(c) The change of Pearson coefﬁcient over rounds in setting 1, 2 and 3. (d)(e)(f) Estimated FIL vs. actual FIL in
settings 1, 2 and 3. The inset in (f) shows the results with a wider y-axis range. In setting 1, we counted 200 clients that were
randomly selected from the 1000 clients. In settings 2 and 3, we counted all the clients (50 and 100 clients, respectively).
0.0 0.1 0.2 0.3 0.4
Fraction of clients removed
1.35
1.36
1.37
1.38Loss
Removing low
Removing high
Random
0.0 0.1 0.2 0.3 0.4
Fraction of clients removed
0.85
0.90
0.95
1.00
1.05Loss
(a)
0.0 0.1 0.2 0.3 0.4
Fraction of clients removed
0.52
0.53
0.54
0.55Accuracy
0.0 0.1 0.2 0.3 0.4
Fraction of clients removed
0.74
0.75
0.76Accuracy
 (b)
Figure 4: We remove clients from the client set at a certain
roundx in three different ways: from those with lowest in-
ﬂuence, from those with highest inﬂuence, randomly. (a) We
use inﬂuence on loss and record the corresponding change
in loss. (b) We use inﬂuence on accuracy and record the
corresponding change in accuracy. Top: The results in set-
ting 1 with x = 700. Bottom: The results in setting 3 with
x = 1500.
sense to retrain the model in most real-world scenarios.
Therefore, we conduct client cleansing by removing a subset
of clients at a certain round and then continue the training.
By properly selecting the clients to be removed we can ef-
fectively improve the ﬁnal model. In setting 1, the accuracy
is increased from 53:86% to 55:66% by removing 30% of
the clients from the client set at round 700. In setting 3, the
accuracy is increased from 76:10% to 76:50% by removing
5% of the clients from the client set at round 1500.
Conclusion
In this paper, we proposed Fed-Inﬂuence as a new metric of
clients’ inﬂuence on the global model in federated learning,
inspired by the classical statistics notion of inﬂuence in cen-
tralized learning. The differences between the centralized
learning and federated learning create challenges in com-
puting this new metric. To develop an efﬁcient implementa-
tion, we ﬁrst proposed a basic estimator of individual client’s
inﬂuence on parameter, then further revised the estimator
and established an algorithm with both robustness and linear
cost. It is worth noting that the proposed method is accurate
without assumption on convexity or data identity, which is
validated by the empirical results on different settings. We
also demonstrated how Fed-Inﬂuence helps evaluate clients
and improve the model performance through client cleans-
ing. Our work provides a quantitative insight into the rela-
tionship between individual clients and the global model, we
envision which further enhancing the accountability of fed-
erated learning.
10566

Acknowledgements
This work was supported in part by National Key R&D Pro-
gram of China No. 2019YFB2102200, in part by China NSF
grant No. 62025204, 61972252, 61972254, 61672348, and
61672353, in part by Joint Scientiﬁc Research Foundation
of the State Education Ministry No. 6141A02033702, and in
part by Alibaba Group through Alibaba Innovation Research
Program. The opinions, ﬁndings, conclusions, and recom-
mendations expressed in this paper are those of the authors
and do not necessarily reﬂect the views of the funding agen-
cies or the government. Z. Zheng is the corresponding au-
thor.
References
Caldas, S.; Wu, P.; Li, T.; Konecn ´y, J.; McMahan, H. B.;
Smith, V .; and Talwalkar, A. 2018. LEAF: A Benchmark for
Federated Settings. arXiv preprint arXiv:1812.01097 .
Cook, R.; and Weisberg, S. 1980. Characterizations of an
empirical inﬂuence function for detecting inﬂuential cases
in regression. Technometrics 495–508.
Cook, R. D. 1977. Detection of inﬂuential observation in
linear regression. Technometrics 15–18.
Cook, R. D.; and Weisberg, S. 1982.Residuals and inﬂuence
in regression. New York: Chapman and Hall.
Friedman, J.; Hastie, T.; and Tibshirani, R. 2001. The el-
ements of statistical learning. Springer series in statistics
New York.
Haddadpour, F.; Kamani, M. M.; Mahdavi, M.; and
Cadambe, V . R. 2019. Local SGD with Periodic Averaging:
Tighter Analysis and Adaptive Synchronization. In Proc. of
NeurIPS, 11080–11092.
Hamer, J.; Mohri, M.; and Suresh, A. T. 2020. FedBoost:
A Communication-Efﬁcient Algorithm for Federated Learn-
ing. In Proc. of ICML, 3973–3983.
Hampel, F. R. 1974. The inﬂuence curve and its role in ro-
bust estimation. Journal of the american statistical associa-
tion 383–393.
Hara, S.; Nitanda, A.; and Maehara, T. 2019. Data Cleansing
for Models Trained with SGD. In Proc. of NeurIPS, 4215–
4224.
Jaeckel, L. 1972. The inﬁnitesimal jackknife. Unpublished
memorandum, Bell Telephone Laboratories.
Kairouz, P.; McMahan, H. B.; Avent, B.; Bellet, A.; Bennis,
M.; Bhagoji, A. N.; Bonawitz, K.; Charles, Z.; Cormode, G.;
Cummings, R.; et al. 2019. Advances and open problems in
federated learning. arXiv preprint arXiv:1912.04977 .
Karimireddy, S. P.; Kale, S.; Mohri, M.; Reddi, S. J.; Stich,
S. U.; and Suresh, A. T. 2019. Scaffold: Stochastic con-
trolled averaging for on-device federated learning. arXiv
preprint arXiv:1910.06378 .
Khanna, R.; Kim, B.; Ghosh, J.; and Koyejo, S. 2019. In-
terpreting Black Box Predictions using Fisher Kernels. In
Proc. of AISTATS, 3382–3390.
Koh, P. W.; Ang, K.; Teo, H. H. K.; and Liang, P. 2019. On
the Accuracy of Inﬂuence Functions for Measuring Group
Effects. In Proc. of NeurIPS, 5255–5265.
Koh, P. W.; and Liang, P. 2017. Understanding Black-box
Predictions via Inﬂuence Functions. In Proc. of ICML,
1885–1894.
Li, T.; Sahu, A. K.; Zaheer, M.; Sanjabi, M.; Talwalkar, A.;
and Smith, V . 2020a. Federated Optimization in Heteroge-
neous Networks. In Proc. of MLSys.
Li, X.; Huang, K.; Yang, W.; Wang, S.; and Zhang, Z. 2020b.
On the Convergence of FedAvg on Non-IID Data. In Proc.
of ICLR.
Ly, A.; Marsman, M.; Verhagen, J.; Grasman, R. P.; and Wa-
genmakers, E.-J. 2017. A tutorial on Fisher information.
Journal of Mathematical Psychology 40–55.
McMahan, B.; Moore, E.; Ramage, D.; Hampson, S.; and
y Arcas, B. A. 2017. Communication-Efﬁcient Learning of
Deep Networks from Decentralized Data. In Proc. of AIS-
TATS, 1273–1282.
Rothchild, D.; Panda, A.; Ullah, E.; Ivkin, N.; Stoica, I.;
Braverman, V .; Gonzalez, J.; and Arora, R. 2020. Fetchsgd:
Communication-efﬁcient federated learning with sketching.
In Proc. of ICML, 8253–8265.
Wang, J.; Liu, Q.; Liang, H.; Joshi, G.; and Poor, H. V . 2020.
Tackling the objective inconsistency problem in heteroge-
neous federated optimization. Proc. of NeurIPS .
Yan, Y .; Niu, C.; Ding, Y .; Zheng, Z.; Wu, F.; Chen, G.;
Tang, S.; and Wu, Z. 2020. Distributed Non-Convex Op-
timization with Sublinear Speedup under Intermittent Client
Availability. arXiv preprint arXiv:2002.07399 .
Yu, H.; Yang, S.; and Zhu, S. 2019. Parallel Restarted SGD
with Faster Convergence and Less Communication: Demys-
tifying Why Model Averaging Works for Deep Learning. In
Proc. of AAAI, 5693–5700.
Zhao, Y .; Li, M.; Lai, L.; Suda, N.; Civin, D.; and Chan-
dra, V . 2018. Federated Learning with Non-IID Data.arXiv
preprint arXiv:1806.00582 .
10567