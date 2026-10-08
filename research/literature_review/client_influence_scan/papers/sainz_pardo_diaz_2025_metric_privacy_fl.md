Metric Privacy in Federated Learning for Medical Imaging: Improving Convergence and
Preventing Client Inference Attacks
Judith S´ainz-Pardo D´ıaza,∗, Andreas Athanasioub, Kangsoo Jungb, Catuscia Palamidessib, ´Alvaro L´opez Garc´ıaa
aInstituto de F´ ısica de Cantabria (IFCA), CSIC-UC
Avda. los Castros s/n. 39005 - Santander (Spain)
bINRIA Saclay, Ecole Polytechnique (LIX)
1 Rue Honore d’Estienne d’Orves 91120 - Palaiseau, Ile-de-France (France)
Abstract
Federated learning is a distributed learning technique that allows training a global model with the participation of di fferent data
owners without the need to share raw data. This architecture is orchestrated by a central server that aggregates the local models
from the clients. This server may be trusted, but not all nodes in the network. Then, differential privacy (DP) can be used to privatize
the global model by adding noise. However, this may affect convergence across the rounds of the federated architecture, depending
also on the aggregation strategy employed. In this work, we aim to introduce the notion of metric-privacy to mitigate the impact
of classical server side global-DP on the convergence of the aggregated model. Metric-privacy is a relaxation of DP, suitable for
domains provided with a notion of distance. We apply it from the server side by computing a distance for the di fference between
the local models. We compare our approach with standard DP by analyzing the impact on six classical aggregation strategies. The
proposed methodology is applied to an example of medical imaging and di fferent scenarios are simulated across homogeneous
and non-i.i.d clients. Finally, we introduce a novel client inference attack, where a semi-honest client tries to find whether another
client participated in the training and study how it can be mitigated using DP and metric-privacy. Our evaluation shows that
metric-privacy can increase the performance of the model compared to standard DP, while offering similar protection against client
inference attacks.
Keywords: Federated Learning, Differential Privacy, Metric Privacy, Data Privacy, Medical Imaging
1. Introduction
The development of artificial intelligence models, such as
machine and deep learning (ML /DL) and their application on
distributed data is a field of emerging interest, especially with
the rise of federated learning [1]. This architecture allows to
train a global ML /DL model on multiple data owners without
sharing their data with each other or with a third party [2].
Federated learning (FL) is considered a privacy preserving
machine learning architecture, since the global model is built
from local models trained by each data owner (in the following,
client), without the need for the data to leave the devices, insti-
tutions or data centers that generate and /or store it. However,
these approaches are not free from attacks on the privacy of the
clients involved and the data used, with the most notable being
membership inference attacks [3, 4], reconstruction attacks and
model inversion attacks [5, 6] among others.
Differential privacy (DP) and homomorphic encryption (HE)
are commonly included in FL architectures to add an additional
∗Corresponding author
Email addresses: sainzpardo@ifca.unican.es (Judith S´ainz-Pardo
D´ıaz), andreas.athanasiou@inria.fr (Andreas Athanasiou),
gangsoo.zeong@inria.fr (Kangsoo Jung),
catuscia@lix.polytechnique.fr (Catuscia Palamidessi),
aloga@ifca.unican.es ( ´Alvaro L´opez Garc´ıa)
layer of privacy. For example, if we want to perform a secure
model aggregation, HE can be applied so that the models are
aggregated in an encrypted way [7]. Another option is to add
noise to each local model before sending it for aggregation (ap-
plying DP), in case the aggregator (in the following, the server)
is not trusted.
A variant of standard DP is metric-privacy (also known as
d-privacy) [8, 9] which can be employed in domains with a
notion of distance. Like standard DP, metric-privacy o ffers a
bound on the probability that the same result is obtained from
two different datasets. This probability depends on a parameter
ϵ (like standard DP) but also on the distance between the two
datasets (unlike standard DP). In other words, metric-privacy
ensures that the noise is calibrated to achieve privacy within a
specific “radius” of distance. This is especially beneficial in ap-
plications where hiding an element within a group of neighbors
provides adequate privacy protection. However, metric-privacy
has received little attention in FL.
Protecting against an untrusted central server using DP has
been widely studied in FL. However most studies focus only on
the server using FedAvg(Federated Average) as the aggregation
function. Therefore, there is a gap in the literature regarding the
impact of global-DP in the accuracy of the model when other
aggregation functions are used.
Moreover, in this paper we also pose a different problem to-
Preprint submitted to Elsevier February 4, 2025
arXiv:2502.01352v1  [cs.LG]  3 Feb 2025

gether with its potential solution. Specifically, we assume that
all clients are semi-honest and that the server (i.e. aggregator)
is trusted. However, we suppose that one of the clients can act
as an attacker to find out if another one is participating in the
training or not. We name this a Client Inference Attack (CIA).
Knowing that a specific client participated in the dataset can
lead to further attacks, which can be a privacy concern. For in-
stance, it may strengthen other attacks, such as the membership
inference attack.
Our approach, which will be detailed in Section 6, is to use
metric-privacy from the server side (global-metric-privacy) to
protect the aggregated model from this type of attack. Hereafter
we will refer to global-metric privacy just as metric privacy
for simplicity of terminology. Note that using metric-privacy
we can better adjust the added noise obtaining a better perfor-
mance and convergence of the model (compared to standard
DP), while simultaneously protecting against CIAs.
For the experimental part of this work, an openly available
dataset belonging to the field of medical imaging has been used,
simulating different scenarios in terms of client distribution (ho-
mogeneous or non-i.i.d.), as well as an additional scenario to
test the protection against the proposed client inference attacks.
1.1. Contributions and structure of the work
The main contributions of this work can be summarized as
follows:
• First, we apply global-DP (i.e. server-side DP) to test the
performance of the model across six different aggregation
strategies. This yields interesting results as global-DP af-
fects the performance of the model di fferently depend-
ing on the chosen aggregation function. We experimen-
tally evaluate the performance of metric-privacy, global-
DP and, as a baseline, vanilla FL (i.e. without adding
any noise at all), across di fferent aggregation functions,
showing that metric-privacy o ffers in all cases a better
model accuracy than global-DP. The experiments are con-
ducted using a medical imaging database openly avail-
able and simulating di fferent clients and scenarios. To
the best of our knowledge this comparison between dif-
ferent aggregation strategies and privacy approaches has
not been studied before in scientific papers.
• We study the application of metric-privacy from a trusted
server by introducing a proper distance metric. This met-
ric is dynamically calculated from the server side in each
round only taking into account the model parameters (i.e.
without analyzing the clients’ data distributions, which
could affect privacy). As far as we are aware of, this is
the first work that incorporates metric-privacy into an FL
architecture.
• We introduce the concept ofClient Inference Attackwhere
a semi-honest client tries to infer whether or not a cer-
tain client is participating in the training. We conduct
experiments showing that while both metric-privacy and
global-DP provide adequate protection against this type
of attack, the former produces a more accurate model.
This work is structured as follows: first, in Section 2 we re-
port the state of the art and the related work. Section 3 presents
different aggregation strategies that can be used in federated
learning. Then, Section 4 provides the theoretical basis for
using DP in federated learning. In Section 5 we define the
client inference attacks. Section 6 explains the methodology
followed and how we can apply metric-privacy in FL. Next,
in Section 7 we conduct experiments across homogeneous and
non-i.i.d clients and a CIA scenario. Finally, Section 8 draws
the conclusions and outlines future work.
2. Related work
The incorporation of DP in data science ecosystems, espe-
cially in relation to the training of ML/DL models, is a topic that
has been extensively studied in the field of privacy-preserving
machine learning [10]. Regarding the training of deep learn-
ing models (more specifically deep neural networks), the use
of DP during the model training has been explored in multi-
ple works following the di fferentially private stochastic gradi-
ent descent (DP-SGD) approach [11, 12, 13], which consists of
adding noise during the process of stochastic gradient descent.
This is implemented in di fferent Python libraries, like Tensor-
Flow Privacy [14] or Opacus [15].
FL is susceptible to privacy attacks, despite the fact that
each client trains their model locally, without revealing their
local dataset. Among various inference attacks, the Member-
ship Inference Attack (MIA) [16, 3, 17] determines whether
a particular record is part of a dataset or not. To do so, it is
assumed that the adversary (central server) owns a so-called
shadow model of the target client, which is similar to the target
client’s model (i.e. follows the same distribution) but is trained
on a different, disjoint, dataset. MIAs are widely employed in
the literature for evaluating privacy risks [18].
The Source Inference Attack (SIA) [4] is a more advanced
inference attack. An honest-but-curious server in FL attempts
to identify exactly which client owns a specific data point used
for training. To estimate the source of the data point, the server
leverages the prediction losses of local models on particular
data points by using a Bayesian approach.
Among reconstruction attacks, arguably the most well-known
is Deep Leakage from Gradients (DLG) [5]. DLG usually ex-
ploits the shared gradients during the training process in feder-
ated learning to reconstruct sensitive training data. This attack
shows that even gradient information can leak important details
about the original data. In both reconstruction and inference at-
tacks, applying DP is crucial to protect sensitive information of
the participating clients.
However, in FL adversaries may also take the form of par-
ticipants or other external actors with access to the (shared) ag-
gregated model. In the case of an FL architecture where the
server that orchestrates the training is not trusted, clients can
add DP during training by applying DP-SGD, but they can also
do so once the model is trained, adding noise to the model up-
dates [19]. This will prevent the central server from extracting
information from the data used in the training.
2

Concerning medical imaging, in [20] the authors apply DP-
SGD in a FL architecture for medical image analysis, specifi-
cally using histopathology images. On the other hand, several
works have explored the incorporation of DP to FL architec-
tures. For example, in [21] DP is added from each client to the
model updates before sending the model to the server, specifi-
cally in a personalized federated learning approach.
In this work we assume that the server is trusted. The at-
tacker can either be another client who tries to infer information
about other clients (e.g. find out if a certain client is participat-
ing in the training). Hence the model that we aim to protect
is the aggregated (global) one. Therefore, the server can glob-
ally add DP to the aggregated model after receiving the updates
from all the clients. This will prevent the extraction of informa-
tion by the clients in each round. Also it will protect the final
resulting model when it is made publicly available. Few works
can be found in the literature concerning applying DP from the
server side. For example, in [22] local and central DP are com-
pared in a FL setting showing the protection against backdoor
attacks, defining central DP (CDP) as the case in which“the FL
aggregation function is perturbed by the server” . In this work
we refer to this approach as global-DP.
3. Federated learning aggregation functions
The classic horizontal federated learning architecture in-
volves a server and multiple clients or data owners. On the
one hand, the clients have their data stored locally in their own
infrastructures and are in charge of performing the model train-
ing in a distributed way. On the other hand, the server is in
charge of aggregating the individual locally trained models in
order to build a global model. Thus, the scheme followed un-
der a FL architecture can be summarized in the following steps:
(1) each client receives a model to be trained locally (defined
by the server or consensuated by the clients); (2) each client
trains the model locally with its data; (3) the clients send the
parameters or weights that define the model trained locally (or
the model updates); (4) the server receives the models /updates
from the clients involved and aggregates them to build a global
model; (5) the aggregated global model is distributed back to
the clients to be retrained. This process is repeated from step
(2) for a pre-defined number of rounds.
One of the most critical steps is the aggregation process car-
ried out from the server side, since it is the moment when the
global model is built from all the locally trained models. Thus,
selecting an appropriate aggregation strategy in relation to the
use case employed is key to achieve an aggregated global model
that is robust and accurate. Some classically applied aggrega-
tion functions are presented below:
Federated Average (FedAvg). The most natural approach is to
simply average the parameters. However, it must be taken into
account that there will be clients who have trained the model
using more data than others, thus being able to obtain a more
accurate prediction. Thus, it is important that the average is
made in a weighted way taking into account the number of data
available for each client. Proposed by B. McMahan et al. in
2017 ([2]) when introducing the FL architecture (where its suc-
cessful application was shown for the MNIST, CIFAR10 and
Shakespeare datasets), the Federated Averaging strategy ( Fe-
dAvg) is the most widely used aggregation function in the liter-
ature for its simplicity and effectiveness.
Let n be the number of clients participating in the FL ar-
chitecture, ni the number of data of client i (∀i ∈ {1,..., n})
and w(r)
i the weights or model parameters obtained for each
client i after training in the round r. The aggregation performed
with FedAvg consists on the weighted mean of the parameters
of each clients, as follows, for round r:
w(r) = 1Pn
i=1 ni
nX
i=1
w(r)
i ni (1)
Then, w(r) are the aggregated weights resulting from roundr
that will be transmitted to each client when starting roundr + 1.
Federated Average with Momentum (FedAvgM). This strategy
was implemented in [23] and the proposed FedAvgM method
arises from the idea of improving stability and e fficiency by
using the momentum to store gradients from previous rounds,
also in order to dampen oscillation in the training. Then, let
w(r) the global parameters in roundr and ∆w(r+1) the aggregated
difference between the parameters obtained in rounds r + 1 and
r: ∆w(r+1) = 1P i=1nni
Pn
i=1 ni(w(r)− w(r+1)
i ), note that w(r+1)
i are
the local parameters obtained for client i in round r.
In order to get the aggregated parameters using FedAvgM
we have to calculate the momentum vector in round r v(r) (ini-
tialized as v(0) = 0), in such a way that v(r+1) = βv(r) + ∆w(r+1).
Then, we will calculate the weights in round r + 1 as w(r+1) =
w(r)− v(r+1), beingβ the momentum. Additionally, we can add
the term µ as the server learning rate introducing it as follows:
w(r+1) = w(r)−µv(r+1).
FedAvgM aims to incorporate the idea of momentum at the
server side during the aggregation process to improve the sta-
bility and speed of convergence of the overall model [23]. Ac-
cording to the implementation in the Flower library [24], if the
value for the momentum (β) is strictly greater than zero, some
global initial parameters for the model must be included. As
will be explained in Section 7, a validation dataset will be used
to get these initial model parameters.
Federated Median (FedMedian). The Federated Median strat-
egy ( FedMedian) was introduced in [25]. The proposed al-
gorithm for aggregating using the median takes into account
byzantine machines so that only in the case of normal worker
machines each such client computes the local gradient. Thus,
let gi(w(r)) be the gradients obtained for clienti with the weights
aggregated in round r (wr). The central server will be in charge
of getting the median of the local gradients in each round r:
gmedian(w(r)) = median(gi(w(r)), i∈{ 1,..., n}). Then, once such
aggregated gradient is calculated, the weights are updated for
round r + 1: w(r+1) = w(r)− µgmedian(w(r)) [25], being µ the
step-size or learning rate.
3

FedProx. The main objective of theFedProxaggregation strat-
egy is to deal with heterogeneous settings in FL schemes [26].
Specifically, it is a generalization and re-parametrization of the
FedAvg strategy. In order to calculate w(r+1)
i (the weights for
client i in round r+1) we have to minimize the following objec-
tive function hi for each client i, with Fi(·) the global objective
function at each client i:
h(w, w(r)) = Fi(w) + µ
2||w− w(r)||2.
Thus, w(r+1)
i = arg minw hi(w). Then the server aggregates
the values w(r+1)
i ∀i∈{ 1,..., n} by taking the mean.
Note than in this approach we need to choose a value for the
parameterµ.
Federated Optimization (FedOpt) and adaptive Federated Opti-
mization with Yogi (FedYogi). The FedOpt strategy introduced
in Algorithm 1 from [27] aims to improve FedAvg by using
gradient-based optimizers with given learning rates that must
be customized both from the client and server side. Specifi-
cally, let w(r) the global weights at round r and w(r)
i the weights
for client i at round r, then we can define∆w(r)
i = w(r)
i − w(r) and
∆w(r) = 1
n
Pn
i=1 ∆w(r)
i . From the server side, we will use an op-
timizer for calculating w(r+1) given w(r), ∆w(r) and the learning
rate.
Note that this approach allows the use of di fferent adap-
tive optimizers in the server side, such us Adam, Adagrad or
Yogiamong others (the we will have theFedAdam, FedAdagrad
and FedYogiaggregation strategies respectively). The adaptive
methods are used on the server side while SGD is used from
the client side [27]. More details are given in the pseudocode
of Algorithm 2 from [27].
These strategies require additional parameters, such as the
server-side and client-side learning rate, β1 andβ2 as momen-
tum and second momentum parameters respectively and τ for
controlling the degree of adaptability of the algorithm. Ac-
cording to its implementation in Flower, some initial global pa-
rameters for the model need to be introduced when using these
functions. The use of these global initial parameters provides a
consistent starting point for all clients.
4. Di fferentially private federated learning
Differential privacy (DP) is a privacy preserving technique
that aims to provide a formal guarantee about what an analyst
(adversary) can learn about an individual in a database. The
probability that the adversary observes any event is comparable
in cases where certain information from an individual is or is
not included in the dataset. In addition, with DP we can en-
sure that an adversary with unlimited computational capacity
and auxiliary information cannot break the established privacy
level.
In other words, DP states that an algorithm is di fferentially
private if by viewing its result an adversary cannot know whether
a particular individual’s data is included in the database or not.
This is typically done with the addition of noise (i.e. data ob-
fuscation). The goal is to protect user’s privacy while allowing
the data to be meaningful for analysis. For this purpose, di ffer-
ent mechanisms can be followed to ensure that the added noise
does not significantly alter the analysis.
Let’s start by introducing the notions of adjacency and ran-
domized algorithm which are crucial for defining DP:
Definition 1. Two databasesD and D are adjacent (notated by
D∼D ′) if they differ by exactly one record.
Definition 2. A probabilistic or randomized algorithm for the
query f over the database X is a probabilistic function g from
X to a set of valuesZ such as g :X−→D (Z), withD(Z) the
set of probability distributions inZ.
Now we can defineϵ-differential privacy:
Definition 3. A randomized algorithmM, with domainD and
rangeR, satisfies ϵ-differential privacy if for every pair of ad-
jacent datasets D, D∈D , for every S⊆R andϵ >0 :
P[M(D)∈ S ]≤ eϵP[M(D)∈ S ]
In view of the above definition, the value ofϵ is the privacy
budget, which allows to control the level of privacy (the amount
of privacy loss allowed). We note that the lower the value ofϵ,
the higher the privacy, but, in most of the cases, the lower the
usefulness of the data for analysis (although this is not always
the case [28]).
Approximate DP is a famous relaxation of the previous def-
inition where we allow the above inequality to not hold with a
small probability. We can define (ϵ,δ )-differential privacy as:
Definition 4. A randomized algorithmM, with domainD and
rangeR, verifies (ϵ,δ )-differential privacy if for every two pair
of adjacent datasets D, D∈D and for every S ⊆R ,ϵ >0 and
δ∈ [0, 1]:
P[M(D)∈ S ]≤ eϵP[M(D)∈ S ] +δ
The parameterδ is the probability of exceeding the privacy
budget, i.e. with probability 1 −δ the privacy loss will not be
greater thanϵ.
Some of the most commonly used mechanisms to ensure
DP are among others the Laplace Mechanism, the Exponential
Mechanism and the Gaussian Mechanism.
Let’s introduce the Gaussian Mechanism, as it will be used
during the experimental settings performed in this work.
First, let us define the notion of sensitivity:
Definition 5. Be f :D−→ Rk, the l2-sensitivity of f is defined
as follows:
∆2( f ) := max
||x−y||1
|| f (x)− f (y)||2
Now we can define the Gaussian Mechanism as given in
Definition 6:
4

Definition 6. Given the function f :D−→ Rk, we define the
Gaussian Mechanism (MG) for (ϵ,δ)-differential privacy as fol-
lows:
MG(x, f (·),ϵ,δ ) := f (x) + (Y1,..., Yk).
Note that Yi (∀i∈{ 1,..., k}) follows the distribution N (0,σ 2),
withσ =
∆2( f )
√
2 log(1.25/δ)
ϵ .
DP can be applied using two approaches: locally and glob-
ally. In the local setting we assume that the clients have to ob-
fuscate their data on their own, using local-DP (LDP). On the
other hand, in the global setting the clients totally trust a central
entity to collect their data and add noise to the output of a query
or algorithm, using global-DP (GDP). Note that trusting the cu-
rator or central server is a vital assumption. If such assumption
cannot be made using local-DP is a better choice.
If we extrapolate these notions to a FL architecture, we can
think of two paradigms depending on the side of the network in
which the potential attacker is located:
• Trusted server: in this case we can add DP from the
server side. The goal here is to protect against some
clients that can act as attackers, or to prevent information
extraction from the final global model when it is pub-
lished. This is exactly our aim in this work.
In this line, the idea is to perturb the global aggregated
model by adding noise to it once aggregated. In this
work, we analyze the impact of adding such noise accord-
ing to the aggregation strategy applied, assuming that we
have a network composed by a trusted honest server.
• Untrusted server: in this case the clients will be inter-
ested in sending to the server a noisy version of the model.
This is because some information could be extracted con-
cerning the data of each client by analyzing the local up-
dates that are sent in each round. In this line the clients
can train the model and then add noise before sending
to the server for aggregating, or to perform differentially
private stochastic gradient descent (DP-SGD) for adding
DP during the training process. Recall that we do not
consider this scenario in this work.
Another variation of DP is metric-privacy or d-privacy [8,
29]. In metric-privacy the datasets are not just adjacent (i.e.
they just have one value that is not the same) but their actual
difference is calculated using some distance metric. Then, the
privacy level can be adapted so as it o ffer better privacy when
the distance is small. In other words, metric-privacy allows the
adversary to find a rough estimation of the clients’ data, but the
actual values remain protected. This is done in exchange of
better utility, as less noise is added.
We can define metric-privacy as:
Definition 7. LetD be a domain provided with a metric d :
D2−→ R≥0. The randomized algorithm M with rangeR and
with domainD satisfiesϵ-metric-privacy, if for every two pair
of inputs x, x∈D with distance d(x, x), every y∈R and every
ϵ >0:
P[M(x) = y]≤ eϵ·d(x,x)P[M(x) = y]
In this work our goal is to evaluate the impact of metric-
privacy and classic global-DP according to di fferent aggrega-
tion strategies in FL architectures. Note that our goal is to
guarantee metric-privacy separately for each round of the FL
architecture.
Finally, note that we aim to analyze in practice how can
we improve the convergence of a federated model using global
metric-privacy (in the following metric-privacy or metric-DP)
with respect to classic global-DP, while we are protecting the
aggregated model from information extraction and ensuring a
similar protection regarding client inference attacks. We want
to highlight that this work doesn’t aim to perform a fair compar-
ison between global-DP and metric-privacy (as the amount of
noise added in each approach is di fferent, as will be explained
later), but to introduce this notion for better tuning the amount
of noise added for finding a balance between predictive perfor-
mance and resistance against client inference attacks.
5. Client Inference Attacks
One of the main hypotheses that we want to assess within
this work is that metric-privacy, besides offering a better accu-
racy than standard global-DP in FL, may also be helpful against
inference attacks.
In this section, we introduce the Client Inference Attack
(CIA), a novel attack where a semi-honest client (attacker) re-
ceives the global aggregated model from an honest (trusted)
server and tries to determine if another client belongs to the list
of participants. This attack can serve as a first step to strengthen
other attacks, such as a Membership Inference Attack [3] or a
Source Inference Attack [4].
For instance, consider a model jointly built by hospitals and
competing pharmaceutical companies to treat a disease. No
party (apart from the central server) knows exactly which hospi-
tals or which companies are working on the model. The central
server is honest and releases the aggregated model back to the
clients. However, a curious pharmaceutical company P may
want to launch a CIA attack and figure out whether or not a
certain hospital H is participating in the training. If this CIA
shows that H did indeed participated, then P can facilitate fur-
ther inference attacks on H and the data used (which may be
associated with the patients from the hospital).
In order to achieve this attack, we assume that the attacker
has sufficient knowledge of the target client’s training dataset
to create a shadow training dataset ; a dataset that mimics the
distribution of the target client’s dataset. This shadow dataset
can either be entirely disjoint from the target dataset (but sam-
pled from the same distribution) or it may overlap as a subset,
if the attacker has obtained some data points from the target.
Note that a similar assumption is essential to other well-studied
inference attacks such as the Membership Inference Attack [3].
In addition, the clients participating in a FL architecture can
also receive information about the process from the server, such
5

as the aggregated metrics calculated in each round, in order to
evaluate the convergence of the model.
Let us consider a bit vector s with n bits, with each bit i be-
ing set to 1 if user i participated in some round r in the training
of the model W, and 0 otherwise. We can define the CIA as
follows:
Definition 8 (Client Inference Attack). Given a global aggre-
gated model W and the shadow dataset Dx of client x, the Client
Inference Attack can be defined as:
C(Dx, W) := P(sx = 1|Dx, W)
Note that in this work we assume that the number of clients
is fixed in all rounds. This makes sense in the case of cross-silo
configurations. Thus, the client that is going to perform a CIA,
wants to know if the target has participated or not based on the
shadow dataset available.
First of all, it is clear that global-DP can be a measure that
helps to prevent this type of attack. This is because in CIA
the malicious client analyzes the performance of the aggregated
model. Adding Di fferential Privacy to the aggregated model
limits the amount of information disclosed by mathematical
definition. Thus, adding noise to the aggregated model seeks
to mask the individual contributions of the di fferent participat-
ing clients. However, it is necessary to calibrate the noise added
to maintain a good balance between protection against CIA and
the performance of the model.
In this sense, we propose to use the notion of metric-privacy.
As we are about to explore, adding noise based on a metric al-
lows to make a finer balance between privacy and model perfor-
mance, since in case of similar contributions from the clients
it will be necessary to add less noise. Even in scenarios with
non-i.i.d. clients, metric-privacy can contribute to the proper
calibration of privacy by adding only the “necessary amount”
of noise. Using metric-privacy in this case makes sense as we
assume that the participating clients are static throughout the
rounds.
We will show in Section 7 an experimental analysis of this
approach in an example of medical imaging. The hypothesis
which we will be testing is that metric-privacy can contribute
to better calibrate the noise compared to global-DP. Such noise
calibration can reduce the defense against CIA, but at the same
time improve the performance of the same allowing to achieve
a better balance between utility and privacy.
6. Proposed methodology
We are now ready to describe our proposed approach. Let
us recall our trust model: we assume that there is an honest
central server in a cross silo setting with all the initial clients
participating during the whole training. However, we suppose
that some of the clients involved may act as attackers seeking
to extract information from the models and therefore from the
data of other clients participating in the network. Specifically,
our goal is to protect against Client Inference Attacks.
To do this, we first apply di fferential privacy with fixed
clipping norm to the aggregation process from the server side
(global-DP). By doing so we also prevent information leakage
from external attackers, when publishing the aggregated model
in a model repository. Specifically, in our approach we use the
Gaussian Mechanism setting a noise multiplier associated to the
privacy level we want to define.
Concerning the selection of the parameters for applying DP,
we have to note that the clipping norm is a hyperparameter that
must be optimized. More specifically, for the first round we
calculate the following metric:
˜c = max
i, j∈{1,...,n}
i, j
||wi− w j||2.
We started optimizing the clipping norm using ˜ c, and we
finally take a valueC = 5 for the experimental simulations after
performing different tests.
Next, we propose a novel adaptation of metric-privacy to
deal against such attacks. Recall that with metric-privacy we
aim to offer privacy guarantees based on the distance between
the clients. However, the server does not have access to the
raw data of the clients. Hence the question arises: how can
the server calculate the distance between the clients, without
knowing their data?
To overcome the hurdle, we propose a metric that depends
on the distance between the model updates rather than the ac-
tual clients’ data. In order to do so, the server calculates the
maximum distance for each pair of clients by analyzing the lo-
cal weights received from each one.
Let w(n)
i be the local weights of the model trained for client
i in round n, with w(n)
i (ℓ) being the weights of the layerℓ of the
model in round n. Moreover let L be the set of layers of the
trained model and nc be the number of clients. We define the
distance in round n as follows:
d(n) = max
i, j∈{1,...,nc}
i, j

1
|L|
X
ℓ∈L
||w(n)
i (ℓ)− w(n)
j (ℓ)||F
.
Note that||·|| F represents the Frobenius norm.
The hypothesis which we will be testing is that using met-
ric privacy with d(n) will help to fine-tune the amount of noise
needed in each round, ultimately allowing better convergence.
We explained above that we will compare metric privacy
with global-DP applied using the Gaussian mechanism. For
a fair comparison we will employ the Gaussian mechanism in
metric privacy as well settingϵ =ϵ· d(n) for each round n.
The classic implementation of DP with fixed clipping in FL
involves introducing the clipping norm C, the noise multiplier
(nϵ) and the number of sampled clients nc. Note that the noise
multiplier has the opposite interpretation to the privacy budget,
since the lower the value, the higher the utility and the lower
the privacy, as less noise is injected.
In this line, we have calculated the metric d(n) dynamically
in each round n and use it by dividing the standard deviation of
the Gaussian noise as follows:
6

N
 
0, ne· C
nc· d(n)
!
.
Then, the objective is to analyze the impact of adding metric-
privacy to the aggregated model form the server side, ensuring
that metric-privacy in fulfilled in each FL round. In addition,
we want to evaluate the impact of this privacy enhancing tech-
nology compared to classic global-DP in terms of a client in-
ference attack. Note that if d(n) > 1, we are adding more noise
with metric-privacy if we keep the same noise multiplier, but
we want to study how this method allow us to find a proper
value for the noise added while giving similar protection against
client inference attacks. This will be studied experimentally in
the following section.
7. Experimental results
7.1. Data under study
In this work, for the experimental simulation of the pro-
posed methodology we have taken the data from the Alzheimer
dataset available in [30]. Our goal is to carry out this compar-
ative study with this open dataset that can be meaningful in a
practical scenario. The dataset contains brain MRI images and
is especially useful for the testing and investigation of applied
healthcare AI models. Specifically, the goal with this dataset is
to predict, based on the MRI images, the level of Alzheimer’s
disease of the patient. For this purpose there are 4 categories:
mild demented (0), moderate demented (1), non-demented (2)
and very mild demented (3). The initial train dataset is com-
posed of 5120 images and in the test dataset 1280 images are
available. The distribution of the classes in each of these two
datasets is shown in Table 1.
T otal Class 0 Class 1 Class 2 Class 3
Train 5120 724 49 2566 1781
Test 1280 172 15 634 459
Table 1: Original train and test set distribution.
In addition, Figure 1 shows an example of each class of
images available in the train dataset.
To perform the FL scheme we will simulate 4 clients from
the train dataset, while leaving the initial test set for analyzing
the performance of the final models and for validation. Specif-
ically, the 50% of the test set will be taken as validation from
the server side to train some initial parameters for those aggre-
gation functions that require them. The remaining 50% will be
considered the test set for analyzing the resulting models.
In order to analyze the impact of adding global-DP and met-
ric privacy according to the aggregation function used in FL, we
propose two experimental settings for the same initial central-
ized dataset: homogeneous clients and non-i.i.d clients. Once
the train data has been distributed to the different clients, in both
cases we leave 20% for test in each client and 80% for train.
This split is done in a stratified way for maintaining the sta-
tistical significance associated with the representation of each
class.
Figure 1: Example of the original images from the train set by classes.
7.1.1. Homogeneous clients
In this first approach we perform a division of the initial
training set into four disjunct clients such that the distribution
of the number of images in each class is homogeneous. For
this, we perform a stratified division of the training set by as-
signing to each client the same number of data in a balanced
way (almost the same number of each of each class for each
client split). Thus, Table 2 shows for each client the distribu-
tion of the four classes present in the original database.
T otal Class 0 Class 1 Class 2 Class 3
Client 1 1280 181 12 641 446
Client 2 1280 181 12 642 445
Client 3 1280 181 12 642 445
Client 4 1280 181 13 641 445
Table 2: Homogeneous clients distribution.
7.1.2. Non-i.i.d clients
In this second setting, the aim is to simulate a scenario
where we work with clients who possess different datasets. For
this purpose, the initial training set has been divided among four
disjoint clients, with each assigned a di fferent number of data
points and an unequal distribution of classes represented in the
dataset. Specifically, Table 3 shows the number of data for each
client and the number of data belonging to each predicting class
in each of them.
7.2. Model analyzed
Different models and hyperparameters have been analyzed
during the experimental procedure detailed in this work in order
7

T otal Class 0 Class 1 Class 2 Class 3
Client 1 1792 280 16 881 615
Client 2 768 107 13 368 280
Client 3 2048 257 17 1054 720
Client 4 512 80 3 263 166
Table 3: Non-i.i.d clients distribution.
to select the one implemented. Finally, the implemented model
for performing the different tests (which was fine-tuned taking
into account a validation split of the dataset) is a convolutional
neural network (CNN) composed by the following layers:
• Conv2D layer. 32 neurons. Activation relu. Input shape:
(128, 128, 1).
• MaxPooling2D((2,2)).
• Conv2D layer. 64 neurons. Activation relu.
• MaxPooling2D((2,2)).
• Conv2D layer. 128 neurons. Activation relu.
• Flatten layer.
• Dense layer. 64 neurons. Activation relu.
• Dropout layer. Rate 0.1.
• Dense layer. 32 neurons. Activation relu.
• Dropout layer. Rate 0.1.
• Output: Dense layer. 4 neurons. Activation softmax.
The model is compiled using theAdam optimizer, thesparse
categorical cross-entropy as loss function and the accuracy as
metric. The batch size used for the training in 32 and we train
it during 5 epochs and 20 rounds of the Federated Learning ar-
chitecture.
Note that in order to follow a real use case, we have per-
formed a hyperparameter tuning using a validation dataset. Then,
we have run the model to simulate a real use case with a fixed
seed for the libraries with stochastic behavior, in order to make
a fairer comparison between the three approaches (non-DP, global-
DP and metric-privacy). However, in order to analyze the ro-
bustness of the results, we have tested the performance by run-
ning the model 5 times with each scenario and each aggregation
function (e.g. for the case of i.i.d. clients we have conducted
90 experiments). These results are shown for the test set as a
function of the accuracy in Table A.12 in Appendix A. In view
of the standard deviation obtained in the 5 runs of the model
for such scenario, which is statistically significant, the results
of one run of the model are shown to better fit a real scenario
(see for instance [20] or [31]).
7.3. Results
In this section we show the results obtained with the three
approaches: vanilla FL (baseline, no DP), global-DP and metric-
privacy from the server side with six different aggregation strate-
gies. In addition, we show the aggregated accuracy obtained
for the clients test set and accuracy, precision and F1-score ob-
tained with each strategy for the initial test set. Finally, we
compare the three approaches regarding the prevention of the
Client Inference Attack.
7.3.1. Homogeneous clients
The aggregated results in the last five rounds of the FL ar-
chitecture (mean and standard deviation), using the CNN model
presented in the previous section are shown in Table 4, com-
paring the mean accuracy and standard deviation in the last 5
rounds across the six aggregation strategies.
Mean accuracy± std (last 5 rounds)
Strategy Vanilla FL Global-DP Metric-privacy
FedAvg 0.929± 0.008 0.881 ± 0.013 0.905 ± 0.007
FedAvgM 0.878± 0.013 0.759 ± 0.024 0.799 ± 0.022
FedMedian 0.920± 0.005 0.885 ± 0.013 0.905 ± 0.018
FedProx 0.931± 0.006 0.879 ± 0.008 0.907 ± 0.022
FedOpt 0.956± 0.004 0.918 ± 0.014 0.948 ± 0.005
FedYogi 0.950± 0.006 0.906 ± 0.008 0.911 ± 0.012
Table 4: Comparison on homogeneous clients of the mean and the standard de-
viation of the aggregated accuracy in the last five rounds of the FL scheme. Ag-
gregation performed without using DP and with global-DP and metric-privacy
with noise multiplier of 0.01.
Table 4 shows that with metric-privacy we always achieve
better results compared to those with global-DP, as well as a
better convergence over the course of the rounds (see Appendix
B). The evolution of the accuracy of the aggregated model in
clients test set (aggregated) is shown for each round of the FL
schema in Figure B.2 from Appendix B for the cases of the
global aggregated model without DP, with global-DP and with
metric-privacy.
Both in the case of global-DP and metric-privacy we intro-
duce the same value for the noise multiplier (0.01), however,
note that with metric-privacy we are dividing the noise multi-
plier by the distance d(n) in each round n. If d(n) > 1 we will
be adding more noise (we could expect the model to get worse
in its performance), while if d(n) < 1 we would be adding less
noise, which will result in greater similarity with the non-DP
model. Then, metric-privacy help us to find a proper value for
the standard deviation of the Gaussian noise, while we are still
protecting privacy and client inference attacks, as will be shown
in Section 7.4.
In Table 5 we show the results obtained with each approach
and each aggregation function for the global model obtained
after 20 rounds for the test set in terms of the accuracy, F1-
score and precision. In view of Table 5 we can highlight that
the performance with metric-privacy is better than the one with
global-DP with all the six strategies analyzed in terms of the
8

Vanilla FL Global-DP Metric-privacy
Strategy Accuracy F1-Score Precision Accuracy F1-Score Precision Accuracy F1-Score Precision
FedAvg 0.909 0.903 0.950 0.884 0.862 0.918 0.894 0.843 0.943
FedAvgM 0.884 0.768 0.932 0.762 0.554 0.607 0.823 0.608 0.631
FedMedian 0.932 0.899 0.946 0.875 0.827 0.913 0.895 0.843 0.943
FedProx 0.909 0.848 0.873 0.877 0.853 0.914 0.903 0.877 0.944
FedOpt 0.950 0.936 0.970 0.918 0.877 0.911 0.930 0.917 0.950
FedYogi 0.933 0.914 0.957 0.908 0.841 0.884 0.917 0.888 0.941
Table 5: Accuracy, F1-score and precision obtained in the test set with each approach. Homogeneous clients.
accuracy and the precision, and in 5 out of 6 in terms of the F1-
score. In addition, note that for FedProx strategy, the F1-score
and the precision is better with metric-privacy that without DP,
this may be due to a better generalization capability in terms of
the less-represented classes and a better mitigation of potential
overfitting.
In addition, for completeness of the analysis, Appendix C
shows the value for the AUC and the ROC curve obtained in
each case with each strategy and each of the three proposed
approaches (Figure C.4).
7.3.2. Non-i.i.d clients
Here, we conduct the experiments on the non-i.i.d. clients
case. It should be noted that the distribution of each class in
each data owner is statistically significant with the initial dis-
tribution, as it would not make sense for a hospital to have the
minority class as the majority class, which would not fit the
usual distribution of the pathology.
The non-i.i.d. nature is determined by the amount of data
in each data owner, with one being more representative than
the others as a function of the number of samples, as shown in
Table 3. Then, we have trained the FL architecture with these
four clients and the six aggregation strategies explained above.
Note that some of these strategies will be more appropriate than
others to deal with the non-i.i.d nature of the simulated data
owners.
Table 6 shows the aggregated results obtained for each client’s
test set in terms of the mean accuracy in the last 5 rounds of the
FL scheme as well as the standard deviation in these rounds.
Mean accuracy± std (last 5 rounds)
Strategy Vanilla FL Global-DP Metric-privacy
FedAvg 0.933± 0.002 0.878 ± 0.012 0.927 ± 0.008
FedAvgM 0.909± 0.007 0.790 ± 0.018 0.840 ± 0.016
FedMedian 0.926± 0.009 0.838 ± 0.013 0.905 ± 0.007
FedProx 0.937± 0.014 0.878 ± 0.016 0.943 ± 0.003
FedOpt 0.964± 0.007 0.925 ± 0.009 0.945 ± 0.006
FedYogi 0.945± 0.005 0.912 ± 0.009 0.919 ± 0.010
Table 6: Comparison on non-i.i.d clients of the mean and the standard deviation
of the aggregated accuracy in the last five rounds of the FL scheme. Aggrega-
tion performed without using DP and with global-DP and metric-privacy with
noise multiplier of 0.01.
From Table 6 we can see that the results with metric-privacy
are again better than those obtained with classic global-DP in
the 6 cases analyzed depending on the aggregation function.
Moreover, as we have already seen in the case of homogeneous
clients, we can observe that in the case of FedProx strategy, the
mean accuracy with metric-privacy is even better than without
adding DP. Although again this may be due to a reduction of
overfitting for better adjustment of the underrepresented class,
we can highlight by observing the standard deviation, that in
the best case without DP we would have an accuracy of 0.951,
while in the best case with metric-privacy it would be 0.946.
Therefore, this improvement is due to fluctuations in the over-
all aggregated accuracy in the last 5 rounds, as can be seen in
Figure B.3 of Appendix B.
In addition, Table 7 shows the results obtained with the six
aggregation strategies and the three paradigms (no DP or vanilla
FL, global-DP and metric-privacy) on the test dataset. Specif-
ically, this table shows the model performance in this test set
as a function of the accuracy, F1-score and precision obtained
with the aggregated model obtained after 20 rounds of FL train-
ing. In this case, again the trend is maintained: metric-privacy
improves the global-DP results, in general. Metric privacy is
better in all cases with respect to accuracy and precision and
in 5 out of 6 with respect to F1-score (only slightly worse with
FedAvg). Furthermore, with FedProx we see the anomaly al-
ready commented (i.e. the precision and F1-score obtained with
metric-privacy is better than without DP).
Finally, for further analysis, Figure B.3 in Appendix B
shows the evolution of the aggregated accuracy in the training
set of the di fferent clients in this scenario of non-i.i.d. clients
with the three proposed privacy approaches. In addition Fig-
ure C.5 in Appendix C shows the value for the AUC and the
ROC curve obtained in each case for the test set.
7.4. Impact of a client inference attack
In the last two sections we have seen that by including metric-
privacy in the FL architecture, from the server side, we can im-
prove the accuracy of the model, as we are better tuning the
amount of noise added. However, we are interested in analyz-
ing whether adding metric-privacy can contribute to reduce the
risk of a client inference attack (following the definition pro-
posed in Section 5) in a similar way as with global-DP.
Specifically, as already stated, in this case our assumption is
that a client can act as an attacker to infer whether certain client
is participating in the training, and thus learn information about
9

Vanilla FL Global-DP Metric-privacy
Strategy Accuracy F1-Score Precision Accuracy F1-Score Precision Accuracy F1-Score Precision
FedAvg 0.919 0.930 0.956 0.867 0.787 0.886 0.925 0.914 0.931
FedAvgM 0.909 0.862 0.944 0.784 0.573 0.594 0.837 0.621 0.640
FedMedian 0.933 0.917 0.961 0.859 0.626 0.641 0.917 0.926 0.940
FedProx 0.920 0.910 0.950 0.886 0.871 0.904 0.903 0.890 0.909
FedOpt 0.936 0.927 0.957 0.922 0.913 0.960 0.916 0.885 0.889
FedYogi 0.927 0.870 0.950 0.900 0.845 0.836 0.895 0.847 0.865
Table 7: Accuracy, F1-score and precision obtained in the test set with each approach. Non-i.i.d clients.
it. For example, in this medical imaging case, an attacker might
want to infer whether a certain hospital is participating in the
training, in order to learn about the hospital’s data distribution
and thus sensitive information about the hospital’s patients and
their pathologies.
In this sense, we can intuitively expect that the attack will
be more effective in the first round. Although an attacker would
use additional information to try to extract additional insights
throughout the di fferent rounds, it will be in the first round
(when the model has not yet converged) when it will be pos-
sible to infer more knowledge. As a reminder, we assume that
the clients participating in the federated training are the same
throughout all rounds.
Thus, we are going to simulate a new scenario. In this case,
for the sake of simplicity, we will take 3 clients, one of which
will be the attacker who wants to know whether or not another
client is participating in the training. In addition, this client on
which the attack will be performed (target) will have an anoma-
lous data distribution compared to the distribution on the other
two clients.
In Table 8 the distribution of the clients simulated for this
experiment is shown, with client 1 being the attacker and client
3 being the target. Note that in this case, in order to get a better
fit to the training data of each client, we are going to train the
model for 20 epochs. Then, at the end of the first round, client 1
(attacker) knows its local model as well as the global aggregated
one and some aggregated metrics sent from the server.
T otal Class 0 Class 1 Class 2 Class 3
Client 1 1747 120 9 1122 496
Client 2 1591 180 11 894 506
Client 3 1882 524 29 550 779
Table 8: Clients distribution for simulating a client inference attack.
For our approach concerning CIA from client 1 to client 3
as target, we assume that the former has a shadow dataset from
client 3, which will be a random split of 10% of client 3 train-
ing data. In addition, as the attacker participates in the train-
ing he/she will receive from the server the aggregated metrics
(accuracy and loss) obtained in each round. Then, we can ana-
lyze the loss obtained with the aggregated model for the shadow
train dataset of client 3. In Table 9 we show the results for the
cases where this client participates in the training performing
the aggregation with theFedAvgapproach (as it is the most used
one) and the three privacy approaches.
Client V anilla FL Global-DP Metric-privacy
Aggregated 1.032 3.603 1.135
Target 1.182 4.848 1.506
Difference (%) 12.719 25.679 24.631
Table 9: Loss obtained for the aggregated test set and for the shadow dataset of
client 3 with FedAvgstrategy.
Note that in Table 9 we take into account the loss instead of
other metrics such as the accuracy, as for this multi-class clas-
sification problem it is more representative, as we are dealing
with clients with quite non-i.i.d classes distributions. In view of
this example, we can note that for the FedAvg strategy we get
quite similar values for the relative di fference between the ag-
gregated result in each client’s test sets and the loss for the target
client shadow dataset with global-DP and with metric-privacy.
As we could expect, we get a slightly greater relative difference
with global-DP than with metric-privacy, but in both cases we
are protecting from the CIA attacks in a similar way (more than
in the case with no DP (vanilla FL), with a 12.719% of di ffer-
ence), with a di fference of less than 1.1% between global-DP
and metric-privacy approaches.
In Table 10 we also show the results obtained for this CIA
simulation using the FedOpt strategy, as this is the one that
provides better results in general for the previous scenarios an-
alyzed. We can observe that in this case the convergence of
the model is much better due to the aggregation strategy used,
since it has an auxiliary initial model trained with the valida-
tion dataset. Also, in this case it is worth noting that we obtain
a greater di fference with metric-privacy than with global-DP,
thus providing a greater resistance to the CIA in this particular
example. At the same time, as is shown in Table 11, the results
for the loss of the test set are also better with metric-privacy
than with global-DP.
Then, as analyzed in the previous sections, in this use case
we reach a better balance between protection and model per-
formance with metric-privacy, as in all cases analyzed in Sec-
tions 7.3.1 and 7.3.2 we obtain a significantly better model per-
formance than with global-DP. In addition, in view of the above
we get a similar protection from CIA attacks withFedAvg(slightly
worse than with global-DP), and better for the case of FedOpt
10

Client V anilla FL Global-DP Metric-privacy
Aggregated 0.595 0.743 0.756
Target 0.599 0.807 0.853
Difference (%) 0.598 7.936 11.320
Table 10: Loss obtained for the aggregated test set and for the shadow dataset
of client 3 with FedYogistrategy.
for this new scenario. For completeness of the analysis, Ap-
pendix D shows the results obtained for the other four aggre-
gation functions analyzed.
Finally, let’s see the results for the loss obtained in the test
set with these three clients trained after the first round (see Ta-
ble 11):
Test loss
Strategy Vanilla FL Global-DP Metric-privacy
FedAvg 1.010 3.514 1.106
FedAvgM 0.993 1.048 1.041
FedMedian 1.046 2.071 1.387
FedProx 1.010 1.744 1.149
FedOpt 0.645 0.845 0.823
FedYogi 0.870 1.036 0.902
Table 11: Comparison of the loss obtained for the test set in the first round.
Aggregation performed without using DP, with global-DP with noise multiplier
of 0.01 and with metric-privacy with 0.01 as noise multiplier. CIA scenario.
In view of Table 11, we can see that in all cases the loss ob-
tained with metric-privacy is lower than with global-DP, while
the protection against CIA is similar, and in some cases even
higher (e.g. with FedProx and FedAvgM, see Appendix D).
We have to note that the model is not yet converging, as only
one round of the scheme has been carried out. However, it is
important to analyze the results after this round because it is
the one from which an attacker could extract more information.
Again in this simple approach the best results are obtained with
FedOpt as aggregation strategy.
8. Conclusions and future work
In this paper we have analyzed the impact of including the
notion of metric-privacy in an FL architecture. Specifically, we
have focused on the case where a trusted server and a fixed set
of semi-honest clients are available and can act as attackers in
the network to extract information.
Additionally, we present a new type of attack, the client in-
ference attack (CIA), in which a client in the network aims to
infer whether or not a certain client is participating in the archi-
tecture. This information can serve as a basis for further attacks,
such as reconstruction, source or membership attacks.
In this sense, we propose the use of metric-privacy versus
the use of classic global-DP, so that the noise added in each
round is calibrated according to a distance metric calculated
as the maximum distance between the models of each pair of
clients. Thus, we guarantee metric-privacy in each round, tun-
ing the amount of noise added (allowing for better model per-
formance), and at the same time offer certain guarantees against
CIA.
Specifically, we analyze a use case of medical imaging in
three cases using 6 FL aggregation functions: homogeneous
clients, non-i.i.d clients and a third case with a client with un-
balanced data (target), in order to test the resistance to CIA. We
have observed that in all cases the metric-privacy approach per-
formance is significantly better than the global-DP approach,
while offering a similar resistance to CIA in most cases, and
even better in 2 out of 6 for the third case.
This work has been applied to an openly available dataset
of medical images, but future work includes its applicability to
other types of use cases to analyze the impact of the proposed
methodology, which aim to improve the performance of the FL
architectures while protecting privacy. Note that this is the first
work that we are aware of that employs metric-privacy in a fed-
erated architecture to improve privacy /utility trade-off. At the
same time it is the first one that defines the CIA and presents
an example of application and prevention with global-DP and
metric-privacy. Future work includes the application to new
case studies in different fields, as well as the analysis of di ffer-
ent CIA scenarios and the study of their impact and usefulness
as a basis for other attacks.
CRediT authorship contribution statement Judith S´ainz-Pardo
D´ıaz: Conceptualization, Formal analysis, Investigation, Method-
ology, Software, Validation, Visualization, Writing - Original
Draft. Andreas Athanasiou: Conceptualization, Investigation,
Methodology, Writing - Original Draft. Kangsoo Jung: Con-
ceptualization, Investigation, Methodology, Writing - Original
Draft. Catuscia Palamidessi: Conceptualization, Methodol-
ogy, Writing - Original Draft, Supervision. ´Alvaro L´opez Garc´ıa:
Conceptualization, Funding acquisition, Writing - Original Draft,
Supervision.
Declaration of Competing Interest
The authors declare that they have no known competing finan-
cial interests or personal relationships that could have appeared
to influence the work reported in this paper.
Data availability
The data used in this study are openly available and can be ob-
tained from [30].
Acknowledgments
Judith S ´ainz-Pardo D´ıaz and ´Alvaro L ´opez Garc´ıa would like
to thank the funding through the project AI4EOSC “Artificial
Intelligence for the European Open Science Cloud” that has re-
ceived funding from the European Union’s Horizon Europe re-
search and innovation programme under grant agreement num-
ber 101058593 and from the SIESTA project “Secure Interac-
tive Environments for Sensitive daTa Analytics”, funded by the
European Union (Horizon Europe) under grant agreement num-
ber 101131957.
The work of Andreas Athanasiou is supported by the project
11

CRYPTECS, funded by the ANR (project number ANR-20-
CY AL-0006) and by the BMBF (project number 16KIS1439).
The work of Kangsoo Jung is supported by the project ELSA,
funded by the Horizon Europe Framework (with project num-
ber 101070617). The work of Catuscia Palamidessi is sup-
ported by the project HYPATIA, funded by the ERC (grant
agreement number 835294).
References
[1] L. Li, Y . Fan, M. Tse, K.-Y . Lin, A review of applications in federated
learning, Computers & Industrial Engineering 149 (2020) 106854.
[2] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas,
Communication-efficient learning of deep networks from decentralized
data, in: Artificial intelligence and statistics, PMLR, 2017, pp. 1273–
1282.
[3] R. Shokri, M. Stronati, C. Song, V . Shmatikov, Membership Inference
Attacks Against Machine Learning Models , in: 2017 IEEE Symposium
on Security and Privacy (SP), IEEE Computer Society, Los Alamitos,
CA, USA, 2017, pp. 3–18. doi:10.1109/SP.2017.41.
URL https://doi.ieeecomputersociety.org/10.1109/SP.
2017.41
[4] H. Hu, Z. Salcic, L. Sun, G. Dobbie, X. Zhang, Source inference attacks
in federated learning, in: ICDM, IEEE, 2021.
[5] L. Zhu, Z. Liu, S. Han, Deep leakage from gradients, Advances in Neural
Information Processing Systems 32 (2019).
[6] C. Chen, L. Lyu, H. Yu, G. Chen, Practical attribute reconstruction attack
against federated learning, IEEE Transactions on Big Data (2022).
[7] J. Ma, S.-A. Naas, S. Sigg, X. Lyu, Privacy-preserving federated learn-
ing based on multi-key homomorphic encryption, International Journal
of Intelligent Systems 37 (9) (2022) 5880–5901.
[8] K. Chatzikokolakis, M. E. Andr ´es, N. E. Bordenabe, C. Palamidessi,
Broadening the scope of di fferential privacy using metrics, in:
E. De Cristofaro, M. Wright (Eds.), Privacy Enhancing Technologies,
Springer Berlin Heidelberg, Berlin, Heidelberg, 2013, pp. 82–102.
[9] M. S. Alvim, K. Chatzikokolakis, C. Palamidessi, A. Pazii, Metric-based
local differential privacy for statistical applications, in: 31st Computer
Security Foundations Symposium (CSF 2018), IEEE Computer Society,
2018, pp. 262–267.
[10] K. Pan, Y .-S. Ong, M. Gong, H. Li, A. K. Qin, Y . Gao, Differential privacy
in deep learning: A literature survey, Neurocomputing (2024) 127663.
[11] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Tal-
war, L. Zhang, Deep learning with di fferential privacy, in: Proceedings
of the 2016 ACM SIGSAC conference on computer and communications
security, 2016, pp. 308–318.
[12] T. Ha, T. K. Dang, T. T. Dang, T. A. Truong, M. T. Nguyen, Di fferential
privacy in deep learning: an overview, in: 2019 International Conference
on Advanced Computing and Applications (ACOMP), IEEE, 2019, pp.
97–102.
[13] Y . Xie, P. Li, C. Wu, Q. Wu, Di fferential privacy stochastic gradient
descent with adaptive privacy budget allocation, in: 2021 IEEE inter-
national conference on consumer electronics and computer engineering
(ICCECE), IEEE, 2021, pp. 227–231.
[14] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S.
Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow,
A. Harp, G. Irving, M. Isard, Y . Jia, R. Jozefowicz, L. Kaiser, M. Kud-
lur, J. Levenberg, D. Man ´e, R. Monga, S. Moore, D. Murray, C. Olah,
M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker,
V . Vanhoucke, V . Vasudevan, F. Vi´egas, O. Vinyals, P. Warden, M. Wat-
tenberg, M. Wicke, Y . Yu, X. Zheng, TensorFlow: Large-scale ma-
chine learning on heterogeneous systems, software available from ten-
sorflow.org (2015).
URL https://www.tensorflow.org/
[15] Opacus PyTorch library, Available from opacus.ai.
[16] M. Nasr, R. Shokri, A. Houmansadr, Comprehensive privacy analysis of
deep learning: Passive and active white-box inference attacks against cen-
tralized and federated learning, in: 2019 IEEE symposium on security and
privacy (SP), IEEE, 2019, pp. 739–753.
[17] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramer, Member-
ship inference attacks from first principles, in: 2022 IEEE Symposium on
Security and Privacy (SP), IEEE, 2022, pp. 1897–1914.
[18] H. Hu, Z. Salcic, L. Sun, G. Dobbie, P. S. Yu, X. Zhang, Membership
inference attacks on machine learning: A survey, ACM Comput. Surv.
54 (11s) (Sep. 2022). doi:10.1145/3523273.
URL https://doi.org/10.1145/3523273
[19] A. Pustozerova, J. Baumbach, R. Mayer, Di fferentially private federated
learning: Privacy and utility analysis of output perturbation and dp-sgd,
in: 2023 IEEE International Conference on Big Data (BigData), 2023, pp.
5549–5558. doi:10.1109/BigData59044.2023.10386466.
[20] M. Adnan, S. Kalra, J. C. Cresswell, G. W. Taylor, H. R. Tizhoosh, Fed-
erated learning and di fferential privacy for medical image analysis, Sci-
entific reports 12 (1) (2022) 1953.
[21] R. Hu, Y . Guo, H. Li, Q. Pei, Y . Gong, Personalized federated learning
with differential privacy, IEEE Internet of Things Journal 7 (10) (2020)
9530–9539. doi:10.1109/JIOT.2020.2991416.
[22] M. Naseri, J. Hayes, E. De Cristofaro, Local and central di fferential
privacy for robustness and privacy in federated learning, arXiv preprint
arXiv:2009.03561 (2020).
[23] T.-M. H. Hsu, H. Qi, M. Brown, Measuring the e ffects of non-identical
data distribution for federated visual classification, arXiv preprint
arXiv:1909.06335 (2019).
[24] D. J. Beutel, T. Topal, A. Mathur, X. Qiu, J. Fernandez-Marques, Y . Gao,
L. Sani, H. L. Kwing, T. Parcollet, P. P. d. Gusm ˜ao, N. D. Lane,
Flower: A friendly federated learning research framework, arXiv preprint
arXiv:2007.14390 (2020).
[25] D. Yin, Y . Chen, R. Kannan, P. Bartlett, Byzantine-robust distributed
learning: Towards optimal statistical rates, in: International conference
on machine learning, Pmlr, 2018, pp. 5650–5659.
[26] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V . Smith, Fed-
erated optimization in heterogeneous networks, Proceedings of Machine
learning and systems 2 (2020) 429–450.
[27] S. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Kone ˇcn`y, S. Ku-
mar, H. B. McMahan, Adaptive federated optimization, arXiv preprint
arXiv:2003.00295 (2020).
[28] M. S. Alvim, N. Fernandes, A. McIver, C. Morgan, G. H. Nunes, A
novel analysis of utility in privacy pipelines, using kronecker products
and quantitative information flow, in: Proceedings of the 2023 ACM
SIGSAC Conference on Computer and Communications Security, CCS
’23, Association for Computing Machinery, New York, NY , USA, 2023,
p. 1718–1731. doi:10.1145/3576915.3623081.
URL https://doi.org/10.1145/3576915.3623081
[29] F. Galli, S. Biswas, K. Jung, T. Cucinotta, C. Palamidessi, Group privacy
for personalized federated learning, in: Proceedings of the 9th Interna-
tional Conference on Information Systems Security and Privacy-ICISSP,
SCITEPRESS-Science and Technology Publications, 2023, pp. 252–263.
[30] Falah.G.Salieh, Alzheimer mri dataset (2023).
URL https://huggingface.co/datasets/Falah/Alzheimer_
MRI
[31] M. Hatami, F. Yaghmaee, R. Ebrahimpour, Investigating the potential of
reinforcement learning and deep learning in improving alzheimer’s dis-
ease classification, Neurocomputing 597 (2024) 128119. doi:https:
//doi.org/10.1016/j.neucom.2024.128119.
URL https://www.sciencedirect.com/science/article/pii/
S0925231224008907
[32] R. C. Geyer, T. Klein, M. Nabi, Di fferentially private federated learning:
A client level perspective, arXiv preprint arXiv:1712.07557 (2017).
[33] H. B. McMahan, D. Ramage, K. Talwar, L. Zhang, Learning di fferen-
tially private recurrent language models, arXiv preprint arXiv:1710.06963
(2017).
12

Appendix A. Analysis of the performance of the trained model in di fferent runs
In this section we show the mean accuracy obtained for the three approaches (vanilla FL, global-DP and metric-privacy) with
the different aggregation strategies for the case of homogeneous clients. The results shown in Table A.12 correspond with the mean
and standard deviation of five runs of the models evaluated in the test dataset (90 models are evaluated in total, five runs for each
strategy and privacy approach). Our idea is to evaluate the statistical significance of the trained model.
Mean accuracy± std (5 runs)
Strategy Vanilla FL Global-DP Metric-privacy
FedAvg 0.930± 0.011 0.881 ± 0.013 0.913 ± 0.019
FedAvgM 0.880± 0.004 0.781 ± 0.021 0.812 ± 0.028
FedMedian 0.931± 0.000 0.867 ± 0.013 0.897 ± 0.004
FedProx 0.923± 0.011 0.880 ± 0.014 0.919 ± 0.011
FedOpt 0.947± 0.005 0.912 ± 0.010 0.924 ± 0.010
FedYogi 0.939± 0.008 0.899 ± 0.007 0.895 ± 0.013
Table A.12: Mean and standard deviation of the accuracy obtained for the test set in 5 different runs of the model. Homogeneous clients,
Note that the interest of running the experiments multiple times is from the point of view of the two privacy approaches (that add
a component of randomness, as well as other strategies that use an initial model). In all the cases a seed is fixed in order to start with
the same model in all the clients (we see that withFedMedian without DP the std is 0). In view of the standard deviation obtained in
these 90 experiments (5 for each approach and aggregation function), for the other scenarios (non-i.i.d and client inference attacks),
we show the results obtained for a fixed seed, for a better analysis of the results and for a better fit to a real use case where the
model is run once the hyperparameter optimization has been completed and then it is evaluated on a new test dataset. We can note
than in mean all the results with metric-privacy are better than with global-DP for the test set except for FedYogi. However, if we
take the upper bound taken into account the std, metric-privacy even performs slightly better than global-DP also in this case.
Appendix B. Evolution of the aggregated accuracy in the clients’ test set
In Figure B.2 the aggregated accuracy for each client’s test set in each round of the FL scheme is shown for the case of
homogeneous clients. Here we can note that the models training with metric-privacy from the server side reach a convergence more
similar and softer to the model aggregated without DP that the one aggregated applying global-DP (which shows more fluctuations
over the course of the rounds). It should be noted that in the three cases the trends of the rounds without DP, with global-DP and
with metric-privacy (in the figure referred as MDP) are similar, with FedOpt being the most accurate, followed by FedYogi(which
is more clearly observed with global-DP), and ending in all cases with FedMedian and FedAvgM as the least reliable.
Figure B.2: Evolution of the aggregated accuracy in each round of the FL architecture. Homogeneous clients.
Then, Figure B.3 shows the results obtained for the aggregated accuracy in each round for the case of non-i.i.d clients. Again
in this case the curves are smoother with metric-privacy than with global-DP, due to the amount of noise added. However, it is
interesting to note that with FedAvg, in this scenario, better results are obtained in the first round, which may be counter-intuitive,
but may be due to the fact that the aggregated results are shown and the client with more data (client 2) has more weight than the
other clients in these aggregated results, as well as a greater weight in the aggregation of the model with this strategy. It is interesting
to note that in all three cases the same trend is maintained, with FedOpt being the best performing strategy, while FedAvgM is the
least accurate.
13

Figure B.3: Evolution of the aggregated accuracy in each round of the FL architecture. Non-i.i.d clients.
Appendix C. ROC curves and AUC in the test set
Figure C.4 shows the ROC curves and the AUC obtained with each aggregation strategy for the case of homogeneous clients
and with the three approaches (no-DP or vanilla FL, global-DP, metric-privacy).
Figure C.4: ROC curves and AUC obtained in the client test set with each strategy and DP approach. Homogeneous clients.
Figure C.5: ROC curves and AUC obtained in the client test set with each strategy and DP approach. Non-i.i.d clients.
In view of Figure C.4, we can see that the best results for the scenario of homogeneous clients are reached without DP in 5 out of
6 cases (with FedAvgthe results are slightly better with metric-privacy, as explained in Section 7.3.1). In addition, the ones obtained
with metric-privacy are always better than those with global-DP. The former is also fulfilled for the case of non-i.i.d. clients as is
shown in Figure C.5, but in this case the best results are always obtained without DP.
Note than in order to get the ROC curves and to calculate the AUC in this case of multi-class classification, the micro-averaged
one-vs-rest ROC-AUC score has been used, comparing each class against the other three in the labels.
14

Appendix D. Client inference attack for di fferent aggregation strategies
In Section 7.4 we have shown the performance obtained for the scenario simulated concerning the client inference attack
performed using both FedAvgand FedOpt as aggregation strategies. For completeness of the analysis, the results for the other four
aggregation functions are shown in this appendix.
We can note that fot the case of FedAvgM, metric-privacy provides a greater di fference that global-DP, while it also provides
better results in terms of the loss. This may be due for a reduction in the overfitting to the target client. On the other hand, for
the other three aggregation functions, a better result in terms of the distance is obtained with global-DP, but there are still better
with metric-privacy that without DP. This shows that metric-privacy helps to prevent client inference attacks while allowing a better
convergence of the model than classic global-DP.
Client V anilla FL Global-DP Metric-privacy
Aggregated 0.897 0.969 0.945
Target 1.118 1.115 1.211
Difference (%) 19.782 13.105 21.965
Table D.13: Loss obtained for the aggregated test set and for the shadow
dataset of client 3 with FedAvgM strategy.
Client V anilla FL Global-DP Metric-privacy
Aggregated 1.068 2.079 1.393
Target 1.188 2.911 1.880
Difference (%) 10.129 28.587 25.888
Table D.14: Loss obtained for the aggregated test set and for the shadow
dataset of client 3 with FedMedian strategy.
Client V anilla FL Global-DP Metric-privacy
Aggregated 1.032 1.846 1.163
Target 1.182 2.681 1.548
Difference (%) 12.719 31.173 24.820
Table D.15: Loss obtained for the aggregated test set and for the shadow
dataset of client 3 with FedProx strategy.
Client V anilla FL Global-DP Metric-privacy
Aggregated 0.795 0.949 0.815
Target 0.895 1.170 0.957
Difference (%) 11.150 18.861 14.888
Table D.16: Loss obtained for the aggregated test set and for the shadow
dataset of client 3 with FedYogistrategy.
Appendix E. Code reproducibility
For carrying out the experiments conducted in this work, the following Python libraries and versions with their corresponding
dependencies have been used (only the main ones are detailed): flower (v1.13.0), TensorFlow (v2.14.0), keras (v2.14.0), scikit-
learn (v1.5.2), scipy (v1.14.1), tqdm (v4.66.5), opencv-python (v4.10.0.84), numpy (v1.26.0), pandas (v2.2.3), pillow (v11.0.0) and
matplotlib (v3.9.2).
Appendix F. Including server side metric-Di fferential Privacy in Flower
In this section we explain further details concerning the code implemented and the software used for carrying out the Federated
Learning and for including both global-DP and metric-privacy from the server side. Note that for performing the FL scenario we
have used the Flower Python library [24]. This library allows to include DP in a FL workflow (using [32] and [33] as references)
using different wrappers, both from the client and server side. From the server side, it implements a wrapper that can be applied to
each aggregation strategy for global-DP with server-side fixed and adaptive clipping. In the study conducted in this work we have
applied fixed clipping. The Gaussian mechanism is used in the class DifferentialPrivacyServerSideFixedClipping, which receives
as input the noise multiplier, the clipping norm and the number of clients samples, as explained in Section 6. Usually a value greater
than 1 is recommended for the noise multiplier, but in our use case we analyzed different values in order to select one that allow the
model to converge. In particular, with the value 0.01 we achieved convergence and at the same time it allows us to protect against
the client inference attacks that we are trying to prevent (as shown in Section 7.4).
To add the notion of metric DP, we have modified the code of the class DifferentialPrivacyServerSideFixedClipping if Flower
in such a way that a new function (nameddistance metric) is added for dynamically calculate in each round the maximum distance
between the different pairs of models as explained in Section 6. This function receives the results containing a list with the tuple
(ClientProxy, FitRes).
Specifically, within the functionaggregate fit, a call todistance metric) has been included to calculate the distance once the clip-
ping is performed, and then this distance is entered by dividing the noise multiplier in the call to theadd gaussian noise to params
and compute stdv functions for both adding the noise and calculating the standard deviation of the mechanism applied.
These modifications have been done in the file dp fixed clipping.py that is located under the following structure:
15

flower
...
datasets
dev
doc
e2e
examples
glossary
src
cc/flwr
docker
kotlin
proto/flwr/proto
py
flwr
cli
client
common
proto
server
compat
driver
serverapp
strategy
...
dp fixed clipping.py
...
...
...
flwr tool
swift/flwr
...
16