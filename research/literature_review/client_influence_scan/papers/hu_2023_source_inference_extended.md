1
Source Inference Attacks: Beyond Membership
Inference Attacks in Federated Learning
Hongsheng Hu∗, Xuyun Zhang†, Zoran Salcic∗, Life Senior Member, IEEE, Lichao Sun‡,
Kim-Kwang Raymond Choo§, Senior Member, IEEE, and Gillian Dobbie∗
∗The University of Auckland, New Zealand
†Macquarie University, Australia
‡Lehigh University, USA
§The University of Texas at San Antonio, USA
Abstract—Federated learning (FL) is a popular approach to facilitate privacy-aware machine learning since it allows multiple clients to
collaboratively train a global model without granting others access to their private data. It is, however, known that FL can be vulnerable
to membership inference attacks (MIAs), where the training records of the global model can be distinguished from the testing records.
Surprisingly, research focusing on the investigation of the source inference problem appears to be lacking. We also observe that
identifying a training record’s source client can result in privacy breaches extending beyond MIAs. For example, consider an FL
application where multiple hospitals jointly train a COVID-19 diagnosis model, membership inference attackers can identify the medical
records that have been used for training, and any additional identification of the source hospital can result the patient from the
particular hospital more prone to discrimination. Seeking to contribute to the literature gap, we take the first step to investigate source
privacy in FL. Specifically, we propose a new inference attack (hereafter referred to assource inference attack – SIA), designed to
facilitate an honest-but-curious server to identify the training record’s source client. The proposed SIAs leverage the Bayesian theorem
to allow the server to implement the attack in a non-intrusive manner without deviating from the defined FL protocol. We then evaluate
SIAs in three different FL frameworks to show that in existing FL frameworks, the clients sharing gradients, model parameters, or
predictions on a public dataset will leak such source information to the server. We also conduct extensive experiments on various
datasets to investigate the key factors in an SIA. The experimental results validate the efficacy of the proposed SIAs, e.g., an attack
success rate of 67.1% (baseline 10%) can be achieved when the clients share model parameters with the server. Comprehensive
ablation studies demonstrate that the success of an SIA is directly related to the overfitting of the local models.
Index Terms—Federated Learning, Membership Inference Attacks, Source Inference Attacks, Privacy Leakage.
✦
1 I NTRODUCTION
Recent deep learning advances have partly contributed to
the building of powerful machine learning (ML) models
from large datasets. In practice, however, data often resides
across different organizational entities (also referred to as
data islands). The data records of a single entity, perhaps
with the exception of extremely large technology organiza-
tions, are generally limited and do not represent the entire
data distribution. Thus, stakeholders (e.g., consumers) can
generally benefit if different data owners can collaboratively
train a joint machine learning model based on the union
of different datasets. For example, our society will benefit
if different countries and organizations can collaborate to
collaboratively train COVID-19 diagnosis models using the
broad range of medical data records in these different en-
tities. The exacting privacy regulations (e.g., GDPR [1] in
the European Union and CCPA [2] in the United States),
however, complicate such collaborative efforts.
Federated learning (FL) is one approach that can be
utilized to circumvent the limitations due to data islands,
by allowing multiple clients coordinated by a central server
to train a joint ML model in an iterative manner [3], [4], [5],
[6]. In FL, the clients send their model updates to the server
†Xuyun Zhang is the corresponding author.
but never their raw training dataset, thereby leading to a
privacy-aware paradigm for collaborative model training.
For the example mentioned above, FL can greatly facilitate
the hospitals wishing to train a joint COVID-19 diagnosis
model from the distributed data in different hospitals. A
real-life case in [7] has shown the successful adoption of
FL where an ML model for COVID-19 diagnosis has been
trained with the usage of the geographically distributed
chest computed tomography data collected from different
patients at different hospitals.
However, many recent studies [8], [9], [10], [11], [12],
[13] have shown that FL does not provide sufficient privacy
guarantees, because sensitive information from the training
data can still be revealed during the communication process.
In FL, the clients transmit necessary information from up-
dates, e.g., gradients, to the central server for global model
training. Because the updates are derived from the clients’
private training data, there are several recently proposed
privacy attacks trying to infer the privacy of the clients from
such updates, such as data reconstruction attacks [14], prop-
erty inference attacks [15], preference profiling attacks [13],
and membership inference attacks (MIAs) [16]. Among such
attacks, MIAs aim to identify whether or not a data record
was in the training dataset of a target model (i.e., a member).
While an MIA seems like a simple attack, it can impose
arXiv:2310.00222v1  [cs.CR]  30 Sep 2023

2
severe privacy risks in many settings where knowing that
someone was in a dataset is harmful [17]. For example, by
identifying the fact that a clinical record was in the training
dataset of a medical model associated with a certain disease,
MIAs enable an attacker to know that the owner of the
clinical record has a high chance of having the disease.
In FL, the training data of the FL system consist of all the
training records in the datasets of the clients because they
collaboratively build a global federated model. Thus, MIAs
of existing research in the context of FL [8], [9], [18], [19]
are designed to distinguish the training records from the
testing records of the global model, i.e., they do not require
to identify which client owns a training record, i.e., the
source of the training record. However, it is important and
practical to explore source privacy in FL beyond member-
ship privacy because the leakage of the source information
can further breach privacy. For example, in the previously
mentioned FL application where multiple hospitals jointly
train a COVID-19 diagnosis model, MIAs can only tell who
has had a COVID-19 test, but the further identification of the
source hospital of the people could make them more prone
to discrimination, especially when the hospital is located in
a high-risk region or country [20]. Other types of attacks in
FL such as property inference attacks also fail to explore the
source privacy of clients because they are either designed
to infer other types of private information or the private
information inferred does not attribute to a specific client.
For example, property inference attacks in FL [8] can infer
a certain property in the training data but can not attribute
the property to the client who owns this property.
This paper proposes a novel inference attack, named
Source Inference Attack (SIA), that determines which client
in FL owns a training record. The SIA can be considered
a stronger attack based on the foundation of MIAs, i.e.,
after identifying which data records are training records
via MIAs, the attacker further implements SIAs to de-
termine which client a training record comes from. For
practical reasons, we consider the server can be a semi-
honest (honest-but-curious) attacker who passively tries to
learn the secret based on the information on the identities
of clients and communications between them. Specifically,
the semi-honest server means that the server tries to infer
the private information of the clients without interfering
with the federated training, i.e., without deviating from the
defined FL protocol, which is the main challenge of SIAs.
Note that the attacker can be one of the clients, but we
argue that it is impractical in this case for SIAs because the
client does not know the identities of the other clients, and
it can only access the global model via communications to
the server [21].
We leverage the Bayesian theorem to analyze how to
effectively implement SIAs in FL. We demonstrate that
an honest-but-curious server can estimate the source of
a training record in an SIA by leveraging the prediction
loss of the local models. More specifically, we theoretically
demonstrate that the client with the smallest prediction loss
on the training record should have the highest probability
of owning it. To demonstrate the feasibility of the pro-
posed SIAs to different FL frameworks, we propose three
FL-S IA frameworks that enable the server to conduct the
SIAs in three FL frameworks, FedSGD [3], FedAvg [3],
and FedMD [22]. The purpose of selecting the three FL
frameworks is to show that in existing FL frameworks, the
clients sharing gradients, model parameters, or predictions
on a public dataset will lead to source information leakage to
the server. We conduct extensive experiments on six datasets
and different model architectures under various FL settings
to evaluate the effectiveness of SIAs. The experiment results
validate the efficacy of our proposed SIAs. We conduct a
detailed ablation study to investigate how data distributions
across the clients and the number of local epochs in FL affect
the performance of an SIA. An important finding is that the
success of an SIA is directly related to the overfitting of the
local models, which is mainly caused by the non-IID data
distribution across the clients.
The main contributions of this paper are three-folds:
• We propose a novel inference attack in federated
learning (FL), named as source inference attack (SIA),
which infers the source client of a training record. Be-
yond membership inference attacks, SIAs can further
breach the privacy of the training records in FL.
• We innovatively adopt the Bayesian theorem to an-
alyze how an honest-but-curious server can imple-
ment SIAs in a non-intrusive manner to infer the
source of a training record with the highest proba-
bility by using the prediction loss of local models.
• We show the feasibility and effectiveness of SIAs in
three FL frameworks, including FedSGD, FedAvg,
and FedMD. We perform extensive experiments to
empirically evaluate SIAs in the three frameworks
with various datasets and under different FL set-
tings. The results validate the efficacy of the pro-
posed SIA. Our proposed SIAs shed new light on
how FL reveals sensitive information and the need
to build more private FL frameworks.
This work extends our earlier conference paper [23]. In
Section 2, we introduce additional background material to
give the readers have a broader understanding of the role
and impact of SIAs in FL. In Section 3, we extend the earlier
work in the following ways:
• We use a threat model to describe the attack goal,
target FL systems, attacker, and attack knowledge.
• We provide systematic theoretical analysis and pro-
pose theorems to show how to leverage the predic-
tion loss of the models for conducting SIAs in FL. For
the corresponding theorems, we also present detailed
corresponding proofs.
• We show how to implement SIAs in two other
commonly-used FL frameworks, FedSGD and
FedMD, while in [23] we only introduced the imple-
mentation of SIAs in FedAvg. This extension demon-
strates the broader applicability of SIAs in FL.
In Section 4, we describe our comprehensive experi-
mental evaluation, including the addition in Section 4.6 to
explain why the proposed SIAs can work in FL. In Section 5,
we provide more comprehensive experiments to investigate
whether the popular defense method of differential privacy
can mitigate SIAs. Moreover, in Section 5, we discuss the
limitations and potential research opportunities of our pro-
posed SIAs. In Section 6, we also include additional related

3
work to give the readers a broader picture of privacy attacks
and defenses in FL. Source code for implementing SIAs in
FedSGD, FedAvg, and FedMD are also included 1, while in
[23] we only provide code for SIAs in FedAvg.
2 P RELIMINARIES
This section reviews the background of federated learning
and membership inference attacks.
2.1 Federated Learning
Because data usually exists in the form of isolated islands
and central storage is impractical due to privacy regulations
and laws, federated learning (FL) has been proposed to
allow multiple clients to collaboratively train a machine
learning model in an interactive manner. During the training
phase of FL, the clients send necessary information of the
updates but never their private datasets to the central server.
Because the updates contain less information than the raw
training data, FL has obvious privacy advantages compared
to data center training [3].
Horizontal and vertical FL. Based on the feature space
or the sample ID space the local datasets share, FL can
be divided into horizontal FL and vertical FL [24], [25].
Horizontal FL, aka. cross-device FL, describes FL scenarios
where the local datasets share the same feature space but
are different in samples. An example of horizontal FL is
multiple regional banks having different users from their
respective regions, while the feature spaces of such users
are the same because the banks have very similar businesses
[26]. Vertical FL, aka. cross-silo FL, describes FL scenarios
where the local datasets share the same or similar sample ID
space but differ in feature space. An example of vertical FL is
two different commercial companies in the same city having
the same or very similar customers in the area. However,
due to different business modes, the commercial company
of the bank has the user’s revenue and expenditure transac-
tions, while the commercial company of e-commerce owns
the user’s browsing and purchasing history. Vertical FL
enables the two different companies to jointly build a model
for predicting users’ living behaviors [24].
Homogeneous and heterogeneous FL. Based on architec-
tures, FL frameworks can be divided into two categories,
i.e., FL with a homogeneous architecture and FL with a
heterogeneous architecture [27]. In FL with a homogeneous
architecture, the local models have the same architecture
as the global model, and there are two forms of FL [3]: i)
FedSGD, in which each of the clients sends gradients calcu-
lated on its local data to the server; ii) FedAvg, in which each
of the clients sends the calculated local models’ parameters
to the server. FedSGD has the advantage of convergence
guarantees of the global model but requires frequent com-
munication between the clients and the server [28]. FedAvg
is more communication efficient but the global model may
not perform well when the training data across the clients
are highly non-identically distributed [29]. In FL with a
heterogeneous architecture, the local models do not have
to share the same architecture as the global model, while
1. https://github.com/HongshengHu/SIAs-Beyond MIAs in
Federated Learning
each client can still benefit during the federated training
process. FedMD [22] (Federated Model Distillation) is a
novel FL framework with a heterogeneous architecture,
which shares the knowledge of each client’s local model via
their predictions on an unlabeled public dataset instead of
the local model’s parameters. Compared to FedAvg, FedMD
eliminates the risk of white-box inference attacks [8] and has
the advantage of reduced communication costs.
Note that there are many other FL frameworks such as
FedProx [29], SCAFFOLD [30], FedDF [31], and Cronus [32]
that are proposed to solve different challenges in FL [25],
[28], [33]. These frameworks can be divided into the cat-
egories of homogeneous and heterogeneous FL we intro-
duced above based on their architectures. They differ from
the FL frameworks of FedSGD, FedAvg, and FedMD in how
the local models or the global model is trained, but the
information exchange between the clients and the server is
the same as the three FL frameworks, i.e., the clients sharing
gradients, model parameters, or predictions on an unlabeled
dataset to the server. In this paper, we show the effectiveness
of SIAs in the three FL frameworks of FedSGD, FedAvg, and
FedMD to demonstrate that an honest-but-curious server in
FL can infer source information of the training records, no
matter what kind of updates are uploaded by the clients.
But it is worth noting that SIAs are also effective in other
FL frameworks because their communication exchange be-
tween the clients and the server is the same as the FL
frameworks we evaluated in this paper.
2.2 Membership Inference Attacks
Membership inference attacks (MIAs) aim to identify
whether or not a data record was in the training dataset
of a target model. Although an MIA seems like a simple
attack, it can directly lead to severe privacy breaches of
individuals. For instance, identifying that a patient’s clinical
record was used to train a model associated with a disease
can reveal that the patient has this disease with a high
chance. Although FL has emerged as a popular privacy-
aware learning paradigm, recent works [8], [9], [18], [34],
[35], [36], [37] have demonstrated the success of MIAs on FL
models. For instance, Melis et al. [8] show that a malicious
client in FL can infer whether or not a location profile was
in the FourSquare dataset that was used to train the global
model. In FL, because the training dataset of the FL system
consists of all the local training data records, the existing
research of MIAs are designed to infer whether or not a data
record was used to train the global model, but not to identify
whether a data record was used to train a local model [19].
Currently, there are no attacks in FL to explore which
client (i.e., the source) owns the training records identified
by MIAs, while it is important and practical to explore the
source information of the training records. For instance, in a
promising FL application where multiple hospitals jointly
train a COVID-19 model for COVID-19 diagnosis, an at-
tacker can implement MIAs to infer who has been tested for
COVID-19, but further identification of the source hospital
where the people are from will make them more prone to
discrimination, especially when the hospital is in a high-
risk region or country [20]. In this paper, we propose SIAs
to show the feasibility of breaching the source privacy of the

4
training records in FL. Our proposed SIAs shed new light
on how FL reveals sensitive information and the necessity
to build more private FL frameworks.
3 S OURCE INFERENCE ATTACKS
In this section, we first introduce the threat model. Then, we
show how to leverage the Bayesian theorem to analyze how
the attacker can perform SIAs based on the prediction loss
of the local models.
3.1 Threat Model
Target FL systems. As this is the first paper that investigates
the source privacy of the training records in the FL system
and to avoid the vague definition of SIAs, we consider SIAs
on horizontal FL (see Section 2.1 for detailed introduction
of horizontal FL) where there is only one source client for
one data record. While in vertical FL, one data record can
correspond to multiple source clients, and SIAs under this
setting seems more interesting, we leave the investigation of
SIAs in vertical FL for our future work.
We consider the server can be a semi-honest attacker
who passively tries to learn the secret data from the com-
munication updates uploaded by the local clients. During
the attack process, the server follows the training protocol
of the FL system but will passively try to learn the secret
by mounting the SIAs. The server can acquire the communi-
cation updates uploaded by the local clients. Finally, based
on the communication updates, the server can acquire the
source information. There is a possibility that the attacker
can be one of the clients. However, because local clients
in FL can only observe the global model parameters while
having little information about the identities of other clients
[21], it is almost impossible for a local client to achieve the
attack goal (detailed in the following paragraph).
During the attack process, the local clients follow the
training protocol of the FL system. Specifically, the local
clients download the global aggregation results calculated
by the server and then perform local model updates. After
that, the clients upload the necessary information of updates
to the server for aggregation.
Attack goal. The attacker of SIAs in FL aims to identify
which client owns a training record that participants in
the federated training process. The goal of this attack is
motivated by FL applications where source information is
sensitive. A motivating example is the FL application of
multiple hospitals training a COVID-19 diagnosis model
(see Section 1 and Section 2.2). Another example is the
FL application of multiple users collaboratively training an
image classification model. If an attacker can identify which
user owns a sensitive image, the attacker can directly obtain
the user’s privacy based on the sensitive content of that
image [8].
Attack knowledge. We consider the attacker of the central
server is honest-but-curious: The central server will not devi-
ate from the defined FL protocol but will attempt to infer
the source information from legitimately received informa-
tion from the local clients. Specifically, the central server
implements SIAs to infer the source information of the
clients based on the received gradients, model parameters,
or predictions on an unlabeled dataset from the local clients.
However, the central server will not actively manipulate
these updates from the local clients and thus without af-
fecting the utilities of the FL model.
Because SIAs are considered as further attacks based on
the foundation of MIAs, we follow the similar setting in
the literature of MIAs [16], [19], [38] that the attacker is
given a data record we called a target record, and it has been
identified as a training record by MIAs. Note that we do not
focus on how the attacker obtains the training record from
the clients but focus on investigating the potential source
privacy leakage of the training record, while one can refer
to data reconstruction attacks [10], [14], [28], [39], [40] to see
how an attacker in FL can reconstruct the training data. An
SIA is said to succeed if the attacker can correctly identify
which local client the target record comes from.
3.2 Source Inference Attack Method
In this paper, we focus the horizontal FL on classification
tasks. Let Dtrain = {D1,··· ,DK} (assuming there are K
clients) be the training dataset of the FL system, where each
Di corresponds to the local training dataset of the client
i. We assume there are n data records z1,··· ,zn in Dtrain.
Each data record is represented asz = (x,y ), wherex is the
feature andy is the class label.
Source status. In horizontal FL, each target record exists
in the local dataset of one client. Thus, we can use a K-
dimensional multinomial vector s to represent the source
status of each target record. In s, the k-th element equals 1
representing the target record belongs to the client k, while
all the remaining elements equal 0. For example, assuming
there are six clients in FL and the target record z comes
from the second client. Then, the multinomial variable s is
represented bys = [0, 1, 0, 0, 0, 0]T.
We assume the target record zi comes from the client
k with the probability λ, i.e., the probability of the k-th
element in si equals 1 is λ, denoted as P(sik = 1) = λ.
Without loss of generality, we take the case of z1 to study
the source inference problem, which is defined as follows.
Definition 1 (Source Inference). Given a local modelθk and a
target recordz1, source inference onz1 aims to infer the posterior
probability ofz1 belonging to the client k:
S(θk,z1) := P(s1k = 1|θk,z1). (1)
For source inference by Definition 1, we aim to derive an
explicit formula for S(θk,z1) from the Bayesian perspec-
tive, which can provide insights on how to leverage the
prediction loss of the local clients for implementing SIAs.
We denote τ ={z2,··· ,zn,s2,··· ,sn} as the set which
includes the remaining training records and their source
status. The explicit formula of S(θk,z1) is given by the
following theorem.
Theorem 1. Given a local model θk and a target record z1, the
source inference is given by:
S(θk,z1) = Eτ

σ

log( P(θk|s1k = 1,z1,τ )
P(θk|s1k = 0,z1,τ ) ) +µλ

,
(2)
where µλ = log( λ
1−λ ), and σ(·) is a sigmoid function
defined asσ(x) = (1 + e−x)−1.

5
Proof. Applying the law of total expectation [41], we have:
S(θk,z1) = P(s1k = 1|θk,z1)
= Eτ [P(s1k = 1|θk,z1,τ )]. (3)
Applying the Bayes’ formula, we have:
P(s1k = 1|θk,z1,τ ) = P(θk|s1k = 1,z1,τ )P(s1k = 1|z1,τ )
P(θk|z1,τ ) .
(4)
As the source variables s1,··· ,sn are independent, event
s1k = 1 is independent fromz1,τ . Thus, we have:
P(s1k = 1|z1,τ ) = P(s1k = 1). (5)
Let:
ϕ := P(θk|s1k = 1,z1,τ )P(s1k = 1). (6)
ω := P(θk|s1k = 0,z1,τ )P(s1k = 0). (7)
Plugging eqn. (5), eqn. (6), and eqn. (7) into eqn. (4), we
have:
P(s1k = 1|θk,z1,τ ) = P(θk|s1k = 1,z1,τ )P(s1k = 1)
P(θk|z1,τ )
= ϕ
ϕ +ω
=σ

log
ϕ
ω

.
(8)
Given that P(sik = 1) = λ, we have:
log
ϕ
ω

= log
 P(θk|s1k = 1,z1,τ )
P(θk|s1k = 0,z1,τ )

+ log
 λ
1−λ

.
(9)
Letµλ = log( λ
1−λ ), we have:
S(θk,z1) = Eτ [P(s1k = 1|θk,z1,τ )]
= Eτ

σ

log
ϕ
ω

= Eτ

σ

log( P(θk|s1k = 1,z1,τ )
P(θk|s1k = 0,z1,τ ) +µλ

,
(10)
which concludes the proof.
We observe that Theorem 1 does not have the loss ℓ(·)
form and only relies on the posterior parameterθk in expec-
tation given{z1,··· ,zn,s1,··· ,sn} is a random variable.
To make S(θk,z1) more explicit with the loss term, we
assume an ML algorithm produced parameters θ follows
a posterior distribution. Specifically, following the assump-
tion in the previous work [42], we assume the posterior
distribution of an ML model θ as follows:
p(θ|z1,··· ,zn)∝e− 1
γ
Pn
i=1ℓ(θ,zi), (11)
where γ is a temperature parameter which controls
the stochasticity of θ. Following this assumption, given
{z1,··· ,zn,s1,··· ,sn}, the posterior distribution of θk
follows:
p(θk|z1,··· ,zn,s1,··· ,sn)∝e− 1
γ
Pn
i=1sikℓ(θk,zi). (12)
We further define the posterior distribution of θk
given training records z2,··· ,zn and their source status
s2,··· ,sn :
pτ (θk) := e− 1
γ
Pn
i=2sikℓ(θk,zi)
R
te− 1
γ
Pn
i=2sikℓ(t,zi)dt
, (13)
where the denominator is a constant value. The following
theorem explicitly demonstrates how to conduct the source
inference with the loss term.
Theorem 2. Given local resulting model θk and a target record
z1, the source inference attack is given by:
S(θk,z1) = Eτ [σ (g(z1,θ,pτ ) +µλ)], (14)
where
ℓpτ (z1) : =−γ log
Z
t
e− 1
γℓ(t,z1)pτ (t)dt

, (15)
ℓ(θk,z1) : =−γ log

e− 1
γℓ(θk,z1)

, (16)
g(z1,θ,pτ ) : = 1
γ (ℓpτ (z1)−ℓ(θk,z1)). (17)
Proof. Forϕ andω defined in eqn. (6) and eqn. (7), we have:
ϕ =λ e− 1
γℓ(θk,z1)e− 1
γ
Pn
i=2sikℓ(θk,zi)
R
te− 1
γℓ(t,z1)e− 1
γ
Pn
i=2sikℓ(t,zi)dt
=λ e− 1
γℓ(θk,z1)pτ (θk)
R
te− 1
γℓ(t,z1)pτ (t)dt
,
(18)
ω = (1−λ) e− 1
γ
Pn
i=2sikℓ(θk,zi)
R
te− 1
γ
Pn
i=2sikℓ(t,zi)dt
= (1−λ)pτ (θk).
(19)
Thus, we have:
log
ϕ
ω

=− log
Z
t
e− 1
γℓ(t,z1)pτ (t)dt

+ log

e− 1
γℓ(θk,z1)

+ log
 λ
1−λ

= 1
γ (ℓpτ (z1)−ℓ (θk,z1)) +µλ.
(20)
S(θk,z1) = Eτ

σ

log
ϕ
ω

= Eτ

σ
 1
γ (ℓpτ (z1)−ℓ (θk,z1)) +µλ

= Eτ [σ (g(z1,θ,pτ ) +µλ)],
(21)
which concludes the proof.
3.3 Analysis of the Source Inference Attack
Theorem 2 implies that an attacker can infer the posterior
probability of a target record belonging to a particular client
by leveraging its prediction loss of the client’s local model.
Now we analyze the meaning of each term in eqn. (14)
and discuss how they affect the posterior probability. We
conclude from Theorem 2 and derive an attack method that
leverages the prediction loss of the local models to infer the
source client of a target record.
Meaning of g(z1,θ,pτ ). The term g(z1,θ,pτ ) is the gap
between ℓpτ (z1) and ℓ (θk,z1). Because τ is a training
set that does not contain any information about z1, pτ
corresponds to a posterior distribution of an ML model’s
parameters that trained without using z1. Note that ℓ(·) is
a loss function that measures the performance of the model
on a data record.ℓ (θk,z1) is the local modelθk’s evaluation

6
Client 1 Model 1
Client 2 Model 2
Client K Model K
Upload
Download
Server
Model
aggragation
SIAs
Data record
Client 1
Client 2
Client K
Belongs to
Fig. 1: An overview of SIAs in FL. In each communication round, each client transmits the necessary information of updates
to the central server for aggregation. The central server faithfully follows the defined FL protocol while inferring the source
clients of the target data records from legitimately received information from the local clients.
of the loss on the target record z1. Comparing eqn. (15)
and eqn. (16), we can find that ℓpτ (z1) is the expectation
of the loss ℓ(·,z1) over the typical models that have not
seenz1. Thus, we can interpretg(z1,θ,pτ ) as the difference
betweenθk’s loss onz1 and other models’ (trained without
z1) average loss onz1.
In other words, g(z1,θ,pτ ) is the differences between
the prediction loss of the local model of client k and the
average prediction loss of other clients’ local models. If
ℓpτ (z1) ≈ ℓ (θk,z1), which means the client k behaves
almost the same as other clients onz1, theng(z1,θ,pτ )≈ 0.
Since σ(µλ) = λ, the posterior probability S(θk,z1) is
equal to λ. Thus, we have no source information gain on
z1 beyond prior knowledge. In FL, the prior knowledge
P(sik = 1) = λ = 1
K . In this case, the source inference
equals random guess . However, if ℓpτ (z1) > ℓ (θk,z1),
that is, the client k performs better than other clients on
z1, g(z1,θ,pτ ) becomes positive. When g(z1,θ,pτ ) > 0,
P(s1k = 1|θk,z1) > λand thus we gain non-trivial source
information on z1. Moreover, since σ(·) is non decreasing,
smaller ℓ (θk,z1)) indicates higher probability that z1 be-
longs to the client k.
Conclusion from Theorem 2. We conclude that the smaller
the loss of client k’s local model on a target record z1, the
higher posterior probability that z1 belongs to the client
k. This motivates us to design the source inference attack
that the client whose local model has the smallest loss on
a target record should own this record. Moreover, if the
client’s local model’s behavior on its local training data is
different from that of other clients, our attack will always
achieve better performance than randomly guessing. We
give more empirical evidence in Section 4.
3.4 Source Inference Attacks in Different FL Frame-
works
In this paper, we investigate SIAs in three FL frameworks
under horizontal FL. Specifically, we investigate SIAs in
FedSGD [3], FedAvg [3], and FedMD [22] where local clients
upload gradients, model parameters, or predictions on an
unlabeled dataset to the server. The success of SIAs in the
three FL frameworks sheds light on how the communica-
tions between clients and the semi-honest central server in
existing FL frameworks enable the server to mount SIAs to
steal source information. Fig. 1 shows an overview of how
the honest-but-curious server implements SIAs in FL.
Intuitively, the server in FedAvg can directly conduct
an SIA in each communication round because it receives
the parameters of local models from the clients. Thus, the
server can directly use the local models to calculate the
prediction loss of a target record for implementing source
inference. However, in FedSGD and FedMD, the server
cannot directly implement SIAs because it cannot directly
leverage the updates from the clients to calculate the predic-
tion loss of the local models on the target records. In these
two frameworks, we introduce two strategies to enable the
server to implement the proposed SIAs.
In FedSGD, each client k uploads the average gradient
gk =∇ℓ(θt−1,Dk) calculated on its local data Dk at the
current global model θt−1. Thus, the server can leverage
the gradient from each client to update the global model
θk
t←θt−1−ηgk separately, whereη is a fixed learning rate
in the FL framework. Note thatθk
t essentially is the updated
local model of the clientk in thet-th communication round.
Thus, the server can useθk
t to calculate the prediction loss of
each local model in each communication round to conduct
SIAs.
In FedMD, the server can leverage knowledge distilla-
tion [43], [44] to achieve SIAs. Knowledge distillation aims
to transfer knowledge of larger models to smaller models
so that the smaller models are as accurate as larger models.
The larger models are referred to as teacher models and
the smaller models are referred to as students models.
During knowledge distillation, the student model is trained
to match the logits of the teacher model for learning its
knowledge. Knowledge distillation enables the smaller stu-

7
dent model to have similar performance to their teacher
models [45]. In FedMD, the server cannot directly use the
updates from the clients to calculate the prediction loss of
the target records because the updates are predictions of
a public dataset. However, because the clients’ predictions
on the public dataset represent the knowledge of the local
models, the server can leverage knowledge distillation to
mount SIAs. Specifically, for each of the local clients, the
server considers it as a teacher and leverages its predictions
to train a student model to mimic the local model. Because
the student models are expected to behave similarly to
the local models, the server can use the prediction loss of
the student models on the target records as the estimate
of the prediction loss of the local models. Thus, based on
the estimated prediction loss, the semi-honest server can
mount SIAs in FedMD. Note that although the student
models mimic the local models, there is a bias between the
estimated prediction loss and the actual loss of the local
model. However, if the estimated prediction loss of the
client owning the target instance is distinguishable from the
estimated losses of other clients, the SIAs can still succeed,
which we will show in the experiments.
Based on the analysis above, we propose three FL frame-
works F EDSGD-SIA, F EDAVG-SIA, and F EDMD-SIA that
allow an honest-but-curious server to conduct SIAs without
deviating from the normal FedSGD, FedAvg, or FedMD
protocols. Algorithm 1, Algorithm 2, and Algorithm 3 de-
scribe F EDSGD-SIA, F EDAVG-SIA, and F EDMD-SIA, re-
spectively. Each algorithm consists of two steps, i.e., Server
executes and ClientUpdate. In each algorithm, we assume
there are K clients and we take a target record of z as an
example to show how to implement SIAs.
FEDSGD-SIA: i) Server executes: As depicted in Al-
gorithm 1, the honest-but-curious server faithfully follows
FedSGD protocol (Lines 2-5, 8, 10, 11, 13, and 14) while
implementing SIAs (Lines 6, 7, and 9). First, the server
initializes the weights of the global model randomly fol-
lowing the step in Line 2. Next, in each communication
round, the server will receive the gradient calculated on
the local private dataset from each client (Lines 3-5). Then,
the server can use the gradient from each client to update
the global model separately to obtain the local models
(Line 6). The server calculates each local model’s prediction
loss on z and obtains the source i of z by finding which
client has the smallest loss on z (Line 7 and 8). Last, the
server averages the gradients to update a new global model
(Line 10), which is then distributed for the next round of
updating. ii) ClientUpdate: Each client has its own private
dataset. In each communication round, the client calculates
the gradient based on the private dataset and the current
global model (Line 13). Then, the gradients are sent back to
the server (Line 14).
FEDAVG-SIA: i) Server executes: As depicted in Algo-
rithm 2, the honest-but-curious server faithfully follows
FedAvg protocol (Lines 2-5, 7, 9, 10, 12-18) while imple-
menting SIAs (Lines 6 and 8). First, the server initializes the
weights of the global model randomly (Line 2). Next, in each
communication round, the server will receive the updated
models from clients (Lines 3, 4, and 5) and calculates each
model’s prediction loss on z (Line 6). The server then
obtains the source i of z by finding which client has the
Algorithm 1 FEDSGD-SIA The K clients are indexed byk;
η represents the learning rate;z represents a target record.
1: Server executes
2: initializeθ0 ▷ initialize weights of the global model
3: for each roundt = 1 toT do
4: for each clientk do
5: gk
t← ClientUpdate(θt−1) ▷ local gradient of the
client k at round t
6: θk
t←θt−1−ηgk ▷ use the gradient of the client
to obtain the local model
7: Computeℓ(θk
t,z) ▷ calculate the local prediction
loss onz
8: end for
9: i←argmin(ℓ(θ1,z),··· ,ℓ (θK,z)) ▷ identify the
source client
10: θt←θt−1−ηP
kgk
t ▷ update the global model
11: end for
12: ClientUpdate(θ) ▷ run on client k
13: Computegk =∇ℓ(θ,Dk)
14: returngk ▷ return the gradient to the central server
Algorithm 2 FEDAVG-SIA The K clients are indexed by
k; B represents the local mini-batch size; E represents the
number of local epochs; η represents the learning rate; z
represents a target record.
1: Server executes
2: initializeθ0 ▷ initialize weights of the global model
3: for each roundt = 1 toT do
4: for each clientk do
5: θk
t← ClientUpdate(θt−1) ▷ local model weight
of client k at round t
6: Computeℓ(θk
t,z) ▷ calculate the local prediction
loss onz
7: end for
8: i←argmin(ℓ(θ1,z),··· ,ℓ (θK,z)) ▷ identify the
source client
9: θt←P
k
n(k)
n θk
t ▷ update the global model
10: end for
11: ClientUpdate(θ) ▷ run on client k
12:B← (splitDk into multiple batches of size B)
13: for each local epochi from 1 toE do
14: for batchb∈B do
15: θ←θ−η∇ℓ(b;θ) ▷ mini-batch gradient descent
16: end for
17: end for
18: returnθ ▷ return local model to the central server
smallest loss onz (Line 8). Last, the server averages the up-
loaded local models to allocate a new global model as usual
(Line 9). ii) ClientUpdate: In each communication round,
the local clients will update the global model distributed
from the server. Each client performs local updates by using
mini-batch gradient descent to update the local models’
weights (Lines 12-17). To minimize communication with the
server, clients update local models for several epochs. Then,
the updated models are sent back to the server (Line 18).
FEDMD-SIA: i) Server executes: As depicted in Algo-
rithm 3, the honest-but-curious server faithfully follows
FedMD protocol (Lines 2-7, 9-11, 14, 15, 17, and 19-26)

8
Algorithm 3 FEDMD-SIA The K clients are indexed by k;
z represents a target record; D0 is a public dataset; Dk is a
private dataset of the client k;θk represents the local model
of the client k; θs is a student model.
1: Initialization phase
2: the clients initialize local models θ1,··· ,θK and each
client k updates her model weight θk on her private
datasetDk
3: for each clientk do
4: Yk
0 = PREDICT (θk,D 0)
5: sendYk
0 to the server
6: end for
7: server calculates ˜Y0 = 1
K
P
kYk
0
8: Server executes
9: for each roundt = 1 toT do
10: for each clientk do
11: Yk
t ← ClientUpdate(θk, ˜Yt−1,D 0) ▷ predictions
of client k on D0 at round t
12: θk
s← TRAIN (θs,Y k
t ,D 0) ▷ server trains a student
model to mimic the local model
13: Computeℓ(θk
s,z) ▷ calculate local loss onz
using the student model
14: end for
15: end for
16: i←argmin(ℓ(θ1
s,z),··· ,ℓ (θK
s ,z)) ▷ identify the
source client
17: ˜Yt = 1
K
P
kYk
t ▷ prediction aggregation at the server
18: ClientUpdate(θ, ˜Yt−1,D 0) ▷ run on
client k
19: for each local epochi from 1 toE1 do
20: Digest:θ← TRAIN (θ, ˜Yt−1,D 0) ▷ train the model on
the public dataset
21: end for
22: for each local epochi from 1 toE2 do
23: Revisit:θ← TRAIN (θ,Dk) ▷ train the model on the
private dataset
24: end for
25: Y = PREDICT (θ,D 0) ▷ predictions of class scores on D0
26: returnY ▷ return predictions to the central server
while implementing SIAs (Lines 12, 13, and 16). In each
communication round, the server receives the predictions
of the public dataset from each client (Lines 9-11). Then, for
each client, the server trains a student model on the public
dataset to imitate the local model (Line 12). The server
calculates each student model’s prediction loss on z and
obtains the source i by finding which student model has
the smallest prediction loss onz (Lines 13 and 16). Last, the
server aggregates the predictions from each client for the
next round of updating (Line 17). ii) ClientUpdate: First,
each client trains the local model on the soft-labeled public
dataset to approach the consensus on the public dataset
(Lines 19-21 for Digest). Then, each client trains the local
model on the private dataset (Lines 22-25 for Revisit). Last,
each client computes the predictions on the public dataset
and sends the predictions to the server (Line 26).
Complexity analysis of SIAs. The proposed SIAs leverage
the prediction loss of local models to infer the source client
of a target record. The computational complexity of SIAs
mainly depends on two factors. One is the evaluation of
the prediction loss of local models, and the other is the
identification of the client having the smallest prediction
loss. In F EDAVG-SIA, because the server directly leverages
the uploaded local models to evaluate the prediction loss,
the computation cost is determined by the model size. The
time complexity of SIAs in F EDAVG-SIA with respect to
the number of clients K is O(K). As the server received
the gradients from each client to update the global model
in F EDSGD-SIA, the local models can be obtained by the
server for SIAs with no extra computation cost. Thus, the
computational complexity of F EDSGD-SIA is the same as
FEDAVG-SIA. In F EDMD-SIA, SIAs are more complicated
than that in F EDSGD-SIA and F EDAVG-SIA because the
server requires a student model to mimic the behavior of
local models. The computation cost of training the stu-
dent models is mainly determined by the student model
size and the size of the public dataset. Then, the server
can implement SIAs based on the student models, with a
time complexity of O(K) with respect to the number of
clients. In our experiments using PyTorch with a single GPU
NVIDIA Tesla P40, the execution time of SIAs in F EDSGD-
SIA and F EDAVG-SIA is within seconds and in F EDMD-
SIA is within one minute.
4 E XPERIMENTS
In this section, we empirically evaluate F EDSGD-SIA,
FEDAVG-SIA, and F EDMD-SIA. We first introduce datasets
and model architectures used in the experiments. Then, we
demonstrate the effectiveness of SIAs in the three FL frame-
works and conduct a detailed ablation study to identify how
different factors in FL influences the performance of SIAs. In
the end, we discuss why our SIAs work.
4.1 Datasets and Model Architectures
Datasets. The datasets used in our experiments are re-
ported in Table 1. We create an IIDSynthetic dataset to allow
us to precisely manipulate data heterogeneity. We generate
Synthetic as described in previous works [29], [46]. The
remaining datasets are widely used datasets for simulating
and evaluating the privacy leakage on machine learning
models [11], [15], [16], [47]. For MNIST and CIFAR10, the
training dataset and testing datasets have been divided
when downloading them. For the remaining datasets, we
use the train test split function from the sklearn 2 toolkit to
randomly select 80% samples as the training records (before
partitioning client data), and the remaining 20% records are
used as the testing records. We use the four datasets to eval-
uate F EDSGD-SIA and F EDAVG-SIA. Because F EDMD-
SIA requires a public dataset to share the knowledge of
the clients, we evaluate it on paired datasets. Specifically,
we select paired MNIST/FEMNIST and CIFAR-10/CIFAR-
100. For MNIST/FEMNIST pairs, we select MNIST as the
public data and a subset of the Federated Extended MNIST
(FEMNIST) [5] as the private data. For CIFAR-10/CIFAR-
100 pairs, CIFAR-10 is selected as the public dataset, and
the private dataset is a subset of CIFAR-100.
2. https://scikit-learn.org/stable/

9
TABLE 1: A summary of datasets used in the experiments.
Dataset #Records #Classes Dimension of records
Synthetic 100k 10 60
MNIST 3 70k 10 1x28x28
CIFAR-10 4 60k 10 3x32x32
FEMNIST 5 80k 62 1x28x28
CIFAR-100 6 60k 100 3x32x32
Purchase 7 197.3k 100 600
TABLE 2: The DNN architecture is used for classification
tasks. (a) The CNN architecture is used for MNIST, and
CIFAR-10. (b) The FC architecture is used for Synthetic and
Purchase.
(a) CNN architecture
Layer type Size
Input
Convolution + ReLU 5×5×32
Max Pooling 2×2
Convolution + ReLU 5×5×64
Max Pooling 2×2
FC + ReLU 512
FC + ReLU 128
Activation Softmax
Output
(b) MLP architecture
Layer type Size
Input
FC + ReLU 200
Activation Softmax
Output
Models. We use deep neural networks (DNNs) as the
global models in FL for the classification tasks. Specifically,
we use convolutional neural networks (CNN) for the image
datasets of MNIST and CIFAR-10. For binary datasets of
Synthetic and Purchase, we use fully-connected (FC) neural
networks. The architecture of the CNN and FC classifiers
used in F EDSGD-SIA and F EDAVG-SIA are described in
Table 2. Because F EDMD-SIA is designed for FL with het-
erogeneous architectures, we adopt the setting of the differ-
ent CNN architectures of the clients in [22], which proposed
the FedMD framework. For the detailed description of the
CNN architectures of the clients in F EDMD-SIA, readers
can refer to [22]. The student model in F EDMD-SIA used
for imitating the local models is listed in Table 2a. Note that
the DNN architectures used in this paper do not necessarily
achieve the best performance for the considered datasets in
FL, because our goal is not to attack the best DNN architec-
ture. In this paper, we aim to show that FL is vulnerable to
SIAs.
4.2 Evaluation Metrics, Baseline, and Parameter Set-
tings
Metric. We use attack success rate (ASR) to evaluate SIAs,
which is the most commonly used metric to evaluate the
performance of a given attack approach. ASR [19] is defined
as the fraction of the number of attacks that successfully
identify the source of the target instances:
ASR = # Successful attacks
# All attacks .
3. http://yann.lecun.com/exdb/mnist/
4. https://www.cs.toronto.edu/ kriz/cifar.html
5. https://github.com/TalwalkarLab/leaf
6. https://www.cs.toronto.edu/ kriz/cifar.html
7. https://www.kaggle.com/c/acquire-valued-shoppers-
challenge/data
For example, consider there are 100 target instances for iden-
tifying their source clients, and SIAs successfully identify 60
instances’ source clients. Then, the ASR is calculated as 60%.
Baseline. Because we are the first to propose SIAs, there
is no current work we can compare. Thus, to demonstrate
the effectiveness of SIAs, we consider a trivial attack of ran-
domly guessing as the baseline of SIAs. Randomly guessing
randomly selects a client as the source of the target training
record. The ASR of randomly guessing is defined as 1
K ,
whereK is the number of clients.
Hyper-parameter setting. We use the SGD optimizer for
all the models with a learning rate of 0.01. We assume there
are 10 clients in the FL. In each trial of the experiment, 100
training records from each client are selected as the target
records for SIAs. For all the learning tasks, we set the total
number of communication rounds to 20, which is enough for
the global model to converge. During each communication
round, we record the ASR of SIAs. We report the mean of
ASR over five different random seeds.
4.3 Factors in Source Inference Attacks
Data distribution α. In FL, the training data across the
clients are usually non-IID [48]. This means the local data
from one client can not be considered as samples drawn
from the overall data distribution. To simulate the hetero-
geneity distribution of client data, we leverage a Dirichlet
distribution as in previous works [31], [49], [50], [51], [52]
to create disjoint non-IID training data for each client. The
degree of non-IID is controlled by the value of α (α > 0)
of the Dirichlet distribution. For example, α = 100 imitates
almost identical local data distributions. With a smaller α,
each client is more likely to have the training records from
only one class. We use Fig. 2 to visualize how the training
records of CIFAR-10 are distributed among 10 clients when
setting differentα values.
Number of local epochs E. In FEDAVG-SIA and F EDMD-
SIA, each client can update their model for several epochs,
and then send the model weights or predictions to the
server. Many recent studies [53], [54], [55], [56] have shown
that DNN models can easily memorize their training data.
Intuitively, if a client trains the local model with more
epochs, the local updated model should better remember
the information of the local dataset and be more confident to
classify its training records. Accordingly, the prediction loss
of a target record calculated from the update of the source
client will be much smaller than that calculated from the
updates of other clients, which will be beneficial for SIAs.
4.4 Effectiveness of Source Inference Attacks
To demonstrate the effectiveness of SIAs, we evaluate
FEDSGD-SIA, F EDAVG-SIA, and F EDMD-SIA under com-
mon settings: We divide the training data to the clients in the
three FL frameworks with α = 1 , resulting in a moderate
non-IID distribution. We set E = 5 for F EDAVG-SIA, and
E1 = 1,E 2 = 5 for F EDMD-SIA, which are the common
settings for FedAvg [3] and FedMD [22].
Fig. 3 shows the ASR of F EDSGD-SIA, F EDAVG-SIA,
and F EDMD-SIA during each communication round. We
can observe from Fig. 3 that the ASR on all datasets is larger
than the baseline of 10% in each communication round,

10
/uni00000013/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c
/uni00000026/uni0000004f/uni0000004c/uni00000048/uni00000051/uni00000057/uni00000003/uni0000002c/uni00000027/uni00000056
/uni00000013
/uni00000014
/uni00000015
/uni00000016
/uni00000017
/uni00000018
/uni00000019
/uni0000001a
/uni0000001b
/uni0000001c/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000003/uni0000004f/uni00000044/uni00000045/uni00000048/uni0000004f/uni00000056
(a)α = 100
/uni00000013/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c
/uni00000026/uni0000004f/uni0000004c/uni00000048/uni00000051/uni00000057/uni00000003/uni0000002c/uni00000027/uni00000056
/uni00000013
/uni00000014
/uni00000015
/uni00000016
/uni00000017
/uni00000018
/uni00000019
/uni0000001a
/uni0000001b
/uni0000001c/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000003/uni0000004f/uni00000044/uni00000045/uni00000048/uni0000004f/uni00000056 (b)α = 1
/uni00000013/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c
/uni00000026/uni0000004f/uni0000004c/uni00000048/uni00000051/uni00000057/uni00000003/uni0000002c/uni00000027/uni00000056
/uni00000013
/uni00000014
/uni00000015
/uni00000016
/uni00000017
/uni00000018
/uni00000019
/uni0000001a
/uni0000001b
/uni0000001c/uni00000026/uni0000004f/uni00000044/uni00000056/uni00000056/uni00000003/uni0000004f/uni00000044/uni00000045/uni00000048/uni0000004f/uni00000056 (c)α = 0.1
Fig. 2: Illustration of the number of samples per class allocated to each client at different Dirichlet distribution alpha values,
for CIFAR-10 with 10 clients. The x-axis represents the IDs of the clients. The y-axis represents the class labels of CIFAR-10.
The dot size reflects the number of samples allocated to the clients. As we can see, the data distribution across the clients
becomes more and more non-IID as α decreases.
/uni00000013/uni00000017/uni0000001b/uni00000014/uni00000015/uni00000014/uni00000019
/uni00000006/uni00000003/uni00000052/uni00000049/uni00000003/uni00000046/uni00000052/uni00000050/uni00000050/uni00000058/uni00000051/uni0000004c/uni00000046/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000055/uni00000052/uni00000058/uni00000051/uni00000047/uni00000056
/uni00000013/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000014
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000016
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000018/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000056/uni00000058/uni00000046/uni00000046/uni00000048/uni00000056/uni00000056/uni00000003/uni00000055/uni00000044/uni00000057/uni00000048
/uni00000036/uni0000005c/uni00000051/uni00000057/uni0000004b/uni00000048/uni00000057/uni0000004c/uni00000046
/uni00000033/uni00000058/uni00000055/uni00000046/uni0000004b/uni00000044/uni00000056/uni00000048
/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037
/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013
/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048
(a) F EDSGD-SIA
/uni00000013/uni00000017/uni0000001b/uni00000014/uni00000015/uni00000014/uni00000019
/uni00000006/uni00000003/uni00000052/uni00000049/uni00000003/uni00000046/uni00000052/uni00000050/uni00000050/uni00000058/uni00000051/uni0000004c/uni00000046/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000055/uni00000052/uni00000058/uni00000051/uni00000047/uni00000056
/uni00000013/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000014
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000016
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000018
/uni00000013/uni00000011/uni00000019/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000056/uni00000058/uni00000046/uni00000046/uni00000048/uni00000056/uni00000056/uni00000003/uni00000055/uni00000044/uni00000057/uni00000048
/uni00000036/uni0000005c/uni00000051/uni00000057/uni0000004b/uni00000048/uni00000057/uni0000004c/uni00000046
/uni00000033/uni00000058/uni00000055/uni00000046/uni0000004b/uni00000044/uni00000056/uni00000048
/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037
/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013
/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048 (b) F EDAVG-SIA
/uni00000013/uni00000017/uni0000001b/uni00000014/uni00000015/uni00000014/uni00000019
/uni00000006/uni00000003/uni00000052/uni00000049/uni00000003/uni00000046/uni00000052/uni00000050/uni00000050/uni00000058/uni00000051/uni0000004c/uni00000046/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000055/uni00000052/uni00000058/uni00000051/uni00000047/uni00000056
/uni00000013/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000014
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000016
/uni00000013/uni00000011/uni00000017/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000056/uni00000058/uni00000046/uni00000046/uni00000048/uni00000056/uni00000056/uni00000003/uni00000055/uni00000044/uni00000057/uni00000048
/uni00000029/uni00000028/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013/uni00000013/uni00000025/uni00000044/uni00000056/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048 (c) F EDMD-SIA
Fig. 3: Attack success rate (ASR) of the three proposed FL frameworks during each communication round. In each plot,
thex axis represents the number of communication rounds, and the y axis represents ASR. Each line is the mean ASR of 5
runs of experiments with shaded regions corresponding to 95% confidence interval. (a) ASR of F EDSGD-SIA. (b) ASR of
FEDAVG-SIA. (c) ASR of F EDMD-SIA. As we can see, the ASR on all datasets is larger than the baseline, i.e., randomly
guessing, demonstrating the effectiveness of SIAs.
demonstrating the effectiveness of SIAs. This indicates that a
semi-honest server can steal significant source information
of the training data records via our proposed SIAs in any
communication rounds during federated training. We can
also see that the ASR of each FL framework differs in differ-
ent datasets. This is because the local models are overfitted
with a different level to their local training data. We will
discuss this phenomenon in more detail in subsection 4.6.
Takeaway 1 Our proposed SIAs are effective on the
FedSGD, FedAvg, and FedMD.
4.5 Ablation Study
We conduct a detailed ablation study on F EDSGD-SIA,
FEDAVG-SIA, and F EDMD-SIA to learn how data distri-
bution and local epochs influence the effectiveness of SIAs.
We record and report the highest ASR during the federated
training process. Table 3 shows the ASR when setting differ-
ent levels of non-IID data distribution and different number
of local epochs.
Evaluation of non-IID data distribution. As we can
see in Table 3, the ASR of all the three FL frameworks
on all datasets increase when the degree of non-IID data
distribution across clients increases. For example, the ASR
of FEDSGD-SIA on CIFAR-10 increases from 17.6% to 58.3%
when the non-IID data distribution increase fromα = 100 to
α = 0.1. This is because the more non-IID of the local data is,
the more different the updated local models will be, which
benefits SIAs. For example, for the CIFAR-10 task, a client
is highly likely to have training records from only one class
(e.g., deer) when the degree of the non-IID data distribution
is high. It is expected that such a client’s local model will
perform well in predicting its own records of deer images
but perform badly in predicting the other clients’ records
such as dogs and trucks because the local model had never
seen such images during the update process. Thus, the local
model will have very small prediction losses on its own
records and large losses on the other records. The distin-
guishable prediction losses across different local models of
the clients enable the server to easily implement SIAs to
infer where a training record comes from.
Evaluation of local epochs. As shown in most scenarios
in Table 3, the increase of E from 1 to 10 makes the ASR of
FEDSGD-SIA and F EDMD-SIA increase. For example, the
ASR of F EDAVG-SIA on CIFAR-10 increases from 16.6% to
51.1% when the local epochs increase from 1 to 10 when
α is set to 100. This is because the more epochs the clients
update, the more confident the local model is in predicting
its training data. However, we also observe that there are

11
TABLE 3: Understanding the impact of data distribution and local epochs in SIAs. For each value of the parameter, we
report the averaged attack success rate over 5 different random seeds with its standard deviation. Because F EDSGD-
SIA transmits gradient calculated on the current global model (equivalent to E = 1 ) and does not involve training local
models for several epochs, we leave the column E = 5 andE = 10 of the ASR of F EDSGD-SIA blank.
The attack success rate (%) of source inference attacks
Datasets α = 100 α = 1 α = 0.1
E = 1 E = 5 E = 10 E = 1 E = 5 E = 10 E = 1 E = 5 E = 10
FEDSGD-SIA
Synthetic 19.1± 0.4 — — 30.9± 2.6 — — 55.9± 3.2 — —
Purchase 15.7± 0.4 — — 30.6± 1.0 — — 63.9± 1.6 — —
MNIST 12.7± 0.3 — — 23.1± 0.5 — — 50.2± 3.7 — —
CIFAR-10 17.6± 0.3 — — 28.5± 0.7 — — 58.3± 5.2 — —
FEDAVG-SIA
Synthetic 19.2± 0.5 19 .7± 0.5 18 .9± 0.6 28 .5± 1.4 28 .1± 1.8 28 .5± 1.2 53 .6± 1.3 50 .8± 2.6 51 .7± 3.3
Purchase 15.6± 0.5 21 .9± 0.3 28 .2± 0.5 31 .4± 0.8 32 .6± 0.7 34 .8± 0.5 67 .1± 0.4 64 .4± 0.8 66 .2± 0.9
MNIST 12.1± 0.1 12 .8± 0.3 13 .5± 0.3 23 .7± 0.9 23 .3± 0.4 22 .1± 0.7 58 .4± 4.9 53 .1± 1.1 42 .3± 2.6
CIFAR-10 16.6± 0.2 47 .8± 0.4 51 .1± 1.1 26 .3± 0.7 49 .9± 0.7 55 .8± 0.7 56 .8± 3.9 60 .9± 3.9 62 .5± 1.9
FEDMD-SIA FEMNIST 15.1± 0.4 17 .2± 0.5 17 .5± 0.6 23 .2± 1.1 25 .4± 1.3 24 .5± 1.1 42 .5± 3.3 40 .6± 1.0 46 .7± 2.1
CIFAR-100 18.2± 0.2 18 .8± 0.8 20 .5± 0.4 23 .9± 1.1 28 .1± 1.9 25 .5± 0.6 40 .3± 1.7 43 .5± 3.8 45 .6± 3.9
scenarios, e.g., FEDAVG-SIA on MNIST when α = 1 and
α = 0.1, that increasing E does not lead to the increase
of ASR but leads to a decrease. We suspect this is because
training the local model with more epochs not only makes
it more confident to predict its training records but also
helps it to generalize better to other clients’ data. In this
case, the prediction losses across different local models will
become less distinguishable, which leads to a decrease in
ASR. We will further explain how E influences ASR from
the perspective of overfitting in the following section.
Takeaway 2
• Higher data heterogeneity among local clients
results in more effective SIAs.
• Larger local epochs in clients usually lead to
more effective SIAs.
4.6 Why Source Inference Attacks Work
Machine Learning models especially DNNs are usually
overparameterized with high complexity. This enables
DNNs to learn patterns effectively from the training data
on one side while on the other side such models can have
unnecessarily high capacities to memorize the details of the
training data [53], [54], [56], which can lead to the overfitting
of ML models. An overfitted ML model cannot generalize
well on its test data, i.e., the model performs much better
on its training data than test data. In FL, the local dataset
of a client often has a limited number of records and fails
to represent the whole data distribution, which exacerbates
the overfitting of local models. An overfitted local model of
a client is expected to have a much smaller prediction loss on
a training record of its own than the loss of a training record
of other clients. The distinguishable prediction losses of the
local models of the different clients enable our proposed
SIAs to work effectively. Moreover, the more overfitted the
local models are, the more effective the SIAs will be.
We use generalization error [57] to measure the over-
fitting level of the local models, which is a widely used
metric to quantify overfitting in existing works [19]. The
generalization error of an ML model is defined as the
absolute difference between the training accuracy and the
testing accuracy of the model. Here, we first calculate the
generalization error for each of the local models by using its
local training dataset and the global testing dataset. Then,
we calculate the averaged generalization error of the local
models to reflect the overfitting level of the FL system.
The impact of overfitting on ASR. Fig. 4 shows the
impact of overfitting on the performance of SIAs. As we
can see, for FEDSGD-SIA, F EDAVG-SIA, and F EDMD-SIA,
the honest-but-curious server can obtain a higher attack
success rate when the FL system is more overfitted to the
training dataset. This observation validates our analysis
of the relationship between the overfitting level and the
performance of SIAs.
The impact of non-IID distribution on overfitting. Fig. 5
examines the effect of different levels of non-IID data dis-
tribution on the overfitting level of the FL system. As we
can see, for all the FL systems, increasing the level of non-
IID data across the clients (i.e., decreasing α from 100 to 10
and 0.1) will inevitably increase the overfitting level. This is
because the more non-IID data is, the less representative the
local data will be, which makes the local model less possible
to generalize well beyond its own training data.
The impact of local epoch on overfitting. Fig 6 shows the
impact of the number of epochs on the overfitting level of
FEDAVG-SIA and F EDMD-SIA. We can see that in F EDMD-
SIA, increasing the number of local epochs does not increase
the overfitting level of the FL system too much, which
results in a slight increase of the attack success rate as we
can observe in Table 3. In F EDAVG-SIA, we observe that
there is a large increase of the overfitting level on CIFAR-10
when we increase E from 1 to 10, while there is no such
trend for the other datasets. This observation explains why
the ASR of CIAFR-10 is more sensitive to the local epoch
compared to other datasets in Table 3.
In summary, the overfitting of the local models directly
contributes to the success of SIAs, while overfitting is mainly
caused by the non-IID data distribution across the clients.
In FL, if the local dataset fails to represent the overall data
distribution, the local model can easily overfit to the local
training dataset, as depicted in Fig. 5. Because an overfitted
local model cannot generalize well to the data beyond
its local training records, it will behave differently on its

12
/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b
/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000056/uni00000058/uni00000046/uni00000046/uni00000048/uni00000056/uni00000056/uni00000003/uni00000055/uni00000044/uni00000057/uni00000048
/uni00000036/uni0000005c/uni00000051/uni00000057/uni0000004b/uni00000048/uni00000057/uni0000004c/uni00000046
/uni00000033/uni00000058/uni00000055/uni00000046/uni0000004b/uni00000044/uni00000056/uni00000048
/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037
/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013
(a) F EDSGD-SIA
/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b
/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000056/uni00000058/uni00000046/uni00000046/uni00000048/uni00000056/uni00000056/uni00000003/uni00000055/uni00000044/uni00000057/uni00000048
/uni00000036/uni0000005c/uni00000051/uni00000057/uni0000004b/uni00000048/uni00000057/uni0000004c/uni00000046
/uni00000033/uni00000058/uni00000055/uni00000046/uni0000004b/uni00000044/uni00000056/uni00000048
/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037
/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013 (b) F EDAVG-SIA
/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001b
/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013/uni00000024/uni00000057/uni00000057/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000056/uni00000058/uni00000046/uni00000046/uni00000048/uni00000056/uni00000056/uni00000003/uni00000055/uni00000044/uni00000057/uni00000048
/uni00000029/uni00000028/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013/uni00000013 (c) F EDMD-SIA
Fig. 4: Understanding the impact of overfitting in SIAs. In each plot, the x axis represents the overfitting level of the FL
framework, and the y axis represents ASR. As we can see, for all the datasets in all the three FL frameworks, the more
overfitted the local models are, the higher the ASR will be.
/uni00000013/uni00000011/uni00000013/uni00000014/uni00000014 /uni00000014/uni00000013
1
/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000036/uni0000005c/uni00000051/uni00000057/uni0000004b/uni00000048/uni00000057/uni0000004c/uni00000046
/uni00000033/uni00000058/uni00000055/uni00000046/uni0000004b/uni00000044/uni00000056/uni00000048
/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037
/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013
(a) F EDSGD-SIA
/uni00000013/uni00000011/uni00000013/uni00000014/uni00000014 /uni00000014/uni00000013
1
/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000036/uni0000005c/uni00000051/uni00000057/uni0000004b/uni00000048/uni00000057/uni0000004c/uni00000046
/uni00000033/uni00000058/uni00000055/uni00000046/uni0000004b/uni00000044/uni00000056/uni00000048
/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037
/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013 (b) F EDAVG-SIA
/uni00000013/uni00000011/uni00000013/uni00000014/uni00000014 /uni00000014/uni00000013
1
/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000029/uni00000028/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013/uni00000013 (c) F EDMD-SIA
Fig. 5: Understanding the impact of the non-IID data distribution on the overfitting level of the FL framework. In the three
FL frameworks, the local epoch sets to 1. In each plot, the x axis represents the inverse of α, and the y axis represents the
overfitting level.
/uni00000014 /uni00000018 /uni00000014/uni00000013
/uni0000002f/uni00000052/uni00000046/uni00000044/uni0000004f/uni00000003/uni00000048/uni00000053/uni00000052/uni00000046/uni0000004b
/uni00000013
/uni00000013/uni00000011/uni00000014
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000016
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000018
/uni00000013/uni00000011/uni00000019/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000036/uni0000005c/uni00000051/uni00000057/uni0000004b/uni00000048/uni00000057/uni0000004c/uni00000046
/uni00000033/uni00000058/uni00000055/uni00000046/uni0000004b/uni00000044/uni00000056/uni00000048
/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037
/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013
(a) F EDAVG-SIA
/uni00000014 /uni00000018 /uni00000014/uni00000013
/uni0000002f/uni00000052/uni00000046/uni00000044/uni0000004f/uni00000003/uni00000048/uni00000053/uni00000052/uni00000046/uni0000004b
/uni00000013
/uni00000013/uni00000011/uni00000013/uni00000018
/uni00000013/uni00000011/uni00000014/uni00000013
/uni00000013/uni00000011/uni00000014/uni00000018
/uni00000013/uni00000011/uni00000015/uni00000013
/uni00000013/uni00000011/uni00000015/uni00000018/uni00000032/uni00000059/uni00000048/uni00000055/uni00000049/uni0000004c/uni00000057/uni00000057/uni0000004c/uni00000051/uni0000004a/uni00000003/uni0000004f/uni00000048/uni00000059/uni00000048/uni0000004f
/uni00000029/uni00000028/uni00000030/uni00000031/uni0000002c/uni00000036/uni00000037/uni00000026/uni0000002c/uni00000029/uni00000024/uni00000035/uni00000010/uni00000014/uni00000013/uni00000013 (b) F EDMD-SIA
Fig. 6: Understanding the impact of the local epoch on the overfitting level of the FL framework. In the three FL frameworks,
the data distribution value α sets to 100. In each plot, the x axis represents the number of local epoch, and the y axis
represents the overfitting level.
training data from other clients’ data, which guarantees
the feasibility of SIAs. Many recent works [29], [31], [46],
[48], [58] have shown that the non-IID data distribution has
brought important challenges to FL such as model conver-
gence guarantees. In this paper, we demonstrate another
challenge of non-IID from the perspective of privacy: The
leakage of source privacy about the training data.
There are several promising applications where SIAs can
be applied. First, as a newly proposed inference attack, SIAs
can be leveraged to evaluate the privacy-preserving ability
of an FL framework. A strong privacy-aware FL framework
should guarantee the source privacy of the clients. Second,
because no defense methods have been specifically pro-
posed for mitigating SIAs, they can inspire researchers to
propose novel defense methods or design new FL frame-
works for protecting clients’ source privacy. Last, we have
identified that the overfitting of an FL framework is the
main success factor for SIAs. Thus, SIAs can be used to help
evaluate and understand the overfitting phenomenon of FL
frameworks. For example, SIAs can be used to determine
how many local epochs a local model should update to
avoid overfitting.

13
TABLE 4: The evaluation of source inference defenses via differential privacy. In F EDSGD-SIA and F EDAVG-SIA, the
training accuracy and testing accuracy is the accuracy of the trained global model. In F EDMD-SIA, because there is no
global model, the training accuracy and testing accuracy is the averaged accuracy of the local models. The privacy budget
is calculated every communication round.
Datasets
FL without differential privacy FL with differential privacy
Training acc.(%) Testing acc.(%) ASR (%) Training acc.(%) Testing acc.(%) ASR (%) Privacy budgets
FEDSGD-SIA
Synthetic 84.3± 0.5 83 .9± 0.4 30 .9± 2.6 34 .9± 1.3 34 .7± 1.1 22 .5± 0.7 1.8
Purchase 87.8± 0.1 85 .9± 0.2 30 .6± 1.0 14 .6± 0.3 14 .2± 0.1 23 .2± 0.6 2.1
MNIST 99.3± 0.1 99 .1± 0.1 23 .1± 0.5 35 .7± 0.7 32 .1± 0.5 15 .8± 1.9 1.9
CIFAR-10 67.8± 0.1 64 .5± 0.2 28 .5± 0.7 13 .8± 0.5 13 .7± 0.3 22 .1± 1.2 1.8
FEDAVG-SIA
Synthetic 94.2± 0.4 93 .8± 0.4 28 .5± 1.2 51 .7± 1.5 51 .2± 0.9 21 .4± 1.3 1.9
Purchase 94.2± 0.1 89 .5± 0.1 34 .8± 0.5 33 .6± 1.8 32 .1± 1.3 24 .8± 1.2 6.8
MNIST 99.7± 0.1 99 .2± 0.1 22 .1± 0.7 10 .9± 0.1 9 .9± 0.1 11 .8± 0.5 3.2
CIFAR-10 96.3± 0.6 70 .1± 0.5 55 .8± 0.7 10 .1± 0.1 9 .9± 0.1 19 .5± 0.6 2.8
FEDMD-SIA FEMNIST 99.7± 0.1 84 .1± 1.5 24 .5± 1.1 16 .7± 0.1 16 .5± 0.1 16 .3± 0.3 4.8
CIFAR-100 99.5± 0.1 72 .1± 0.9 25 .5± 0.6 16 .6± 0.1 16 .2± 0.1 17 .1± 0.5 9.6
Takeaway 3
• The overfitting of local models is the main rea-
son why SIAs can succeed.
• Higher data heterogeneity results in a higher
overfitting level of local models.
5 D ISCUSSION AND FUTURE WORK
5.1 Defenses against SIAs
Differential privacy. As a probabilistic privacy mecha-
nism, differential privacy (DP) [59] provides a mathemat-
ically provable privacy guarantee. Recently, many works
[60], [61], [62], [63], [64] suggest DP can be applied to ML
models to defend against inference attacks such as member-
ship inference attacks and property inference attacks. When
an ML model is trained with differential privacy guarantees,
the learned model is expected not to learn or remember any
specific data details. By definition, if the local models in
FL are differentially private, the success probability of SIAs
should be reduced because the communication updates cal-
culated from such models should contain less information
about the local training datasets. We discuss and evaluate
whether DP can effectively mitigate SIAs.
In the experiments, we set α = 1 for all three FL
frameworks and set E = 10 for F EDAVG-SIA and F EDMD-
SIA. Here, we consider record-level DP implemented with
DP-SGD [65], which is the first and the most widely used
differentially private training method. We train differen-
tially private local models in each communication round
before sending the updates to the central server. In our
experiments, we fine-tune the noise of DP to obtain an attack
success rate slightly larger than the baseline of random
guess (i.e., 10% because of 10 clients) while reporting the
training and testing accuracy of the FL system.
Table 4 compares the FL systems with DP to the FL
systems without DP in terms of the model utility (measured
by the testing accuracy) and the attack success rate of
SIAs. As we can see, differential privacy indeed reduces the
ASR of F EDSGD-SIA, F EDAVG-SIA, and F EDMD-SIA on
all the datasets. However, there is a significant model utility
reduction in the FL systems. In some cases, the global model
does not converge, e.g., F EDAVG-SIA trained on MNIST
and CIFAR-10. We are aware that there exists client-level DP
[61], [66] in FL where the local clients’ privacy is preserved
while the utility of the FL system is maintained. However,
client-level DP requires a very large number of clients
(e.g., thousands in [66]) to achieve the desired privacy-
utility guarantee. Applying them in our experiments is not
expected to achieve satisfactory results as there are only ten
clients in the FL system.
Regularization techniques. Regularization techniques
such as L2-regularization and Dropout [67] are leveraged
to defend against membership inference attacks on machine
learning models [19]. Intuitively, regularization techniques
can help local models to reduce their overfitting degrees
to the local training datasets. Thus, it is promising that we
can leverage regularization techniques to defend against
SIAs in FL. However, regularization techniques are not
perfect as not all of them can achieve a satisfactory trade-
off between privacy and model utility: strong regularization
can also significantly reduce model performance [16]. We
leave the investigation of finding appropriate regularization
techniques as a defense against SIAs for our future work.
5.2 Limitations and Future Work for SIAs
Number of clients. In our experiments, to demonstrate
the effectiveness of SIAs, we evaluate the FL systems with
ten clients, which is a small number. This FL scenario
corresponds to federated to business mode where a handful
of organizations jointly build a useful model, e.g., a small
number of banks collaboratively build a fraud detection
model [21], [24]. There also exists federated to customer
mode where thousands or even millions of clients involved
in FL systems, e.g., thousands of mobile devices jointly train
a model for next-word prediction [62]. Intuitively, SIAs are
much more challenging under federated to customer mode
because finding the source of a target record from thousands
of clients is difficult. Under this setting, the performance of
our proposed SIAs is expected to drop while still performing
better than randomly guessing as long as the local client’s
model’s behavior on its local training data is different from
that of other clients (see discussion in Section 3.3).
Model size. As this is the first paper that investigates the
source privacy leakage in FL, we only evaluate the FL
system with relatively small deep learning models such as

14
CNN models with only two convolutional layers but have
not evaluated large models. This is because the purpose of
our experiments is to show SIAs are effective in the FL
frameworks of FedSGD, FedAvg, and FedMD but not to
attack the best deep models. However, it would be interest-
ing to investigate SIAs in FL with large models of millions
of parameters such as VGG [68] and Resnet [69]. Models
with large sizes on one side have strong learning ability
while from the other side have the unnecessary capability to
memorize their training data [70], which might be beneficial
for SIAs. We leave the investigation of how model size
influences the performance of SIAs for future work.
Best inference round. Our proposed SIAs enable the server
to infer the source of a target record in every communication
round. As we can see in Fig. 3, the attack success rate of
our proposed SIAs varies in each communication round
on all the datasets. This can be because the local models
overfit their local training datasets to different degrees in
different communication rounds. There are two promising
directions after this paper: i) It would be interesting if we
can find the communication round that can achieve the best
attack performance; ii) Since the server can save the updates
of the clients in every communication round, it would
be interesting to investigate the possibility of proposing a
new attack approach that can leverage all the information
collected during training to achieve better performance.
6 R ELATED WORK
6.1 Privacy Attacks in FL
We summarize the existing privacy attacks and compare
them with our proposed source inference attacks in FL in
Table 5. For each of the existing attacks, we introduce them
with more detailed descriptions as follows.
Data reconstruction attacks. This type of attack aims to re-
construct individual client’s class-wise training data records
that represent a whole class or instance-wise training data
records. Class-wise data reconstruction attacks are usually
achieved by GANs techniques [72] that leverage the model
updates as discriminators to generate generic representa-
tions of class-wise data records [11], [14]. Instance-wise data
reconstruction attacks leverage optimization techniques to
iteratively optimize a dummy data record with a label so
that the gradients on the dummy record are close to the
gradients uploaded from a client [10], [73]. The instance-
wise data reconstruction attacks firstly were limited in a set-
ting where the mini-batch is one [10], [73], and recent works
[40], [74] have relaxed this assumption and demonstrate the
effectiveness in the setting of large mini-batch sizes.
Property inference attacks. This type of attack aims to
infer the properties of clients’ training data, e.g., inferring
when a particular person first appears in a client’s photos or
when the client begins to visit a certain type of location [8].
Property inference attacks are usually achieved by training
a binary classifier that takes as input the parameters of the
model and outputs whether the model’s training data has
the target property or not [15], [75].
Feature inference attacks. This type of attack targets
vertical FL (see Section 2.1 for a detailed introduction of
vertical FL) and aim to infer the feature values of clients [71].
Feature inference attacks leverage the solving of mathemat-
ical equality for simple models such as logistic regression
because the prediction output and the input containing the
target features can be constructed as a set of equations. For
complex models such as neural networks, a generative re-
gression network is trained through an optimization process
and is leveraged to compute the target features. Recently,
the work [76] demonstrates that feature inference attacks
can also reconstruct the private input on centralized trained
models based their explanation values of predictions. The
work [76] shows that an adversary having an auxiliary
dataset and black-box access to the model can successfully
attack popular Machine Learning as a Service platforms
such as Google Cloud and IBM aix360.
Preference profiling attacks. This type of attack aims to
infer the data preference of clients, e.g., in the FL application
of recommender systems, a malicious server infers which
item a client likes or dislikes [13]. The attack intuition is the
sensitivity of gradients during the FL training reflects the
sample size of a class: a small gradient change indicates the
sample size of a class is large. The statistical heterogeneity
in FL amplifies the gradient sensitivity across classes and
thus benefits the preference profiling attacks.
Membership inference attacks. Membership inference
attacks (MIAs), which are the most related inference attacks
to the proposed SIAs, aim to identify the training data of a
model [19]. MIAs are usually achieved by training a binary
classifier [16] or leveraging a threshold of a metric such as
prediction confidence [38] and prediction entropy [77] to
decide whether a data record is training data or not. Recent
studies [78], [79], [80], [81], [82], [83] have shown that many
different types of model such as semantic segmentation
models, text-to-image generation models, graph neural net-
works, multi-modal models, and recommender systems are
vulnerable to MIAs. Currently, most studies of MIAs [16],
[38], [84], [85], [86], [87] are investigated under centralized
settings where one dataset containing all training instances
is used for training the models.
In the context of FL, MIAs were first investigated in
FedSGD where a malicious client or server can infer whether
or not a specific location profile was used for federated
training based on the observation of the non-zero gradients
of the embedding layer of the global model [8]. Then, MIAs
are investigated in FedAvg where a malicious client can
passively infer the membership privacy of the FL system or
actively craft her updated model parameters to steal more
membership privacy [9]. However, because the purpose of
MIAs is to distinguish training data from testing data of the
FL system, the existing research of MIAs ignores exploring
the source client of the training data. In this paper, we
propose SIAs to fill this gap and demonstrate SIAs are
effective in different FL frameworks.
6.2 Privacy Defenses in FL
Cryptography-based defense. Cryptography-based de-
fense leverages homomorphic encryption (HE) or secure
multiparty computation (SMC) techniques to defend against
privacy leakage in FL. In HE-based FL systems [88], [89],
[90], [91], [92], [93], [94], [95], each local client encrypts
their gradients or model parameters using a public key

15
TABLE 5: A summary of privacy attacks in federated learning.
Privacy Attacks Attacker FL framework Attack Goal
Client Server Horizontal FL Vertical FL
Data reconstruction attacks [40] Reconstruct training data of the client
Property inference attacks [8] Infer property of other clients’ training data
Feature inference attacks [71] Infer features in other clients’ training data
Preference profiling attacks [13] Infer the training data distribution of the client
Membership inference attacks [9] Infer the membership information of a data sample
Source inference attacks. (Ours) Infer the source information of a training sample
: applicable; : not applicable
and sends the ciphertexts to a central server. In SMC-based
FL systems [96], [97], each client adds random values to
their gradients or model parameters for masking their true
updates. The cryptography-based FL systems prevent the
server from knowing the clients’ local updates. Under this
type of defense, the existing privacy attacks like member-
ship inference attacks [9], property inference attacks [8], data
reconstruction attacks [10], [40], [73], preference profiling
attack [13], as well as our proposed SIAs cannot be immedi-
ately mounted. However, HE and SMC are computationally
expensive that increase substantial additional communica-
tion and computation overheads in the FL system, which
may prevent clients with limited computing resources and
bandwidth from participating in FL. [21].
Differential privacy. Compared to cryptography-based
methods, DP is more computationally efficient, adding noise
directly to the private data to trade off privacy and util-
ity [21]. Existing works of privacy-preserving FL mainly
focus on either the centralized DP mechanism that requires
a trusted central server [61], [62] to add noise to the local
updates or a local differential privacy mechanism where
each client perturbs its updates before sending it to an
untrusted aggregator [12], [98]. These privacy-preserving
methods [22], [61], [62], [92], [96], [98] have been evaluated
and demonstrated to be effective for mitigating privacy
attacks in FL. However, because of the perturbations caused
by DP noise, the FL systems inevitably sacrifice the utilities
of the models for mitigating privacy attacks [86], [99].
In this paper, we investigate whether the most widely
used DP mechanism DP-SGD [65] can help to mitigate SIAs
in FL. The intuition is that the semi-honest server will be
difficult to infer private information of a single instance
from the differentially private local models because they
do not contain details of the local datasets. However, as
demonstrated in our experiments, applying vanilla DP in
FL does not produce satisfactory results for mitigating SIAs,
since it suffers from an unacceptable trade-off between
model accuracy and defense effectiveness against SIAs. Re-
cently, several DP mechanisms [86], [100], [101] have been
designed particularly for FL training that can achieve better
privacy-utility trade-offs than DP-SGD. For such advanced
DP mechanisms in FL, it is worth investigating whether they
can effectively mitigate our proposed SIAs while maintain-
ing the high utility of the FL model.
7 C ONCLUSION
In this paper, we propose a new inference attack called
SIAs in federated learning, which allows an honest-but-
curious server to identify the source client of a training
record. We adopt the Bayesian theorem to derive an infer-
ence method enabling the server to gain significant source
information of the training data during each communica-
tion round. We propose F EDSGD-SIA, F EDAVG-SIA, and
FEDMD-SIA showing that in existing FL frameworks, the
clients sharing gradients, model parameters, or the predic-
tions on a public dataset will lead to source information
leakage to the server. We evaluate SIAs on many datasets
under different federated settings. The comprehensive ex-
perimental results validate the effectiveness of SIAs. We
also conduct a detailed ablation study to investigate how
the non-IID data distribution and local epochs impact the
performance of SIAs. We identify that the overfitting of local
models is the key factor contributing to the success of SIAs.
We evaluate differential privacy as a defense mechanism to
mitigate SIAs, but the experimental results suggest differ-
ential privacy is not a good solution because of the unac-
ceptable trade-offs between model utilities and the defense
effectiveness against SIAs. We discuss the limitations of our
proposed SIAs and potential research opportunities. The in-
vestigation of finding appropriate regularization techniques
as a defense against SIAs to achieve satisfactory privacy-
utility trade-offs becomes our future work.
ACKNOWLEDGEMENT
Dr. Xuyun Zhang is supported only by ARC DECRA Grant
DE210101458. The work of K.-K. R. Choo was supported
only by the Cloud Technology Endowed Professorship.
REFERENCES
[1] A. Mantelero, “The EU proposal for a General Data Protection
Regulation and the roots of the ‘right to be forgotten’,” Computer
Law & Security Review, 2013.
[2] L. de la Torre, “A guide to the california consumer privacy act of
2018,” Available at SSRN 3275571, 2018.
[3] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A.
y Arcas, “Communication-efficient learning of deep networks
from decentralized data,” in International Conference on Artificial
Intelligence and Statistics. PMLR, 2017, pp. 1273–1282.
[4] V . Smith, C.-K. Chiang, M. Sanjabi, and A. S. Talwalkar, “Feder-
ated multi-task learning,” Advances in neural information processing
systems, vol. 30, 2017.
[5] S. Caldas, S. M. K. Duddu, P . Wu, T. Li, J. Kone ˇcn`y, H. B.
McMahan, V . Smith, and A. Talwalkar, “Leaf: A benchmark for
federated settings,” arXiv preprint arXiv:1812.01097, 2018.

16
[6] K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman,
V . Ivanov, C. Kiddon, J. Kone ˇcn`y, S. Mazzocchi, B. McMahan
et al. , “Towards federated learning at scale: System design,”
Proceedings of Machine Learning and Systems , vol. 1, pp. 374–388,
2019.
[7] Y. Xu, L. Ma, F. Yang, Y. Chen, K. Ma, J. Yang, X. Yang, Y. Chen,
C. Shu, Z. Fan et al., “A collaborative online ai engine for ct-based
covid-19 diagnosis,” medRxiv, 2020.
[8] L. Melis, C. Song, E. De Cristofaro, and V . Shmatikov, “Exploiting
unintended feature leakage in collaborative learning,” in 2019
IEEE Symposium on Security and Privacy (S&P) . IEEE, 2019, pp.
691–706.
[9] M. Nasr, R. Shokri, and A. Houmansadr, “Comprehensive pri-
vacy analysis of deep learning: Passive and active white-box
inference attacks against centralized and federated learning,” in
2019 IEEE symposium on security and privacy (S&P) . IEEE, 2019,
pp. 739–753.
[10] L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,”
Advances in Neural Information Processing Systems , vol. 32, 2019.
[11] Z. Wang, M. Song, Z. Zhang, Y. Song, Q. Wang, and H. Qi,
“Beyond inferring class representatives: User-level privacy leak-
age from federated learning,” in IEEE INFOCOM 2019-IEEE
Conference on Computer Communications . IEEE, 2019, pp. 2512–
2520.
[12] S. Truex, N. Baracaldo, A. Anwar, T. Steinke, H. Ludwig,
R. Zhang, and Y. Zhou, “A hybrid approach to privacy-
preserving federated learning,” in Proceedings of the 12th ACM
Workshop on Artificial Intelligence and Security , 2019, pp. 1–11.
[13] C. Zhou, Y. Gao, A. Fu, K. Chen, Z. Dai, Z. Zhang, M. Xue,
and Y. Zhang, “Ppa: Preference profiling attack against federated
learning,” in Network and Distributed Systems Security Symposium
2022. Internet Society, 2022.
[14] B. Hitaj, G. Ateniese, and F. Perez-Cruz, “Deep models under the
gan: information leakage from collaborative deep learning,” in
Proceedings of the 2017 ACM SIGSAC Conference on Computer and
Communications Security. ACM, 2017, pp. 603–618.
[15] K. Ganju, Q. Wang, W. Yang, C. A. Gunter, and N. Borisov,
“Property inference attacks on fully connected neural networks
using permutation invariant representations,” inProceedings of the
2018 ACM SIGSAC Conference on Computer and Communications
Security. ACM, 2018, pp. 619–633.
[16] R. Shokri, M. Stronati, C. Song, and V . Shmatikov, “Membership
inference attacks against machine learning models,” in 2017 IEEE
symposium on security and privacy (S&P) . IEEE, 2017, pp. 3–18.
[17] N. Homer, S. Szelinger, M. Redman, D. Duggan, W. Tembe,
J. Muehling, J. V . Pearson, D. A. Stephan, S. F. Nelson, and D. W.
Craig, “Resolving individuals contributing trace amounts of dna
to highly complex mixtures using high-density snp genotyping
microarrays,” PLoS genetics, vol. 4, no. 8, 2008.
[18] A. Pustozerova and R. Mayer, “Information leaks in federated
learning,” in Proceedings of the Network and Distributed System
Security Symposium, vol. 10, 2020.
[19] H. Hu, Z. Salcic, L. Sun, G. Dobbie, P . S. Yu, and X. Zhang,
“Membership inference attacks on machine learning: A survey,”
ACM Computing Surveys (CSUR), vol. 54, no. 11s, pp. 1–37, 2022.
[20] D. Devakumar, G. Shannon, S. S. Bhopal, and I. Abubakar,
“Racism and discrimination in covid-19 responses,” The Lancet ,
vol. 395, no. 10231, p. 1194, 2020.
[21] L. Lyu, H. Yu, X. Ma, C. Chen, L. Sun, J. Zhao, Q. Yang, and S. Y.
Philip, “Privacy and robustness in federated learning: Attacks
and defenses,” IEEE transactions on neural networks and learning
systems, 2022.
[22] D. Li and J. Wang, “Fedmd: Heterogenous federated learning via
model distillation,” arXiv preprint arXiv:1910.03581, 2019.
[23] H. Hu, Z. Salcic, L. Sun, G. Dobbie, and X. Zhang, “Source
inference attacks in federated learning,” in2021 IEEE International
Conference on Data Mining (ICDM). IEEE, 2021, pp. 1102–1107.
[24] Q. Yang, Y. Liu, T. Chen, and Y. Tong, “Federated machine learn-
ing: Concept and applications,” ACM Transactions on Intelligent
Systems and Technology (TIST), vol. 10, no. 2, pp. 1–19, 2019.
[25] Q. Li, Z. Wen, Z. Wu, S. Hu, N. Wang, Y. Li, X. Liu, and B. He, “A
survey on federated learning systems: vision, hype and reality
for data privacy and protection,” IEEE Transactions on Knowledge
and Data Engineering, 2021.
[26] Y. Cheng, Y. Liu, T. Chen, and Q. Yang, “Federated learning
for privacy-preserving ai,” Communications of the ACM , vol. 63,
no. 12, pp. 33–36, 2020.
[27] L. Lyu, H. Yu, and Q. Yang, “Threats to federated learning: A
survey,” arXiv preprint arXiv:2003.02133, 2020.
[28] X. Yin, Y. Zhu, and J. Hu, “A comprehensive survey of privacy-
preserving federated learning: A taxonomy, review, and future
directions,” ACM Computing Surveys (CSUR) , vol. 54, no. 6, pp.
1–36, 2021.
[29] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and
V . Smith, “Federated optimization in heterogeneous networks,”
Proceedings of Machine Learning and Systems , vol. 2, pp. 429–450,
2020.
[30] S. P . Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T.
Suresh, “Scaffold: Stochastic controlled averaging for federated
learning,” in International Conference on Machine Learning. PMLR,
2020, pp. 5132–5143.
[31] T. Lin, L. Kong, S. U. Stich, and M. Jaggi, “Ensemble distillation
for robust model fusion in federated learning,”Advances in Neural
Information Processing Systems, vol. 33, pp. 2351–2363, 2020.
[32] H. Chang, V . Shejwalkar, R. Shokri, and A. Houmansadr,
“Cronus: Robust and heterogeneous collaborative learning with
black-box knowledge transfer,” arXiv preprint arXiv:1912.11279 ,
2019.
[33] S. Wang, S. Nepal, K. Moore, M. Grobler, C. Rudolph, and
A. Abuadbba, “Octopus: Overcoming performance and privati-
zation bottlenecks in distributed learning,” IEEE Transactions on
Parallel and Distributed Systems , vol. 33, no. 12, pp. 3460–3477,
2022.
[34] H. Lee, J. Kim, S. Ahn, R. Hussain, S. Cho, and J. Son, “Digestive
neural networks: A novel defense strategy against inference
attacks in federated learning,” computers & security , vol. 109, p.
102378, 2021.
[35] J. Zhang, J. Zhang, J. Chen, and S. Yu, “Gan enhanced mem-
bership inference: A passive local attack in federated learning,”
in ICC 2020-2020 IEEE International Conference on Communications
(ICC). IEEE, 2020, pp. 1–6.
[36] J. Chen, J. Zhang, Y. Zhao, H. Han, K. Zhu, and B. Chen, “Beyond
model-level membership privacy leakage: an adversarial ap-
proach in federated learning,” in2020 29th International Conference
on Computer Communications and Networks (ICCCN) . IEEE, 2020,
pp. 1–9.
[37] W. Yuan, C. Yang, Q. V . H. Nguyen, L. Cui, T. He, and H. Yin,
“Interaction-level membership inference attack against federated
recommender systems,” arXiv preprint arXiv:2301.10964, 2023.
[38] A. Salem, Y. Zhang, M. Humbert, M. Fritz, and M. Backes,
“Ml-leaks: Model and data independent membership inference
attacks and defenses on machine learning models,” in Network
and Distributed Systems Security Symposium 2019. Internet Society,
2019.
[39] L. Fowl, J. Geiping, W. Czaja, M. Goldblum, and T. Goldstein,
“Robbing the fed: Directly obtaining private data in federated
learning with modified models,” arXiv preprint arXiv:2110.13057,
2021.
[40] F. Boenisch, A. Dziedzic, R. Schuster, A. S. Shamsabadi,
I. Shumailov, and N. Papernot, “When the curious abandon
honesty: Federated learning is not private,” arXiv preprint
arXiv:2112.02918, 2021.
[41] D. C. Montgomery and G. C. Runger, Applied statistics and proba-
bility for engineers. John Wiley & Sons, 2010.
[42] A. Sablayrolles, M. Douze, C. Schmid, Y. Ollivier, and H. J ´egou,
“White-box vs black-box: Bayes optimal strategies for member-
ship inference,” in International Conference on Machine Learning .
PMLR, 2019, pp. 5558–5567.
[43] J. Ba and R. Caruana, “Do deep nets really need to be deep?”
Advances in neural information processing systems , vol. 27, 2014.
[44] G. Hinton, O. Vinyals, J. Dean et al., “Distilling the knowledge in
a neural network,” arXiv preprint arXiv:1503.02531 , vol. 2, no. 7,
2015.
[45] E. J. Crowley, G. Gray, and A. J. Storkey, “Moonshine: Distilling
with cheap convolutions,” Advances in Neural Information Process-
ing Systems, vol. 31, 2018.
[46] X. Li, K. Huang, W. Yang, S. Wang, and Z. Zhang, “On the con-
vergence of fedavg on non-iid data,” in International Conference on
Learning Representations. PMLR, 2019.
[47] B. Jayaraman and D. Evans, “Evaluating differentially private
machine learning in practice,” in 28th USENIX Security Sympo-
sium (USENIX Security 19). USENIX Association, 2019, pp. 1895–
1912.

17
[48] T. Li, A. K. Sahu, A. Talwalkar, and V . Smith, “Federated learning:
Challenges, methods, and future directions,” IEEE Signal Process-
ing Magazine, vol. 37, no. 3, pp. 50–60, 2020.
[49] C. Xie, K. Huang, P .-Y. Chen, and B. Li, “Dba: Distributed
backdoor attacks against federated learning,” in International
Conference on Learning Representations. PMLR, 2019.
[50] E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V . Shmatikov,
“How to backdoor federated learning,” in International Conference
on Artificial Intelligence and Statistics. PMLR, 2020, pp. 2938–2948.
[51] M. Yurochkin, M. Agarwal, S. Ghosh, K. Greenewald, N. Hoang,
and Y. Khazaeni, “Bayesian nonparametric federated learning of
neural networks,” in International Conference on Machine Learning.
PMLR, 2019, pp. 7252–7261.
[52] T.-M. H. Hsu, H. Qi, and M. Brown, “Measuring the effects
of non-identical data distribution for federated visual classifica-
tion,” arXiv preprint arXiv:1909.06335, 2019.
[53] C. Song, T. Ristenpart, and V . Shmatikov, “Machine learning
models that remember too much,” in Proceedings of the 2017 ACM
SIGSAC Conference on Computer and Communications Security .
ACM, 2017, pp. 587–601.
[54] N. Carlini, C. Liu, ´U. Erlingsson, J. Kos, and D. Song, “The se-
cret sharer: Evaluating and testing unintended memorization in
neural networks,” in 28th USENIX Security Symposium (USENIX
Security 19). USENIX Association, 2019, pp. 267–284.
[55] S. K. Murakonda and R. Shokri, “Ml privacy meter: Aiding reg-
ulatory compliance by quantifying the privacy risks of machine
learning,” arXiv preprint arXiv:2007.09339, 2020.
[56] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Under-
standing deep learning (still) requires rethinking generalization,”
Communications of the ACM, vol. 64, no. 3, pp. 107–115, 2021.
[57] M. Hardt, B. Recht, and Y. Singer, “Train faster, generalize better:
Stability of stochastic gradient descent,” in International conference
on machine learning. PMLR, 2016, pp. 1225–1234.
[58] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V . Chandra, “Feder-
ated learning with non-iid data,” arXiv preprint arXiv:1806.00582,
2018.
[59] C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating
noise to sensitivity in private data analysis,” in Theory of cryptog-
raphy conference. Springer, 2006, pp. 265–284.
[60] R. Shokri and V . Shmatikov, “Privacy-preserving deep learning,”
in Proceedings of the 2015 ACM SIGSAC Conference on Computer
and Communications Security. ACM, 2015, pp. 1310–1321.
[61] R. C. Geyer, T. Klein, and M. Nabi, “Differentially private
federated learning: A client level perspective,” arXiv preprint
arXiv:1712.07557, 2017.
[62] H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang, “Learning
differentially private recurrent language models,” in International
Conference on Learning Representations. PMLR, 2018.
[63] M. Naseri, J. Hayes, and E. De Cristofaro, “Toward robustness
and privacy in federated learning: Experimenting with local and
central differential privacy,”arXiv preprint arXiv:2009.03561, 2020.
[64] M. A. Rahman, T. Rahman, R. Lagani `ere, N. Mohammed, and
Y. Wang, “Membership inference attack against differentially
private deep learning model.” Trans. Data Priv. , vol. 11, no. 1,
pp. 61–79, 2018.
[65] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov,
K. Talwar, and L. Zhang, “Deep learning with differential pri-
vacy,” in Proceedings of the 2016 ACM SIGSAC Conference on
Computer and Communications Security. ACM, 2016, pp. 308–318.
[66] H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang, “Learning
differentially private recurrent language models,” arXiv preprint
arXiv:1710.06963, 2017.
[67] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and
R. Salakhutdinov, “Dropout: a simple way to prevent neural net-
works from overfitting,” The journal of machine learning research ,
vol. 15, no. 1, pp. 1929–1958, 2014.
[68] K. Simonyan and A. Zisserman, “Very deep convolutional
networks for large-scale image recognition,” arXiv preprint
arXiv:1409.1556, 2014.
[69] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning
for image recognition,” in Proceedings of the IEEE conference on
computer vision and pattern recognition, 2016, pp. 770–778.
[70] N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss,
K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson et al. ,
“Extracting training data from large language models,” in 30th
USENIX Security Symposium (USENIX Security 21) . USENIX
Association, 2021, pp. 2633–2650.
[71] X. Luo, Y. Wu, X. Xiao, and B. C. Ooi, “Feature inference attack
on model predictions in vertical federated learning,” in2021 IEEE
37th International Conference on Data Engineering (ICDE) . IEEE,
2021, pp. 181–192.
[72] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-
Farley, S. Ozair, A. C. Courville, and Y. Bengio, “Generative
adversarial nets,” in Proceedings of the 28th International Conference
on Neural Information Processing Systems, 2014.
[73] B. Zhao, K. R. Mopuri, and H. Bilen, “idlg: Improved deep
leakage from gradients,” arXiv preprint arXiv:2001.02610, 2020.
[74] H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, and
P . Molchanov, “See through gradients: Image batch recovery
via gradinversion,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2021, pp. 16 337–16 346.
[75] Z. Wang, Y. Huang, M. Song, L. Wu, F. Xue, and K. Ren,
“Poisoning-assisted property inference attack against federated
learning,” IEEE Transactions on Dependable and Secure Computing ,
2022.
[76] X. Luo, Y. Jiang, and X. Xiao, “Feature inference attack on shapley
values,” in Proceedings of the 2022 ACM SIGSAC Conference on
Computer and Communications Security, 2022, pp. 2233–2247.
[77] L. Song and P . Mittal, “Systematic evaluation of privacy risks of
machine learning models.” in USENIX Security Symposium, vol. 1,
no. 2, 2021, p. 4.
[78] T. Chobola, D. Usynin, and G. Kaissis, “Membership inference
attacks against semantic segmentation models,” arXiv preprint
arXiv:2212.01082, 2022.
[79] Y. Wu, N. Yu, Z. Li, M. Backes, and Y. Zhang, “Membership
inference attacks against text-to-image generation models,” arXiv
preprint arXiv:2210.00968, 2022.
[80] M. Conti, J. Li, S. Picek, and J. Xu, “Label-only membership
inference attack against node-level graph neural networks,” in
Proceedings of the 15th ACM Workshop on Artificial Intelligence and
Security, 2022, pp. 1–12.
[81] P . Hu, Z. Wang, R. Sun, H. Wang, and M. Xue, “Mˆ 4i:
Multi-modal models membership inference,” arXiv preprint
arXiv:2209.06997, 2022.
[82] Z. Wang, N. Huang, F. Sun, P . Ren, Z. Chen, H. Luo, M. de Ri-
jke, and Z. Ren, “Debiasing learning for membership inference
attacks against recommender systems,” in Proceedings of the 28th
ACM SIGKDD Conference on Knowledge Discovery and Data Mining,
2022, pp. 1959–1968.
[83] G. Zhang, B. Liu, T. Zhu, M. Ding, and W. Zhou, “Label-
only membership inference attacks and defenses in semantic
segmentation models,” IEEE Transactions on Dependable and Secure
Computing, 2022.
[84] S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha, “Privacy risk
in machine learning: Analyzing the connection to overfitting,”
in 2018 IEEE 31st computer security foundations symposium (CSF) .
IEEE, 2018, pp. 268–282.
[85] X. He, H. Liu, N. Z. Gong, and Y. Zhang, “Semi-leak: Membership
inference attacks against semi-supervised learning,” in Computer
Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, Oc-
tober 23–27, 2022, Proceedings, Part XXXI . Springer, 2022, pp.
365–381.
[86] X. Yuan and L. Zhang, “Membership inference attacks and
defenses in neural network pruning,” in 31st USENIX Security
Symposium (USENIX Security 22), 2022, pp. 4561–4578.
[87] H. Hu, Z. Salcic, G. Dobbie, J. Chen, L. Sun, and X. Zhang,
“Membership inference via backdooring,” in Proceedings of the
Thirty-First International Joint Conference on Artificial Intelligence,
IJCAI 2022. ijcai.org, 2022, pp. 3832–3838.
[88] C. Zhang, S. Li, J. Xia, W. Wang, F. Yan, and Y. Liu, “{BatchCrypt}:
Efficient homomorphic encryption for {Cross-Silo} federated
learning,” in 2020 USENIX annual technical conference (USENIX
ATC 20), 2020, pp. 493–506.
[89] K. Cheng, T. Fan, Y. Jin, Y. Liu, T. Chen, D. Papadopoulos, and
Q. Yang, “Secureboost: A lossless federated learning framework,”
IEEE Intelligent Systems, vol. 36, no. 6, pp. 87–98, 2021.
[90] S. Hardy, W. Henecka, H. Ivey-Law, R. Nock, G. Patrini, G. Smith,
and B. Thorne, “Private federated learning on vertically parti-
tioned data via entity resolution and additively homomorphic
encryption,” arXiv preprint arXiv:1711.10677, 2017.
[91] C. Liu, S. Chakraborty, and D. Verma, “Secure model fusion for
distributed learning using partial homomorphic encryption,” in
Policy-Based Autonomic Data Governance. Springer, 2019, pp. 154–
179.

18
[92] Y. Liu, Y. Kang, C. Xing, T. Chen, and Q. Yang, “A secure
federated transfer learning framework,” IEEE Intelligent Systems,
vol. 35, no. 4, pp. 70–82, 2020.
[93] V . Nikolaenko, U. Weinsberg, S. Ioannidis, M. Joye, D. Boneh,
and N. Taft, “Privacy-preserving ridge regression on hundreds
of millions of records,” in 2013 IEEE symposium on security and
privacy. IEEE, 2013, pp. 334–348.
[94] Y. Zheng, S. Lai, Y. Liu, X. Yuan, X. Yi, and C. Wang, “Aggregation
service for federated learning: An efficient, secure, and more
resilient realization,” IEEE Transactions on Dependable and Secure
Computing, 2022.
[95] N. M. Jebreel, J. Domingo-Ferrer, A. Blanco-Justicia, and
D. S ´anchez, “Enhanced security and privacy via fragmented
federated learning,” IEEE Transactions on Neural Networks and
Learning Systems, 2022.
[96] K. Bonawitz, V . Ivanov, B. Kreuter, A. Marcedone, H. B. McMa-
han, S. Patel, D. Ramage, A. Segal, and K. Seth, “Practical
secure aggregation for privacy-preserving machine learning,” in
proceedings of the 2017 ACM SIGSAC Conference on Computer and
Communications Security, 2017, pp. 1175–1191.
[97] P . Mohassel and Y. Zhang, “Secureml: A system for scalable
privacy-preserving machine learning,” in 2017 IEEE symposium
on security and privacy (SP). IEEE, 2017, pp. 19–38.
[98] L. Sun, J. Qian, and X. Chen, “Ldp-fl: Practical private aggre-
gation in federated learning with local differential privacy,” in
Proceedings of the Thirtieth International Joint Conference on Artificial
Intelligence (IJCAI-21), 2021.
[99] K. Wei, J. Li, M. Ding, C. Ma, H. H. Yang, F. Farokhi, S. Jin, T. Q.
Quek, and H. V . Poor, “Federated learning with differential pri-
vacy: Algorithms and performance analysis,” IEEE Transactions
on Information Forensics and Security, vol. 15, pp. 3454–3469, 2020.
[100] K. Wei, J. Li, M. Ding, C. Ma, H. Su, B. Zhang, and H. V . Poor,
“User-level privacy-preserving federated learning: Analysis and
performance optimization,” IEEE Transactions on Mobile Comput-
ing, vol. 21, no. 9, pp. 3388–3401, 2021.
[101] J. Zheng, H. Tian, W. Ni, W. Ni, and P . Zhang, “Balancing ac-
curacy and integrity for reconfigurable intelligent surface-aided
over-the-air federated learning,” IEEE Transactions on Wireless
Communications, vol. 21, no. 12, pp. 10 964–10 980, 2022.
Hongsheng Hu is currently a PhD at Faculty
of Engineering, University of Auckland, New
Zealand. His research focuses on AI privacy
and security, especially membership inference
attacks, differential privacy, and inference at-
tacks in the context of federated learning. He has
published 8 international refereed journal and
conference papers, including ACM Computing
Surveys, IJCAI, and ICDM.
Xuyun Zhang is currently working as a senior
lecturer in School of Computing at Macquarie
University (Sydney, Australia). Besides, he has
the working experience in University of Auckland
and NICTA (now Data61, CSIRO). He received
his PhD degree in Computer and Information
Science from University of Technology Sydney
(UTS) in 2014, and his MEng and BSc degrees
from Nanjing University. His research interests
include scalable and secure machine learning,
big data mining and analytics, big data privacy
and cyber security, cloud/edge/service computing and IoT, etc. He is
the recipient of 2021 ARC DECRA Award and several other prestigious
awards, and has been listed as one of the Clarivate 2021 Highly Cited
Researchers.
Zoran Salcic (Life Senior Member, IEEE) re-
ceived the B.E., M.E., and Ph.D. degrees in
electrical and computer engineering from Sara-
jevo University in 1972, 1974, and 1976, re-
spectively. He is a Professor and the Chair of
computer systems engineering with University of
Auckland, New Zealand. He has published more
than 400 peer-reviewed journal and conference
papers, and several books. His main research in-
terests include various aspects of cyber-physical
systems that include complex digital systems
design, custom-computing machines, design automation tools, hard-
ware–software co-design, formal models of computation, sensor net-
works and Internet of Things, and languages for concurrent and dis-
tributed systems and their applications such as industrial automation,
intelligent buildings and environments, and collaborative systems with
service robotics, IoT, big data processing and many more. He is a
Fellow of the Royal Society of New Zealand. He was a recipient of the
Alexander von Humboldt Research Award in 2010.
Lichao Sun is currently an Assistant Professor
in the Department of Computer Science and
Engineering at Lehigh University. Before that, he
received my Ph.D. degree in Computer Science
at University of Illinois, Chicago in 2020, under
the supervision of Prof. Philip S. Yu. Further
before, he obtained M.S. and B.S. from Univer-
sity of Nebraska Lincoln. His research interests
include security and privacy in deep learning and
data mining. He mainly focuses on AI security
and privacy, social networks, and natural lan-
guage processing applications. He has published more than 45 research
articles in top conferences and journals like CCS, USENIX-Security,
NeurIPS, KDD, ICLR, AAAI, IJCAI, ACL, NAACL, TII, TNNLS, TMC.
Kim-Kwang Raymond Choo (Senior Member,
IEEE) received the Ph.D. degree in information
security from the Queensland University of Tech-
nology, Australia, in 2006. He currently holds
the Cloud Technology Endowed Professorship at
The University of Texas at San Antonio. He was
a recipient of the 2022 IEEE Hyper-Intelligence
Technical Committee (HITC) Award for Excel-
lence in Hyper-Intelligence (Technical Achieve-
ment), the 2022 IEEE Technical Committee on
Homeland Security (TCHS) Research and Inno-
vation Award in 2022, the 2022 IEEE Technical Committee on Secure
and Dependable Measurement (TCSDM) Mid-Career Award, and the
2019 IEEE Technical Committee on Scalable Computing (TCSC) Award
for Excellence in Scalable Computing (Middle Career Researcher).
Gillian Dobbie obtained a PhD in Computer
Science from the University of Melbourne. She
joined the Department of Computer Science at
the University of Auckland in 2001, and is now
a Professor. Gill’s primary research area covers
the modelling, management and efficient pro-
cessing of data. She has worked on a diverse
range of topics including formal methods, log-
ical foundations of data models, modeling of
semistructured data, data warehousing, data pri-
vacy, data mining, and recurrent pattern mining.
Her most recent research projects have focused on adversarial attacks
and defences. She has published over 160 international refereed journal
and conference papers.