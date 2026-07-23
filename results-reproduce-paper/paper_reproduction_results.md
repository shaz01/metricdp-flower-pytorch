# Non-CIA Paper Reproduction — H100 Detailed Evaluation

## Protocol

- 122 final global models: 90 homogeneous main, 18 non-IID main, 12 Appendix-B noise=0.003, and 2 Appendix-B FedAvg global-DP high-noise checks.
- Main settings: 4 clients, 20 rounds, 5 local epochs, batch 32, learning rate 0.001, clipping norm 5, noise 0.01.
- Homogeneous seeds: 42–46. Non-IID and Appendix-B seed: 42.
- Final model is saved transiently, evaluated on the server final-test split and all client held-out splits, then deleted after JSON/NPZ validation.
- CIA experiments are excluded.
- Previous accuracy-only results remain under `results-old-20260722-160446/`.
- H100 execution uses two concurrent experiment lanes; each experiment runs four concurrent clients.

## Artifact schema

- Run history: `results/runs/<run>.json`
- Detailed metrics: `results/evaluations/<run>.evaluation.json`
- Raw labels/probabilities/predictions: `results/evaluations/<run>.predictions.npz`
- Full log: `results/logs/<run>.log`

## Per-run results

| Run | Status | Final accuracy | Macro F1 | Macro precision | Macro OVR AUC | Duration | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-42` | PASS | 0.9625 | 0.9636085848813208 | 0.9662177633046698 | 0.9921465928320606 | 0s | validated-existing-detailed-artifacts |
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-42` | PASS | 0.9375 | 0.9500111457668664 | 0.9505185416341604 | 0.9849552292772757 | 117s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-42` | PASS | 0.953125 | 0.9570103368750514 | 0.9585864472761016 | 0.9897814003867121 | 122s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-42` | PASS | 0.5390625 | 0.2846194792116242 | 0.2649840334070253 | 0.7210923750569032 | 107s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-42` | PASS | 0.35 | 0.18783258201064262 | 0.17430762009075262 | 0.5297235800371493 | 114s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-42` | PASS | 0.9390625 | 0.9274761882520608 | 0.9467036821504587 | 0.9906313095956352 | 131s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-42` | PASS | 0.9703125 | 0.953677416699107 | 0.9453248467989034 | 0.9945992463488407 | 143s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-42` | PASS | 0.7484375 | 0.5351664723409768 | 0.5459055882875763 | 0.9186521668144757 | 125s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-42` | PASS | 0.940625 | 0.9282319127692105 | 0.9509968116239531 | 0.9849785733778815 | 124s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-42` | PASS | 0.5546875 | 0.2953689576154475 | 0.27657641819034223 | 0.7364946950947027 | 144s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-42` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 157s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-42` | PASS | 0.9421875 | 0.9112924348109115 | 0.9202233804482112 | 0.9828600237298025 | 112s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-42` | PASS | 0.9375 | 0.9439519609299888 | 0.9499588274044796 | 0.9852097671985509 | 111s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-42` | PASS | 0.7984375 | 0.5824059668986916 | 0.5792344773874745 | 0.9414722059658178 | 145s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-42` | PASS | 0.9390625 | 0.9464252957464324 | 0.9505287782140204 | 0.9834966434705018 | 158s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42` | FAIL |  |  |  |  | 85s | training-failed |
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-42` | PASS | 0.496875 | 0.16795948124400006 | 0.37402190923317685 | 0.5155942186306421 | 123s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-42` | PASS | 0.95625 | 0.9619277108433735 | 0.9703137796887796 | 0.9891394576289261 | 116s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-43` | PASS | 0.9453125 | 0.8934486755214844 | 0.8611847904389887 | 0.987118152626344 | 113s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-43` | PASS | 0.9 | 0.7366817463995899 | 0.914417118580271 | 0.9813108624742863 | 112s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-43` | PASS | 0.9390625 | 0.9390218725670849 | 0.9311321508334791 | 0.9852180859147174 | 112s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-43` | PASS | 0.5328125 | 0.27832063162317355 | 0.2580666237786981 | 0.7285587463706453 | 111s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-43` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 114s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-43` | PASS | 0.9390625 | 0.8913847066208522 | 0.876264449689532 | 0.9821898602828443 | 159s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-43` | PASS | 0.9328125 | 0.9451017791448433 | 0.9448843411074205 | 0.9857342584502125 | 172s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-43` | PASS | 0.7796875 | 0.5801317335127121 | 0.5704082823810778 | 0.9399660798293329 | 116s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-43` | PASS | 0.9390625 | 0.9507422402159245 | 0.9508606668473841 | 0.983689699999476 | 112s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-43` | PASS | 0.5171875 | 0.25778816199376947 | 0.2415843959961607 | 0.731525625211461 | 124s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-43` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 127s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-43` | PASS | 0.928125 | 0.8993952147300253 | 0.9528591307304135 | 0.9846649156315737 | 132s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-43` | PASS | 0.94375 | 0.9534979922925594 | 0.9538844482120221 | 0.9865409845915615 | 116s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-43` | PASS | 0.784375 | 0.5817669326189505 | 0.5823355557794078 | 0.9399844295842208 | 120s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-43` | PASS | 0.934375 | 0.9456899130982315 | 0.9474643658479981 | 0.9816659550832565 | 126s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43` | FAIL |  |  |  |  | 62s | training-failed |
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-43` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 130s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-43` | PASS | 0.953125 | 0.9288309302465054 | 0.9351370447663405 | 0.9856897119880952 | 114s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-44` | PASS | 0.94375 | 0.8976032026030496 | 0.8752025380338795 | 0.9835187545338462 | 103s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-44` | PASS | 0.9046875 | 0.8597678330162406 | 0.9299030669439169 | 0.9829953997612124 | 114s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-44` | PASS | 0.940625 | 0.9295018222540457 | 0.9494451174021514 | 0.984511622952821 | 114s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-44` | PASS | 0.565625 | 0.29912495675239303 | 0.27793103448275863 | 0.7259621209180553 | 146s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-44` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 149s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-44` | PASS | 0.95 | 0.9543537022830294 | 0.9588715457744699 | 0.9845520620996162 | 115s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-44` | PASS | 0.9296875 | 0.9042729453296426 | 0.9042847743964142 | 0.9835135070999061 | 111s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-44` | PASS | 0.7859375 | 0.576609729832659 | 0.5777788064524808 | 0.9352832545004761 | 175s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-44` | PASS | 0.9390625 | 0.9286644999880294 | 0.9517175919254698 | 0.9851696414699718 | 200s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-44` | PASS | 0.5515625 | 0.28465415891639245 | 0.26488884678539854 | 0.724139017467669 | 179s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-44` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 138s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-44` | PASS | 0.95625 | 0.949106967176057 | 0.9414311253388357 | 0.988913937719753 | 119s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-44` | PASS | 0.9484375 | 0.9353483156216404 | 0.9572019895918497 | 0.9884892110101617 | 112s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-44` | PASS | 0.8109375 | 0.7774944880908672 | 0.8527453378446697 | 0.9510623224793374 | 116s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-44` | PASS | 0.9484375 | 0.9533605253263379 | 0.9611470675991056 | 0.9828733820094622 | 118s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44` | FAIL |  |  |  |  | 85s | training-failed |
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-44` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 122s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-44` | PASS | 0.95 | 0.9574677310153537 | 0.9610172400202321 | 0.984179824787563 | 117s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-45` | PASS | 0.9671875 | 0.9682647471549887 | 0.9788349754652707 | 0.9909515230743773 | 107s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-45` | PASS | 0.9328125 | 0.879215346918561 | 0.9487179487179487 | 0.9896757573831612 | 114s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-45` | PASS | 0.95625 | 0.94241333674511 | 0.9392091539512895 | 0.9915026242138091 | 116s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-45` | PASS | 0.553125 | 0.284686352417073 | 0.26611060821587135 | 0.7132480183964907 | 164s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-45` | PASS | 0.35625 | 0.13133640552995393 | 0.08934169278996865 | 0.4989852579158584 | 155s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-45` | PASS | 0.953125 | 0.9390450920127937 | 0.9672115465715974 | 0.9862274724873175 | 113s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-45` | PASS | 0.93125 | 0.7848336869791381 | 0.7826458400064061 | 0.9801326950468475 | 115s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-45` | PASS | 0.7546875 | 0.5521925197497409 | 0.551916040750692 | 0.9107716361282792 | 167s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-45` | PASS | 0.95 | 0.9326866003369942 | 0.9503801024235041 | 0.9884003629889329 | 168s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-45` | PASS | 0.559375 | 0.2979676147693153 | 0.27873588567395907 | 0.7440416476813848 | 123s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-45` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.4812011408110002 | 125s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-45` | PASS | 0.9375 | 0.9025455038092116 | 0.9492177711726602 | 0.9857810693016476 | 116s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-45` | PASS | 0.9515625 | 0.9105096353098924 | 0.9591891072233347 | 0.9876871849665094 | 122s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-45` | PASS | 0.7609375 | 0.5492344175893347 | 0.5608071326821327 | 0.9149414828070497 | 121s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-45` | PASS | 0.946875 | 0.9341809622967834 | 0.9618103289082217 | 0.9870318331644929 | 126s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-45` | FAIL |  |  |  |  | 101s | training-failed |
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-45` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 118s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavg__noise-0.01__seed-46` | PASS | 0.9703125 | 0.9738960954987853 | 0.9814109078590786 | 0.9890877309208963 | 133s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-45` | PASS | 0.9640625 | 0.9622558389460981 | 0.9739168158051724 | 0.989963961996456 | 145s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedavgm__noise-0.01__seed-46` | PASS | 0.921875 | 0.8913899251709125 | 0.9408794049269924 | 0.9915123857114828 | 111s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedmedian__noise-0.01__seed-46` | PASS | 0.9453125 | 0.9561343051655674 | 0.9560308972073678 | 0.9863159397505121 | 113s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedprox__noise-0.01__seed-46` | PASS | 0.5609375 | 0.2990644995429616 | 0.27991473860007227 | 0.7310613409463538 | 158s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedopt__noise-0.01__seed-46` | PASS | 0.359375 | 0.13218390804597702 | 0.08984375 | 0.5 | 158s | trained-evaluated-model-deleted |
| `main__homogeneous__vanilla__fedyogi__noise-0.01__seed-46` | PASS | 0.9640625 | 0.9691078446950222 | 0.9735480297332426 | 0.9914124491827763 | 108s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavg__noise-0.01__seed-46` | PASS | 0.95625 | 0.9630639604161599 | 0.9704437669376694 | 0.9887037146451221 | 107s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedavgm__noise-0.01__seed-46` | PASS | 0.8171875 | 0.773108625098335 | 0.8591047610904827 | 0.9545732479356197 | 121s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedmedian__noise-0.01__seed-46` | PASS | 0.95 | 0.9594204471244278 | 0.965061858076564 | 0.9867421072817529 | 119s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedopt__noise-0.01__seed-46` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 131s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedprox__noise-0.01__seed-46` | PASS | 0.5609375 | 0.3088204950690818 | 0.5284872974673589 | 0.751314242404335 | 136s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavg__noise-0.01__seed-46` | PASS | 0.959375 | 0.9683273043614015 | 0.9761770737002935 | 0.9887387413559358 | 114s | trained-evaluated-model-deleted |
| `main__homogeneous__global-dp__fedyogi__noise-0.01__seed-46` | PASS | 0.9578125 | 0.9640326024106922 | 0.9687082568941205 | 0.9915052984001156 | 121s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedavgm__noise-0.01__seed-46` | PASS | 0.8046875 | 0.8129564698662818 | 0.8459308560416983 | 0.9526717791160086 | 139s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedmedian__noise-0.01__seed-46` | PASS | 0.953125 | 0.9280491695625321 | 0.9127250826956723 | 0.9897334739563759 | 154s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-46` | FAIL |  |  |  |  | 69s | training-failed |
| `main__homogeneous__metric-privacy__fedprox__noise-0.01__seed-46` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.646319860299464 | 123s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedyogi__noise-0.01__seed-46` | PASS | 0.9375 | 0.9509542197370895 | 0.9663183440123094 | 0.9880936601062882 | 117s | trained-evaluated-model-deleted |
| `main__non-iid__vanilla__fedavg__noise-0.01__seed-42` | PASS | 0.9671875 | 0.9683982551343941 | 0.9771259726560358 | 0.9916554171547829 | 119s | trained-evaluated-model-deleted |
| `main__non-iid__vanilla__fedavgm__noise-0.01__seed-42` | PASS | 0.9375 | 0.9034484254978166 | 0.9438026517742052 | 0.9850564965739451 | 119s | trained-evaluated-model-deleted |
| `main__non-iid__vanilla__fedmedian__noise-0.01__seed-42` | PASS | 0.9640625 | 0.9645983526224509 | 0.9661570908582915 | 0.9923328764013435 | 122s | trained-evaluated-model-deleted |
| `main__non-iid__vanilla__fedprox__noise-0.01__seed-42` | PASS | 0.5453125 | 0.2850552539236396 | 0.26438014380143804 | 0.7302288535338339 | 123s | trained-evaluated-model-deleted |
| `main__non-iid__vanilla__fedopt__noise-0.01__seed-42` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 124s | trained-evaluated-model-deleted |
| `main__non-iid__vanilla__fedyogi__noise-0.01__seed-42` | PASS | 0.95 | 0.9333394249638965 | 0.9680755577507423 | 0.9882794324805465 | 149s | trained-evaluated-model-deleted |
| `main__non-iid__global-dp__fedavg__noise-0.01__seed-42` | PASS | 0.95 | 0.955808354863839 | 0.9633995121520347 | 0.9896590525608022 | 151s | trained-evaluated-model-deleted |
| `main__non-iid__global-dp__fedavgm__noise-0.01__seed-42` | PASS | 0.746875 | 0.5374594124152038 | 0.5467031513153424 | 0.9207000408259435 | 128s | trained-evaluated-model-deleted |
| `main__non-iid__global-dp__fedmedian__noise-0.01__seed-42` | PASS | 0.9453125 | 0.9324506026332027 | 0.9489652696174435 | 0.989049183723471 | 126s | trained-evaluated-model-deleted |
| `main__non-iid__global-dp__fedprox__noise-0.01__seed-42` | PASS | 0.56875 | 0.35010676502132665 | 0.3924917402858579 | 0.7553951574294515 | 135s | trained-evaluated-model-deleted |
| `main__non-iid__global-dp__fedopt__noise-0.01__seed-42` | PASS | 0.359375 | 0.13218390804597702 | 0.08984375 | 0.5249810400091682 | 140s | trained-evaluated-model-deleted |
| `main__non-iid__global-dp__fedyogi__noise-0.01__seed-42` | PASS | 0.94375 | 0.7006806518702281 | 0.6998986957642181 | 0.990448056043675 | 132s | trained-evaluated-model-deleted |
| `main__non-iid__metric-privacy__fedavg__noise-0.01__seed-42` | PASS | 0.9625 | 0.965235848299549 | 0.9669374834325761 | 0.9886299504868457 | 122s | trained-evaluated-model-deleted |
| `main__non-iid__metric-privacy__fedavgm__noise-0.01__seed-42` | PASS | 0.7875 | 0.5680381364133591 | 0.5774906759361127 | 0.939379240661667 | 128s | trained-evaluated-model-deleted |
| `main__non-iid__metric-privacy__fedmedian__noise-0.01__seed-42` | PASS | 0.95 | 0.9381989857692548 | 0.9580344186195673 | 0.9864395470807296 | 131s | trained-evaluated-model-deleted |
| `main__non-iid__metric-privacy__fedopt__noise-0.01__seed-42` | FAIL |  |  |  |  | 59s | training-failed |
| `main__non-iid__metric-privacy__fedprox__noise-0.01__seed-42` | PASS | 0.4953125 | 0.18974656779534826 | 0.33376918416801293 | 0.4843231289412313 | 133s | trained-evaluated-model-deleted |
| `main__non-iid__metric-privacy__fedyogi__noise-0.01__seed-42` | PASS | 0.9546875 | 0.9248096981708107 | 0.9064419171650939 | 0.9882940697354269 | 132s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedavg__noise-0.003__seed-42` | PASS | 0.95 | 0.9533476956720386 | 0.9594361707133445 | 0.9885035598510519 | 109s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedavgm__noise-0.003__seed-42` | PASS | 0.76875 | 0.5500009825383289 | 0.5476294013701778 | 0.9354849305478233 | 129s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedmedian__noise-0.003__seed-42` | PASS | 0.940625 | 0.9316905790923078 | 0.9228260272467589 | 0.9887668373005036 | 129s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedprox__noise-0.003__seed-42` | PASS | 0.553125 | 0.29243196447369174 | 0.27223444374607164 | 0.7257096214344619 | 120s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedopt__noise-0.003__seed-42` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 126s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedyogi__noise-0.003__seed-42` | PASS | 0.9484375 | 0.9179117056124765 | 0.9243063016489146 | 0.9845371329934806 | 122s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__metric-privacy__fedavg__noise-0.003__seed-42` | PASS | 0.94375 | 0.9484572301106903 | 0.9515302379445822 | 0.9876959174434514 | 113s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__metric-privacy__fedavgm__noise-0.003__seed-42` | PASS | 0.7671875 | 0.5515422819391153 | 0.5514758012386096 | 0.9356642022988368 | 122s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__metric-privacy__fedmedian__noise-0.003__seed-42` | PASS | 0.9453125 | 0.9479497128076382 | 0.9488780581962071 | 0.9889578967711983 | 123s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__metric-privacy__fedopt__noise-0.003__seed-42` | FAIL |  |  |  |  | 74s | training-failed |
| `appendix-b__homogeneous__metric-privacy__fedprox__noise-0.003__seed-42` | PASS | 0.5671875 | 0.32160936902448917 | 0.5302565270338504 | 0.7558865339514844 | 122s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__metric-privacy__fedyogi__noise-0.003__seed-42` | PASS | 0.9609375 | 0.949231615288259 | 0.9425874586338262 | 0.9905923718764251 | 141s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedavg__noise-0.05__seed-42` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 136s | trained-evaluated-model-deleted |
| `appendix-b__homogeneous__global-dp__fedavg__noise-0.1__seed-42` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.49949262895792923 | 105s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43` | FAIL |  |  |  |  | 68s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42` | FAIL |  |  |  |  | 124s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44` | FAIL |  |  |  |  | 101s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-45` | FAIL |  |  |  |  | 88s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-46` | FAIL |  |  |  |  | 68s | training-failed |
| `appendix-b__homogeneous__metric-privacy__fedopt__noise-0.003__seed-42` | FAIL |  |  |  |  | 85s | training-failed |
| `main__non-iid__metric-privacy__fedopt__noise-0.01__seed-42` | PASS | 0.4953125 | 0.16562173458725182 | 0.123828125 | 0.5 | 150s | trained-evaluated-model-deleted |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43` | FAIL |  |  |  |  | 64s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42` | FAIL |  |  |  |  | 112s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44` | FAIL |  |  |  |  | 67s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-45` | FAIL |  |  |  |  | 61s | training-failed |
| `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-46` | FAIL |  |  |  |  | 58s | training-failed |
| `appendix-b__homogeneous__metric-privacy__fedopt__noise-0.003__seed-42` | FAIL |  |  |  |  | 55s | training-failed |

## Final status

- Expected: 122
- Complete and validated: 116
- Missing/failed: 6
- Retained final `.pt` files (failure recovery only): 0

### Missing or failed

- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-45` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-46` — FileNotFoundError(2, 'No such file or directory')
- `appendix-b__homogeneous__metric-privacy__fedopt__noise-0.003__seed-42` — FileNotFoundError(2, 'No such file or directory')

## Homogeneous five-seed server final-test summary

| Privacy | Aggregator | Accuracy mean ± SD | Macro F1 mean ± SD | Macro precision mean ± SD | Macro AUC mean ± SD | n |
|---|---|---:|---:|---:|---:|---:|
| global-dp | fedavg | 0.942188 ± 0.011951 | 0.935792 ± 0.021815 | 0.946706 ± 0.026395 | 0.986962 ± 0.002758 | 5 |
| global-dp | fedavgm | 0.777187 ± 0.027457 | 0.603442 ± 0.096615 | 0.621023 ± 0.133728 | 0.931849 ± 0.017410 | 5 |
| global-dp | fedmedian | 0.943750 ± 0.005741 | 0.939949 ± 0.014257 | 0.953803 ± 0.006312 | 0.985796 ± 0.001815 | 5 |
| global-dp | fedopt | 0.495312 ± 0.000000 | 0.165622 ± 0.000000 | 0.123828 ± 0.000000 | 0.496240 ± 0.008407 | 5 |
| global-dp | fedprox | 0.548750 ± 0.018033 | 0.288920 ± 0.019409 | 0.318055 ± 0.118557 | 0.737503 ± 0.010593 | 5 |
| global-dp | fedyogi | 0.947187 ± 0.012273 | 0.926867 ± 0.028021 | 0.948482 ± 0.018649 | 0.987126 ± 0.003426 | 5 |
| metric-privacy | fedavg | 0.945312 ± 0.009111 | 0.940734 ± 0.024607 | 0.957288 ± 0.011037 | 0.986952 ± 0.001591 | 5 |
| metric-privacy | fedavgm | 0.791875 ± 0.019900 | 0.660772 ± 0.124105 | 0.684211 ± 0.150984 | 0.940026 ± 0.015108 | 5 |
| metric-privacy | fedmedian | 0.944375 ± 0.007542 | 0.941541 ± 0.010210 | 0.946735 ± 0.020041 | 0.984960 ± 0.003333 | 5 |
| metric-privacy | fedprox | 0.495625 ± 0.000699 | 0.166089 ± 0.001045 | 0.173867 ± 0.111890 | 0.532383 ± 0.064050 | 5 |
| metric-privacy | fedyogi | 0.952187 ± 0.009733 | 0.952287 ± 0.013885 | 0.961341 ± 0.015413 | 0.987413 ± 0.002417 | 5 |
| vanilla | fedavg | 0.957812 ± 0.012451 | 0.939364 ± 0.040211 | 0.932570 ± 0.059256 | 0.988565 ± 0.003404 | 5 |
| vanilla | fedavgm | 0.919375 ± 0.016632 | 0.863413 ± 0.078431 | 0.936887 ± 0.014962 | 0.986090 ± 0.004358 | 5 |
| vanilla | fedmedian | 0.946875 ± 0.007574 | 0.944816 ± 0.011733 | 0.946881 ± 0.011556 | 0.987466 ± 0.003031 | 5 |
| vanilla | fedopt | 0.411250 ± 0.076812 | 0.156519 ± 0.024355 | 0.120230 ± 0.034741 | 0.505742 ± 0.013413 | 5 |
| vanilla | fedprox | 0.550312 ± 0.014036 | 0.289163 ± 0.009428 | 0.269401 ± 0.009248 | 0.723985 ± 0.007045 | 5 |
| vanilla | fedyogi | 0.950937 ± 0.016410 | 0.910671 ± 0.076452 | 0.907331 ± 0.079051 | 0.986577 ± 0.006177 | 5 |

## Non-IID fixed-seed server final-test results

| Privacy | Aggregator | Accuracy | Macro F1 | Macro precision | Macro AUC |
|---|---|---:|---:|---:|---:|
| global-dp | fedavg | 0.950000 | 0.955808 | 0.963400 | 0.989659 |
| global-dp | fedavgm | 0.746875 | 0.537459 | 0.546703 | 0.920700 |
| global-dp | fedmedian | 0.945312 | 0.932451 | 0.948965 | 0.989049 |
| global-dp | fedopt | 0.359375 | 0.132184 | 0.089844 | 0.524981 |
| global-dp | fedprox | 0.568750 | 0.350107 | 0.392492 | 0.755395 |
| global-dp | fedyogi | 0.943750 | 0.700681 | 0.699899 | 0.990448 |
| metric-privacy | fedavg | 0.962500 | 0.965236 | 0.966937 | 0.988630 |
| metric-privacy | fedavgm | 0.787500 | 0.568038 | 0.577491 | 0.939379 |
| metric-privacy | fedmedian | 0.950000 | 0.938199 | 0.958034 | 0.986440 |
| metric-privacy | fedopt | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| metric-privacy | fedprox | 0.495312 | 0.189747 | 0.333769 | 0.484323 |
| metric-privacy | fedyogi | 0.954688 | 0.924810 | 0.906442 | 0.988294 |
| vanilla | fedavg | 0.967187 | 0.968398 | 0.977126 | 0.991655 |
| vanilla | fedavgm | 0.937500 | 0.903448 | 0.943803 | 0.985056 |
| vanilla | fedmedian | 0.964063 | 0.964598 | 0.966157 | 0.992333 |
| vanilla | fedopt | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| vanilla | fedprox | 0.545312 | 0.285055 | 0.264380 | 0.730229 |
| vanilla | fedyogi | 0.950000 | 0.933339 | 0.968076 | 0.988279 |

## Appendix B server final-test results

| Noise | Privacy | Aggregator | Accuracy | Macro F1 | Macro precision | Macro AUC |
|---:|---|---|---:|---:|---:|---:|
| 0.003 | global-dp | fedavg | 0.950000 | 0.953348 | 0.959436 | 0.988504 |
| 0.003 | global-dp | fedavgm | 0.768750 | 0.550001 | 0.547629 | 0.935485 |
| 0.003 | global-dp | fedmedian | 0.940625 | 0.931691 | 0.922826 | 0.988767 |
| 0.003 | global-dp | fedopt | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| 0.003 | global-dp | fedprox | 0.553125 | 0.292432 | 0.272234 | 0.725710 |
| 0.003 | global-dp | fedyogi | 0.948438 | 0.917912 | 0.924306 | 0.984537 |
| 0.003 | metric-privacy | fedavg | 0.943750 | 0.948457 | 0.951530 | 0.987696 |
| 0.003 | metric-privacy | fedavgm | 0.767188 | 0.551542 | 0.551476 | 0.935664 |
| 0.003 | metric-privacy | fedmedian | 0.945312 | 0.947950 | 0.948878 | 0.988958 |
| 0.003 | metric-privacy | fedprox | 0.567187 | 0.321609 | 0.530257 | 0.755887 |
| 0.003 | metric-privacy | fedyogi | 0.960938 | 0.949232 | 0.942587 | 0.990592 |
| 0.05 | global-dp | fedavg | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| 0.1 | global-dp | fedavg | 0.495312 | 0.165622 | 0.123828 | 0.499493 |

## Repeated failure analysis

Six requested Metric-Privacy + FedOpt configurations were attempted in all three passes and failed deterministically at round 4:

- Homogeneous main seeds 42, 43, 44, 45, and 46 at noise 0.01.
- Homogeneous Appendix B seed 42 at noise 0.003.

For each, the maximum pairwise client-model distance decreased to exactly zero after round 3. The implemented calibration requires `noise_multiplier / distance`, so `MetricPrivacyServerSideFixedClipping` intentionally raises rather than divide by zero or invent an undocumented epsilon/cap. These failures are retained in the manifest and per-run logs; no metric values were fabricated. The other 116 configurations completed with validated detailed evaluations and raw predictions.

## Final status

- Expected: 122
- Complete and validated: 116
- Missing/failed: 6
- Retained final `.pt` files (failure recovery only): 0

### Missing or failed

- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-42` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-43` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-44` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-45` — FileNotFoundError(2, 'No such file or directory')
- `main__homogeneous__metric-privacy__fedopt__noise-0.01__seed-46` — FileNotFoundError(2, 'No such file or directory')
- `appendix-b__homogeneous__metric-privacy__fedopt__noise-0.003__seed-42` — FileNotFoundError(2, 'No such file or directory')

## Homogeneous five-seed server final-test summary

| Privacy | Aggregator | Accuracy mean ± SD | Macro F1 mean ± SD | Macro precision mean ± SD | Macro AUC mean ± SD | n |
|---|---|---:|---:|---:|---:|---:|
| global-dp | fedavg | 0.942188 ± 0.011951 | 0.935792 ± 0.021815 | 0.946706 ± 0.026395 | 0.986962 ± 0.002758 | 5 |
| global-dp | fedavgm | 0.777187 ± 0.027457 | 0.603442 ± 0.096615 | 0.621023 ± 0.133728 | 0.931849 ± 0.017410 | 5 |
| global-dp | fedmedian | 0.943750 ± 0.005741 | 0.939949 ± 0.014257 | 0.953803 ± 0.006312 | 0.985796 ± 0.001815 | 5 |
| global-dp | fedopt | 0.495312 ± 0.000000 | 0.165622 ± 0.000000 | 0.123828 ± 0.000000 | 0.496240 ± 0.008407 | 5 |
| global-dp | fedprox | 0.548750 ± 0.018033 | 0.288920 ± 0.019409 | 0.318055 ± 0.118557 | 0.737503 ± 0.010593 | 5 |
| global-dp | fedyogi | 0.947187 ± 0.012273 | 0.926867 ± 0.028021 | 0.948482 ± 0.018649 | 0.987126 ± 0.003426 | 5 |
| metric-privacy | fedavg | 0.945312 ± 0.009111 | 0.940734 ± 0.024607 | 0.957288 ± 0.011037 | 0.986952 ± 0.001591 | 5 |
| metric-privacy | fedavgm | 0.791875 ± 0.019900 | 0.660772 ± 0.124105 | 0.684211 ± 0.150984 | 0.940026 ± 0.015108 | 5 |
| metric-privacy | fedmedian | 0.944375 ± 0.007542 | 0.941541 ± 0.010210 | 0.946735 ± 0.020041 | 0.984960 ± 0.003333 | 5 |
| metric-privacy | fedprox | 0.495625 ± 0.000699 | 0.166089 ± 0.001045 | 0.173867 ± 0.111890 | 0.532383 ± 0.064050 | 5 |
| metric-privacy | fedyogi | 0.952187 ± 0.009733 | 0.952287 ± 0.013885 | 0.961341 ± 0.015413 | 0.987413 ± 0.002417 | 5 |
| vanilla | fedavg | 0.957812 ± 0.012451 | 0.939364 ± 0.040211 | 0.932570 ± 0.059256 | 0.988565 ± 0.003404 | 5 |
| vanilla | fedavgm | 0.919375 ± 0.016632 | 0.863413 ± 0.078431 | 0.936887 ± 0.014962 | 0.986090 ± 0.004358 | 5 |
| vanilla | fedmedian | 0.946875 ± 0.007574 | 0.944816 ± 0.011733 | 0.946881 ± 0.011556 | 0.987466 ± 0.003031 | 5 |
| vanilla | fedopt | 0.411250 ± 0.076812 | 0.156519 ± 0.024355 | 0.120230 ± 0.034741 | 0.505742 ± 0.013413 | 5 |
| vanilla | fedprox | 0.550312 ± 0.014036 | 0.289163 ± 0.009428 | 0.269401 ± 0.009248 | 0.723985 ± 0.007045 | 5 |
| vanilla | fedyogi | 0.950937 ± 0.016410 | 0.910671 ± 0.076452 | 0.907331 ± 0.079051 | 0.986577 ± 0.006177 | 5 |

## Non-IID fixed-seed server final-test results

| Privacy | Aggregator | Accuracy | Macro F1 | Macro precision | Macro AUC |
|---|---|---:|---:|---:|---:|
| global-dp | fedavg | 0.950000 | 0.955808 | 0.963400 | 0.989659 |
| global-dp | fedavgm | 0.746875 | 0.537459 | 0.546703 | 0.920700 |
| global-dp | fedmedian | 0.945312 | 0.932451 | 0.948965 | 0.989049 |
| global-dp | fedopt | 0.359375 | 0.132184 | 0.089844 | 0.524981 |
| global-dp | fedprox | 0.568750 | 0.350107 | 0.392492 | 0.755395 |
| global-dp | fedyogi | 0.943750 | 0.700681 | 0.699899 | 0.990448 |
| metric-privacy | fedavg | 0.962500 | 0.965236 | 0.966937 | 0.988630 |
| metric-privacy | fedavgm | 0.787500 | 0.568038 | 0.577491 | 0.939379 |
| metric-privacy | fedmedian | 0.950000 | 0.938199 | 0.958034 | 0.986440 |
| metric-privacy | fedopt | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| metric-privacy | fedprox | 0.495312 | 0.189747 | 0.333769 | 0.484323 |
| metric-privacy | fedyogi | 0.954688 | 0.924810 | 0.906442 | 0.988294 |
| vanilla | fedavg | 0.967187 | 0.968398 | 0.977126 | 0.991655 |
| vanilla | fedavgm | 0.937500 | 0.903448 | 0.943803 | 0.985056 |
| vanilla | fedmedian | 0.964063 | 0.964598 | 0.966157 | 0.992333 |
| vanilla | fedopt | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| vanilla | fedprox | 0.545312 | 0.285055 | 0.264380 | 0.730229 |
| vanilla | fedyogi | 0.950000 | 0.933339 | 0.968076 | 0.988279 |

## Appendix B server final-test results

| Noise | Privacy | Aggregator | Accuracy | Macro F1 | Macro precision | Macro AUC |
|---:|---|---|---:|---:|---:|---:|
| 0.003 | global-dp | fedavg | 0.950000 | 0.953348 | 0.959436 | 0.988504 |
| 0.003 | global-dp | fedavgm | 0.768750 | 0.550001 | 0.547629 | 0.935485 |
| 0.003 | global-dp | fedmedian | 0.940625 | 0.931691 | 0.922826 | 0.988767 |
| 0.003 | global-dp | fedopt | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| 0.003 | global-dp | fedprox | 0.553125 | 0.292432 | 0.272234 | 0.725710 |
| 0.003 | global-dp | fedyogi | 0.948438 | 0.917912 | 0.924306 | 0.984537 |
| 0.003 | metric-privacy | fedavg | 0.943750 | 0.948457 | 0.951530 | 0.987696 |
| 0.003 | metric-privacy | fedavgm | 0.767188 | 0.551542 | 0.551476 | 0.935664 |
| 0.003 | metric-privacy | fedmedian | 0.945312 | 0.947950 | 0.948878 | 0.988958 |
| 0.003 | metric-privacy | fedprox | 0.567187 | 0.321609 | 0.530257 | 0.755887 |
| 0.003 | metric-privacy | fedyogi | 0.960938 | 0.949232 | 0.942587 | 0.990592 |
| 0.05 | global-dp | fedavg | 0.495312 | 0.165622 | 0.123828 | 0.500000 |
| 0.1 | global-dp | fedavg | 0.495312 | 0.165622 | 0.123828 | 0.499493 |

## Validation notes

- Unit tests after restoring the committed DataLoader worker setting: **27 passed, 5 deselected**.
- H100 CUDA health check passed after the Studio restart.
- Execution: 2026-07-22 16:30:55–18:48:15 UTC using two concurrent experiment lanes.
- Final artifact validation: **116/122 complete**; the six missing configurations are exactly the repeatedly failing Metric-Privacy + FedOpt cases documented above.
- Successful transient model files were deleted only after evaluation JSON/NPZ validation; zero final  files remain.
- Correction: zero final model checkpoint files remain.
