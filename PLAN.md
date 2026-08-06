# Runs

- For multi round CIA, use 10% split and perturb with noise whose stdev is 20% of the maximum pixel value.
- Noise ratio = noise multiplier / number of clients. Defined only for IN/OUT-replace settings.
- In original dataset experiments the exact paper 3 client data distribution is being used. Re-use that same distribution and record count in 3 client new dataset experiments. 
- New dataset = CIFAR10 limited to 4 classes

Assertions:
- The target shadow split and its noisy version must be identical between paired IN-remove and OUT-remove runs of the same seed, and reused across Vanilla, Global-DP, and Metric-DP.

## Original dataset reproduction/contesting
Alzheimer dataset
4 clients
FedAvg
Homogeneous
3 seeds
Vanilla, GDP, Metric
Noise multiplier 0.01

## Original dataset 3 client CIA IN-remove
Alzheimer dataset
3 clients
Non-iid distribution specifically from paper
FedAvg
IN-remove
Evaluate for CIA at all rounds (Multi round CIA)
3 seeds
Vanilla, GDP, Metric
Noise multiplier 0.01

## Original dataset 3 client CIA OUT-remove
Alzheimer dataset
3 clients
Non-iid distribution specifically from paper
FedAvg
OUT-remove
Evaluate for CIA at all rounds (Multi round CIA)
3 seeds
Vanilla, GDP, Metric
Noise multiplier 0.01

## New dataset 3 client CIA IN-remove (transfer experiment)
3 clients
Non-iid 
FedAvg
IN-remove
Evaluate for CIA at all rounds (Multi round CIA)
3 seeds
Vanilla, Global, Metric
Noise multiplier 0.01

## New dataset 3 client CIA OUT-remove (transfer experiment)
3 clients
Non-iid 
FedAvg
OUT-remove
Evaluate for CIA at all rounds (Multi round CIA)
3 seeds
Vanilla, Global, Metric
Noise multiplier 0.01

# Runs held off 
These runs are not ran because of decisions not taken yet. Runs above should be safe, and should be ran now.

Blocker 1: we changed datasets while scaling up the clients because the original dataset is too small and it affected results. Now the new dataset (fashion-mnist, filtered to 4 classes) has more data but, now we're in a bit of a tradeoff. Keeping the record count per client the same between 3, 8, 16, 48 clients mean more client means more data to train on -> accuracy might get too high. If we don't keep it the same -> how do you decide on how many records per client per client number? 

Blocker 2: we need to first do a noise ratios sweep before going with these results. We would ideally pick one of the noise ratios to be from the 3 client CIA test. The rest are not yet decided. 0.0125 will likely make most models collapse, and the replacement should be decided with a noise sweep on 3 and 48 clients.

## New dataset accuracy & CIA IN-replace
3, 8, 16, 48 clients
Non-iid 
FedAvg
IN-replace
Evaluate for CIA at all rounds (Multi round CIA)
3 seeds
Vanilla, Global, Metric
Noise ratios for non-vanilla: 0.0025, 0.00625, 0.0125

## New dataset CIA OUT-replace
3, 8, 16, 48 clients
Non-iid
FedAvg
OUT-replace
Evaluate for CIA at all rounds (Multi round CIA)
3 seeds
Vanilla, Global, Metric
Noise ratios for non-vanilla: 0.0025, 0.00625, 0.0125

===

# Update: unblocking above blockers

## New dataset 3 client CIA OUT-remove max records
UPDATE for blocker 1: Here's the solution. Use all of the records instead of following the paper's distribution. That should make around 400 records at 48 clients which should be enough. Here's the transfer experiment.

3 clients
Non-iid 
FedAvg
OUT-remove
Evaluate for CIA at all rounds (Multi round CIA)
3 seeds
Vanilla, Global, Metric
Noise multiplier 0.01
Use all of the records instead of following the paper's distribution

## Noise sweep new dataset
UPDATE for blocker 2: do this noise ratio sweep, and use the resulting ratios on the runs held off instead.

### Part 1: New dataset noise ratio sweep
New dataset
3 clients
Non-iid
FedAvg
IN-replace
Accuracy only
1 calibration seed
Vanilla once
Global, Metric
Noise ratios: 0.0025, 0.003333, 0.00625

If 0.00625 is destructive, additionally test:
Noise ratio: 0.005
Global, Metric

### Part 2: New dataset 48 client noise validation
New dataset
48 clients
Non-iid
FedAvg
IN-replace
Accuracy only
1 calibration seed
Vanilla, Global, Metric