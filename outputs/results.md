# Results

Synthetic corpus: 500 records, 40 requests, 8 exemption categories. TEST records use phrasings that the rules and classifier never saw.

## Retrieval

| Metric | Value |
|---|---|
| Recall (responsive records returned) | 99% |
| Precision of returned set | 76% |
| Precision@10 | 99% |
| Mean average precision | 0.99 |
| Average records returned per request | 13.7 |
| Request parsing: dates / agency / doc type | 100% / 100% / 100% |

v1 baseline: recall 78%, precision 22%.

## Sensitivity flagging (threshold 0.5, high-harm floor 0.25)

| | Dev phrasings | Held-out phrasings |
|---|---|---|
| Sensitive sentences caught | 184/184 (100%) | 63/73 (86%) |
| Sensitive records with at least one flag | 100% | 90% |
| Flags sent to officer | 242 | 95 |
| False alarms | 54 | 20 |
| Right category when caught | 100% | 70% |

Held-out recall by category: Cabinet or Executive Council material 100%, Security or law enforcement 73%, Legal advice (privilege) 55%, Deliberative material 100%, Aboriginal sacred or restricted information 100%, Health information about a person 80%, Personal information 100%, Commercial or business information 100%.

**Reading this honestly.** The detector is near perfect on wording it was built from and weaker on new wording. That gap is the case for the design: the system narrows and prioritises, the officer reads every record, and finalise() is blocked until each one has an officer outcome.

## Detector improvement: before and after

Before: rules + classifier. After: rules + classifier + names + word vectors (NT records); rules + classifier + names (email). Both changes were chosen by leave-phrasing-out cross-validation on dev phrasings only (experiments/detector_cv.py), then the held-out set was scored once.

| Threshold 0.5 | Before | After |
|---|---|---|
| NT held-out phrasings caught | 26/73 | 63/73 |
| NT held-out false alarms | 22 | 20 |
| NT held-out flags sent to officer | 48 | 95 |
| NT dev false alarms | 51 | 54 |
| Real emails held-out (core rules): caught / false alarms | 5/22 · 2/18 | 9/22 · 7/18 |
| Real emails held-out (core + correspondence pack): caught / false alarms | 13/22 · 8/18 | 14/22 · 11/18 |

Cross-validation on dev (rules off, unseen phrasings): classifier (current) 8/47; + word vectors x1 8/47; + word vectors x4 24/47; + word vectors x8 26/47; + names 0.3 8/47; + names 0.3 + vectors x4 28/47; + names 0.3 + vectors x8 27/47; + names 0.4 10/47; + names 0.4 + vectors x4 28/47; + names 0.4 + vectors x8 29/47; + names 0.5 18/47; + names 0.5 + vectors x4 30/47; + names 0.5 + vectors x8 31/47. Chosen: + names 0.4 + vectors x8.

Word vectors are off for email: on the 40 dev emails they flagged every message, a domain-shift effect.

## Error costs (held-out phrasings)

Expected cost = 20 x missed + 1 x flags reviewed. Lowest-cost threshold with the high-harm floor on: **0.15** (missed 0, flags 196, false alarms 88).

| Threshold | Floor | Caught | Missed | Flags | False alarms | Cost |
|---|---|---|---|---|---|---|
| 0.05 | on | 73 | 0 | 234 | 109 | 234 |
| 0.1 | on | 73 | 0 | 211 | 91 | 211 |
| 0.2 | on | 71 | 2 | 184 | 88 | 224 |
| 0.3 | on | 71 | 2 | 179 | 88 | 219 |
| 0.5 | on | 63 | 10 | 95 | 20 | 295 |
| 0.7 | on | 60 | 13 | 73 | 4 | 333 |
| 0.9 | on | 43 | 30 | 56 | 4 | 656 |
| 0.05 | off | 73 | 0 | 234 | 109 | 234 |
| 0.1 | off | 73 | 0 | 211 | 91 | 211 |
| 0.2 | off | 71 | 2 | 184 | 88 | 224 |
| 0.3 | off | 70 | 3 | 175 | 88 | 235 |
| 0.5 | off | 51 | 22 | 76 | 18 | 516 |
| 0.7 | off | 41 | 32 | 44 | 1 | 684 |
| 0.9 | off | 11 | 62 | 14 | 1 | 1254 |

## Real emails (80 hand-labelled Enron messages)

Labels: one annotator (Claude), record level: does this email contain material an FOI officer would need to review? The correspondence rule pack was written from the first 40 only.

| Split | Rule packs | Threshold | Sensitive caught | False alarms | Flags per email |
|---|---|---|---|---|---|
| dev | core only | 0.2 | 18/19 | 21/21 | 4.2 |
| dev | core only | 0.3 | 17/19 | 19/21 | 3.33 |
| dev | core only | 0.5 | 13/19 | 7/21 | 0.8 |
| dev | core only | 0.7 | 6/19 | 4/21 | 0.3 |
| dev | correspondence | 0.2 | 19/19 | 21/21 | 4.88 |
| dev | correspondence | 0.3 | 19/19 | 19/21 | 4.05 |
| dev | correspondence | 0.5 | 18/19 | 10/21 | 1.7 |
| dev | correspondence | 0.7 | 9/19 | 4/21 | 0.42 |
| heldout | core only | 0.2 | 21/22 | 17/18 | 5.05 |
| heldout | core only | 0.3 | 21/22 | 13/18 | 4.1 |
| heldout | core only | 0.5 | 9/22 | 7/18 | 0.68 |
| heldout | core only | 0.7 | 3/22 | 2/18 | 0.15 |
| heldout | correspondence | 0.2 | 21/22 | 18/18 | 5.38 |
| heldout | correspondence | 0.3 | 21/22 | 15/18 | 4.47 |
| heldout | correspondence | 0.5 | 14/22 | 11/18 | 1.4 |
| heldout | correspondence | 0.7 | 6/22 | 4/18 | 0.35 |

## What the held-out set missed (default setting)

- s46_law_enforcement: "The lock combination for the evidence room is recorded on page two."
- s49d_legal_privilege: "Advice received from external solicitors on the lease dispute is summarised for the executive."
- s49d_legal_privilege: "Our in-house legal team considers the contractor would win if the matter went to court."
- s56_health: "Jeffrey Davis has been attending the drug and alcohol counselling service weekly."
- s56_health: "David Soto has been attending the drug and alcohol counselling service weekly."
