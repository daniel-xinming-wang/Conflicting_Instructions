# ConInstruct low-rank subspace analysis

Method adapted from Sharma et al., *A Low-Rank Subspace Analysis of LLM Interventions* (arXiv:2606.14388). Each ConInstruct conflict type is treated as a behavior category. For each matched sample, `delta = h_new - h_org`; centered deltas are stacked and a rank-2 PCA/SVD subspace is fitted.

The paper identifies and removes decision-aligned PCs using a continuous model log-probability margin. That margin is unavailable here. The `label_residualized` analysis instead ranks pooled PCs by correlation with the binary final org/new judge label. It is an approximation and must not be described as an exact replication of decision residualization.

## Mean pairwise conflict-type overlap

| model_id        |   layer | space              |   overlap |
|:----------------|--------:|:-------------------|----------:|
| Qwen/Qwen3.5-4B |       0 | label_residualized |   0.27245 |
| Qwen/Qwen3.5-4B |       0 | raw                |   0.24902 |
| Qwen/Qwen3.5-4B |       8 | label_residualized |   0.28414 |
| Qwen/Qwen3.5-4B |       8 | raw                |   0.23600 |
| Qwen/Qwen3.5-4B |      16 | label_residualized |   0.09140 |
| Qwen/Qwen3.5-4B |      16 | raw                |   0.08047 |
| Qwen/Qwen3.5-4B |      24 | label_residualized |   0.06746 |
| Qwen/Qwen3.5-4B |      24 | raw                |   0.12812 |
| Qwen/Qwen3.5-4B |      31 | label_residualized |   0.04416 |
| Qwen/Qwen3.5-4B |      31 | raw                |   0.06683 |
| Qwen/Qwen3.5-9B |       0 | label_residualized |   0.53674 |
| Qwen/Qwen3.5-9B |       0 | raw                |   0.50528 |
| Qwen/Qwen3.5-9B |       8 | label_residualized |   0.14635 |
| Qwen/Qwen3.5-9B |       8 | raw                |   0.12574 |
| Qwen/Qwen3.5-9B |      16 | label_residualized |   0.08820 |
| Qwen/Qwen3.5-9B |      16 | raw                |   0.12049 |
| Qwen/Qwen3.5-9B |      24 | label_residualized |   0.06947 |
| Qwen/Qwen3.5-9B |      24 | raw                |   0.07388 |
| Qwen/Qwen3.5-9B |      31 | label_residualized |   0.05908 |
| Qwen/Qwen3.5-9B |      31 | raw                |   0.06232 |

## Rank-2 explained variance and label coupling

| model_id        |   layer |   raw_rank2_explained |   residual_rank2_explained |   decision_coupling_overlap |   decision_coupling_angle_deg |
|:----------------|--------:|----------------------:|---------------------------:|----------------------------:|------------------------------:|
| Qwen/Qwen3.5-4B |       0 |               0.31583 |                    0.33112 |                     0.06221 |                      78.19186 |
| Qwen/Qwen3.5-4B |       8 |               0.32158 |                    0.31433 |                     0.08089 |                      77.02102 |
| Qwen/Qwen3.5-4B |      16 |               0.32742 |                    0.29626 |                     0.13848 |                      72.38028 |
| Qwen/Qwen3.5-4B |      24 |               0.33024 |                    0.27291 |                     0.22692 |                      65.72159 |
| Qwen/Qwen3.5-4B |      31 |               0.28608 |                    0.25518 |                     0.16874 |                      71.13407 |
| Qwen/Qwen3.5-9B |       0 |               0.37860 |                    0.37927 |                     0.26391 |                      63.87817 |
| Qwen/Qwen3.5-9B |       8 |               0.28315 |                    0.25713 |                     0.13047 |                      73.13994 |
| Qwen/Qwen3.5-9B |      16 |               0.33676 |                    0.30592 |                     0.24376 |                      64.31576 |
| Qwen/Qwen3.5-9B |      24 |               0.33103 |                    0.28261 |                     0.24034 |                      64.78365 |
| Qwen/Qwen3.5-9B |      31 |               0.27485 |                    0.24731 |                     0.14496 |                      72.44229 |

## Bootstrap stability at layer 24

| model_id        | space              |    mean |     std |     min |     max |
|:----------------|:-------------------|--------:|--------:|--------:|--------:|
| Qwen/Qwen3.5-4B | label_residualized | 0.62356 | 0.21245 | 0.04048 | 0.96895 |
| Qwen/Qwen3.5-4B | raw                | 0.70080 | 0.19385 | 0.13125 | 0.98833 |
| Qwen/Qwen3.5-9B | label_residualized | 0.62801 | 0.20198 | 0.01106 | 0.97429 |
| Qwen/Qwen3.5-9B | raw                | 0.69680 | 0.19098 | 0.01101 | 0.95900 |

## Random rank-2 baseline

| model_id        |   hidden_dimension |   rank |   empirical_random_overlap_mean |   empirical_random_overlap_std |   theoretical_k_over_d |
|:----------------|-------------------:|-------:|--------------------------------:|-------------------------------:|-----------------------:|
| Qwen/Qwen3.5-4B |               2560 |      2 |                       0.0007763 |                      0.0005280 |              0.0007813 |
| Qwen/Qwen3.5-9B |               4096 |      2 |                       0.0004827 |                      0.0003481 |              0.0004883 |

## Cross-model correlation of overlap matrices

|   layer | space              |   spearman_r |   p_value |   n_pairs |
|--------:|:-------------------|-------------:|----------:|----------:|
|       0 | label_residualized |      0.37194 |   0.02550 |        36 |
|       0 | raw                |      0.31042 |   0.06538 |        36 |
|       8 | label_residualized |      0.27207 |   0.10843 |        36 |
|       8 | raw                |      0.61750 |   0.00006 |        36 |
|      16 | label_residualized |      0.18996 |   0.26713 |        36 |
|      16 | raw                |      0.20849 |   0.22236 |        36 |
|      24 | label_residualized |      0.25199 |   0.13817 |        36 |
|      24 | raw                |      0.61364 |   0.00007 |        36 |
|      31 | label_residualized |      0.61519 |   0.00007 |        36 |
|      31 | raw                |      0.79820 |   0.00000 |        36 |

## Exploratory relation between overlap and outcome-profile similarity

| model_id        |   layer | space              | outcome_similarity   |   spearman_r |   p_value |   n_pairs |
|:----------------|--------:|:-------------------|:---------------------|-------------:|----------:|----------:|
| Qwen/Qwen3.5-4B |       0 | label_residualized | new_choice_rate      |     -0.12587 |   0.46449 |        36 |
| Qwen/Qwen3.5-4B |       0 | label_residualized | self_resolves_rate   |     -0.18400 |   0.28272 |        36 |
| Qwen/Qwen3.5-4B |       0 | raw                | new_choice_rate      |     -0.10862 |   0.52830 |        36 |
| Qwen/Qwen3.5-4B |       0 | raw                | self_resolves_rate   |     -0.08762 |   0.61136 |        36 |
| Qwen/Qwen3.5-4B |       8 | label_residualized | new_choice_rate      |     -0.19846 |   0.24593 |        36 |
| Qwen/Qwen3.5-4B |       8 | label_residualized | self_resolves_rate   |     -0.18322 |   0.28478 |        36 |
| Qwen/Qwen3.5-4B |       8 | raw                | new_choice_rate      |     -0.26049 |   0.12493 |        36 |
| Qwen/Qwen3.5-4B |       8 | raw                | self_resolves_rate   |     -0.18000 |   0.29348 |        36 |
| Qwen/Qwen3.5-4B |      16 | label_residualized | new_choice_rate      |     -0.00412 |   0.98098 |        36 |
| Qwen/Qwen3.5-4B |      16 | label_residualized | self_resolves_rate   |     -0.01714 |   0.92098 |        36 |
| Qwen/Qwen3.5-4B |      16 | raw                | new_choice_rate      |     -0.09653 |   0.57546 |        36 |
| Qwen/Qwen3.5-4B |      16 | raw                | self_resolves_rate   |     -0.34957 |   0.03663 |        36 |
| Qwen/Qwen3.5-4B |      24 | label_residualized | new_choice_rate      |     -0.08829 |   0.60863 |        36 |
| Qwen/Qwen3.5-4B |      24 | label_residualized | self_resolves_rate   |     -0.14109 |   0.41177 |        36 |
| Qwen/Qwen3.5-4B |      24 | raw                | new_choice_rate      |     -0.35933 |   0.03137 |        36 |
| Qwen/Qwen3.5-4B |      24 | raw                | self_resolves_rate   |     -0.41554 |   0.01172 |        36 |
| Qwen/Qwen3.5-4B |      31 | label_residualized | new_choice_rate      |     -0.31763 |   0.05906 |        36 |
| Qwen/Qwen3.5-4B |      31 | label_residualized | self_resolves_rate   |     -0.23579 |   0.16622 |        36 |
| Qwen/Qwen3.5-4B |      31 | raw                | new_choice_rate      |     -0.32716 |   0.05146 |        36 |
| Qwen/Qwen3.5-4B |      31 | raw                | self_resolves_rate   |     -0.35704 |   0.03254 |        36 |
| Qwen/Qwen3.5-9B |       0 | label_residualized | new_choice_rate      |     -0.24659 |   0.14711 |        36 |
| Qwen/Qwen3.5-9B |       0 | label_residualized | self_resolves_rate   |     -0.32091 |   0.05635 |        36 |
| Qwen/Qwen3.5-9B |       0 | raw                | new_choice_rate      |      0.02317 |   0.89332 |        36 |
| Qwen/Qwen3.5-9B |       0 | raw                | self_resolves_rate   |     -0.10753 |   0.53249 |        36 |
| Qwen/Qwen3.5-9B |       8 | label_residualized | new_choice_rate      |      0.18121 |   0.29020 |        36 |
| Qwen/Qwen3.5-9B |       8 | label_residualized | self_resolves_rate   |      0.10598 |   0.53843 |        36 |
| Qwen/Qwen3.5-9B |       8 | raw                | new_choice_rate      |      0.10991 |   0.52339 |        36 |
| Qwen/Qwen3.5-9B |       8 | raw                | self_resolves_rate   |     -0.08538 |   0.62053 |        36 |
| Qwen/Qwen3.5-9B |      16 | label_residualized | new_choice_rate      |      0.24839 |   0.14408 |        36 |
| Qwen/Qwen3.5-9B |      16 | label_residualized | self_resolves_rate   |      0.13380 |   0.43659 |        36 |
| Qwen/Qwen3.5-9B |      16 | raw                | new_choice_rate      |      0.13900 |   0.41881 |        36 |
| Qwen/Qwen3.5-9B |      16 | raw                | self_resolves_rate   |      0.25562 |   0.13240 |        36 |
| Qwen/Qwen3.5-9B |      24 | label_residualized | new_choice_rate      |     -0.28366 |   0.09364 |        36 |
| Qwen/Qwen3.5-9B |      24 | label_residualized | self_resolves_rate   |      0.02138 |   0.90152 |        36 |
| Qwen/Qwen3.5-9B |      24 | raw                | new_choice_rate      |     -0.14826 |   0.38816 |        36 |
| Qwen/Qwen3.5-9B |      24 | raw                | self_resolves_rate   |     -0.13611 |   0.42862 |        36 |
| Qwen/Qwen3.5-9B |      31 | label_residualized | new_choice_rate      |     -0.34080 |   0.04196 |        36 |
| Qwen/Qwen3.5-9B |      31 | label_residualized | self_resolves_rate   |     -0.17179 |   0.31644 |        36 |
| Qwen/Qwen3.5-9B |      31 | raw                | new_choice_rate      |     -0.30090 |   0.07454 |        36 |
| Qwen/Qwen3.5-9B |      31 | raw                | self_resolves_rate   |     -0.20733 |   0.22502 |        36 |

## Do taxonomically related conflict types share more geometry?

Type components are defined directly from the ConInstruct taxonomy: content, keyword, phrase, length, format, and style. Examples include 1↔8 (content), 3↔4 (phrase), and 7↔9 (style).

| model_id        |   layer | space              |   n_shared_pairs |   shared_mean_overlap |   nonshared_mean_overlap |   shared_median_overlap |   nonshared_median_overlap |   mannwhitney_u |   p_value |
|:----------------|--------:|:-------------------|-----------------:|----------------------:|-------------------------:|------------------------:|---------------------------:|----------------:|----------:|
| Qwen/Qwen3.5-4B |       0 | label_residualized |                9 |               0.38965 |                  0.23339 |                 0.29372 |                    0.21168 |       189.00000 |   0.00719 |
| Qwen/Qwen3.5-4B |       0 | raw                |                9 |               0.36218 |                  0.21129 |                 0.25564 |                    0.19703 |       180.00000 |   0.01705 |
| Qwen/Qwen3.5-4B |       8 | label_residualized |                9 |               0.41030 |                  0.24208 |                 0.39586 |                    0.24075 |       202.00000 |   0.00174 |
| Qwen/Qwen3.5-4B |       8 | raw                |                9 |               0.39154 |                  0.18415 |                 0.37009 |                    0.13136 |       208.00000 |   0.00084 |
| Qwen/Qwen3.5-4B |      16 | label_residualized |                9 |               0.19351 |                  0.05737 |                 0.09454 |                    0.04493 |       167.00000 |   0.05009 |
| Qwen/Qwen3.5-4B |      16 | raw                |                9 |               0.21883 |                  0.03435 |                 0.17479 |                    0.02693 |       219.00000 |   0.00020 |
| Qwen/Qwen3.5-4B |      24 | label_residualized |                9 |               0.14136 |                  0.04283 |                 0.10040 |                    0.02195 |       167.00000 |   0.05009 |
| Qwen/Qwen3.5-4B |      24 | raw                |                9 |               0.25684 |                  0.08522 |                 0.27637 |                    0.09150 |       211.00000 |   0.00057 |
| Qwen/Qwen3.5-4B |      31 | label_residualized |                9 |               0.11474 |                  0.02063 |                 0.04278 |                    0.01420 |       178.00000 |   0.02039 |
| Qwen/Qwen3.5-4B |      31 | raw                |                9 |               0.17601 |                  0.03044 |                 0.13787 |                    0.02878 |       214.00000 |   0.00039 |
| Qwen/Qwen3.5-9B |       0 | label_residualized |                9 |               0.58337 |                  0.52119 |                 0.52758 |                    0.52201 |       140.00000 |   0.25540 |
| Qwen/Qwen3.5-9B |       0 | raw                |                9 |               0.58659 |                  0.47818 |                 0.61221 |                    0.45417 |       161.00000 |   0.07711 |
| Qwen/Qwen3.5-9B |       8 | label_residualized |                9 |               0.22634 |                  0.11969 |                 0.15492 |                    0.11244 |       156.00000 |   0.10710 |
| Qwen/Qwen3.5-9B |       8 | raw                |                9 |               0.23053 |                  0.09081 |                 0.13657 |                    0.06095 |       180.00000 |   0.01705 |
| Qwen/Qwen3.5-9B |      16 | label_residualized |                9 |               0.20983 |                  0.04766 |                 0.25699 |                    0.03304 |       168.00000 |   0.04643 |
| Qwen/Qwen3.5-9B |      16 | raw                |                9 |               0.23594 |                  0.08201 |                 0.27970 |                    0.05746 |       172.00000 |   0.03388 |
| Qwen/Qwen3.5-9B |      24 | label_residualized |                9 |               0.16641 |                  0.03715 |                 0.11610 |                    0.02067 |       197.00000 |   0.00307 |
| Qwen/Qwen3.5-9B |      24 | raw                |                9 |               0.20043 |                  0.03169 |                 0.16740 |                    0.01549 |       178.00000 |   0.02039 |
| Qwen/Qwen3.5-9B |      31 | label_residualized |                9 |               0.14614 |                  0.03006 |                 0.02057 |                    0.01214 |       166.00000 |   0.05398 |
| Qwen/Qwen3.5-9B |      31 | raw                |                9 |               0.19368 |                  0.01853 |                 0.13733 |                    0.01245 |       217.00000 |   0.00026 |

## Rank sensitivity at layer 24

| model_id        |   layer |   rank |   mean_overlap |   median_overlap |   shared_component_mean_overlap |   nonshared_mean_overlap |   shared_vs_nonshared_u |   shared_vs_nonshared_p_value |   random_k_over_d |
|:----------------|--------:|-------:|---------------:|-----------------:|--------------------------------:|-------------------------:|------------------------:|------------------------------:|------------------:|
| Qwen/Qwen3.5-4B |      24 |      1 |        0.08417 |          0.01070 |                         0.27323 |                  0.02114 |               163.00000 |                       0.06709 |           0.00039 |
| Qwen/Qwen3.5-4B |      24 |      2 |        0.12812 |          0.11037 |                         0.25684 |                  0.08521 |               211.00000 |                       0.00057 |           0.00078 |
| Qwen/Qwen3.5-4B |      24 |      3 |        0.12840 |          0.11134 |                         0.22076 |                  0.09761 |               199.00000 |                       0.00245 |           0.00117 |
| Qwen/Qwen3.5-4B |      24 |      4 |        0.12114 |          0.10404 |                         0.20116 |                  0.09447 |               200.00000 |                       0.00219 |           0.00156 |
| Qwen/Qwen3.5-4B |      24 |      5 |        0.12676 |          0.10818 |                         0.20452 |                  0.10084 |               207.00000 |                       0.00095 |           0.00195 |
| Qwen/Qwen3.5-9B |      24 |      1 |        0.06844 |          0.00454 |                         0.24907 |                  0.00823 |               163.00000 |                       0.06709 |           0.00024 |
| Qwen/Qwen3.5-9B |      24 |      2 |        0.07388 |          0.01914 |                         0.20043 |                  0.03169 |               178.00000 |                       0.02039 |           0.00049 |
| Qwen/Qwen3.5-9B |      24 |      3 |        0.07969 |          0.03998 |                         0.19394 |                  0.04161 |               180.00000 |                       0.01705 |           0.00073 |
| Qwen/Qwen3.5-9B |      24 |      4 |        0.08861 |          0.05430 |                         0.17702 |                  0.05914 |               177.00000 |                       0.02225 |           0.00098 |
| Qwen/Qwen3.5-9B |      24 |      5 |        0.09687 |          0.06282 |                         0.17776 |                  0.06991 |               188.00000 |                       0.00795 |           0.00122 |

## Scope

- This analysis tests whether conflict *families* occupy reproducible low-rank covariance subspaces. It does not estimate a reusable vector for any individual constraint.
- Without model log-probability margins, the decision-related residualization is approximate.
- Without activation projection interventions, overlap cannot be linked causally to cross-type behavioral effects as in the paper.
- The current two models are checkpoints from one model family; cross-family replication remains necessary.