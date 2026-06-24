# Existing-data ConInstruct geometry findings

This report treats each sample as a three-point geometry (`org`, `new`, `conflict`). It does not interpret the single-sample org→new displacement as a reusable concept vector.

## Label coverage

| model_id        | label   |   n |
|:----------------|:--------|----:|
| Qwen/Qwen3.5-4B | org     | 411 |
| Qwen/Qwen3.5-4B | new     | 381 |
| Qwen/Qwen3.5-4B | neither |  72 |
| Qwen/Qwen3.5-9B | org     | 411 |
| Qwen/Qwen3.5-9B | new     | 393 |
| Qwen/Qwen3.5-9B | neither |  60 |

| model_id        | label              |   n |
|:----------------|:-------------------|----:|
| Qwen/Qwen3.5-4B | direct_no_ack      | 733 |
| Qwen/Qwen3.5-4B | self_resolves      | 125 |
| Qwen/Qwen3.5-4B | asks_clarification |   6 |
| Qwen/Qwen3.5-9B | direct_no_ack      | 717 |
| Qwen/Qwen3.5-9B | self_resolves      | 139 |
| Qwen/Qwen3.5-9B | asks_clarification |   6 |
| Qwen/Qwen3.5-9B | other              |   2 |

## Prompt construction

| model_id        |   n_pair_labelled |   prompt_order_unknown |   last_constraint_choice_rate |   new_choice_rate_when_new_last |   new_choice_rate_when_org_last |
|:----------------|------------------:|-----------------------:|------------------------------:|--------------------------------:|--------------------------------:|
| Qwen/Qwen3.5-4B |               792 |                      0 |                        0.4811 |                          0.4811 |                             nan |
| Qwen/Qwen3.5-9B |               804 |                      0 |                        0.4888 |                          0.4888 |                             nan |

All current conflict prompts append the new constraint after the original expanded instruction. There is no order-swap counterfactual, so recency cannot be separated from new-constraint identity.

## Outcome rates by conflict type

| model_id        |   conflict_type_idx |   n |   new_choice_rate |   neither_rate |   self_resolves_rate |   asks_clarification_rate |
|:----------------|--------------------:|----:|------------------:|---------------:|---------------------:|--------------------------:|
| Qwen/Qwen3.5-4B |                   1 | 100 |            0.3579 |         0.0500 |               0.1300 |                    0.0000 |
| Qwen/Qwen3.5-4B |                   2 | 100 |            0.3933 |         0.1100 |               0.0800 |                    0.0000 |
| Qwen/Qwen3.5-4B |                   3 | 100 |            0.1023 |         0.1200 |               0.0300 |                    0.0000 |
| Qwen/Qwen3.5-4B |                   4 | 100 |            0.3626 |         0.0900 |               0.0800 |                    0.0100 |
| Qwen/Qwen3.5-4B |                   5 | 100 |            0.5000 |         0.1600 |               0.0200 |                    0.0100 |
| Qwen/Qwen3.5-4B |                   6 | 100 |            0.5914 |         0.0700 |               0.1000 |                    0.0000 |
| Qwen/Qwen3.5-4B |                   7 | 100 |            0.4667 |         0.1000 |               0.0900 |                    0.0400 |
| Qwen/Qwen3.5-4B |                   8 |  94 |            0.7097 |         0.0106 |               0.5745 |                    0.0000 |
| Qwen/Qwen3.5-4B |                   9 |  70 |            0.9420 |         0.0143 |               0.2571 |                    0.0000 |
| Qwen/Qwen3.5-9B |                   1 | 100 |            0.3370 |         0.0800 |               0.1700 |                    0.0000 |
| Qwen/Qwen3.5-9B |                   2 | 100 |            0.4713 |         0.1300 |               0.0600 |                    0.0100 |
| Qwen/Qwen3.5-9B |                   3 | 100 |            0.0667 |         0.1000 |               0.0400 |                    0.0000 |
| Qwen/Qwen3.5-9B |                   4 | 100 |            0.3229 |         0.0400 |               0.0800 |                    0.0000 |
| Qwen/Qwen3.5-9B |                   5 | 100 |            0.5056 |         0.1100 |               0.0100 |                    0.0400 |
| Qwen/Qwen3.5-9B |                   6 | 100 |            0.5638 |         0.0600 |               0.0900 |                    0.0000 |
| Qwen/Qwen3.5-9B |                   7 | 100 |            0.5579 |         0.0500 |               0.1700 |                    0.0100 |
| Qwen/Qwen3.5-9B |                   8 |  94 |            0.7609 |         0.0213 |               0.5957 |                    0.0000 |
| Qwen/Qwen3.5-9B |                   9 |  70 |            0.9130 |         0.0143 |               0.3000 |                    0.0000 |

## Overall geometry

| model_id        | metric                        |   min_layer |   min_value |   max_layer |   max_value |
|:----------------|:------------------------------|------------:|------------:|------------:|------------:|
| Qwen/Qwen3.5-4B | mean_position_t               |          11 |      0.3694 |           2 |      0.5125 |
| Qwen/Qwen3.5-4B | mean_log_offaxis_ratio        |          11 |      0.5889 |           0 |      0.8369 |
| Qwen/Qwen3.5-4B | fraction_outside_segment      |          15 |      0.0544 |           0 |      0.2049 |
| Qwen/Qwen3.5-4B | mean_endpoint_relative_hidden |           0 |      0.0117 |          31 |      0.2680 |
| Qwen/Qwen3.5-9B | mean_position_t               |           8 |      0.3953 |          18 |      0.5312 |
| Qwen/Qwen3.5-9B | mean_log_offaxis_ratio        |          15 |      0.5700 |           0 |      0.8497 |
| Qwen/Qwen3.5-9B | fraction_outside_segment      |          12 |      0.0278 |           0 |      0.2199 |
| Qwen/Qwen3.5-9B | mean_endpoint_relative_hidden |           0 |      0.0082 |          31 |      0.2701 |

### Qwen/Qwen3.5-4B

| model_id        |   layer |   n |   mean_position_t |   median_position_t |   fraction_new_side |   fraction_outside_segment |   mean_log_offaxis_ratio |   median_offaxis_ratio |   mean_offaxis_fraction |   mean_midpoint_distance_ratio |   median_midpoint_distance_ratio |   mean_log_midpoint_distance_ratio |   mean_endpoint_relative_hidden |
|:----------------|--------:|----:|------------------:|--------------------:|--------------------:|---------------------------:|-------------------------:|-----------------------:|------------------------:|-------------------------------:|---------------------------------:|-----------------------------------:|--------------------------------:|
| Qwen/Qwen3.5-4B |       1 | 864 |            0.4979 |              0.4975 |              0.4954 |                     0.1308 |                   0.7849 |                 1.1222 |                  0.9392 |                         1.3682 |                           1.1834 |                             0.8139 |                          0.0198 |
| Qwen/Qwen3.5-4B |       5 | 864 |            0.4090 |              0.3816 |              0.3762 |                     0.1007 |                   0.7449 |                 1.0470 |                  0.9385 |                         1.2642 |                           1.0935 |                             0.7729 |                          0.0672 |
| Qwen/Qwen3.5-4B |      10 | 864 |            0.4217 |              0.3930 |              0.3704 |                     0.0706 |                   0.6994 |                 0.9287 |                  0.9354 |                         1.1508 |                           0.9868 |                             0.7275 |                          0.0884 |
| Qwen/Qwen3.5-4B |      15 | 864 |            0.4136 |              0.3797 |              0.3727 |                     0.0544 |                   0.5898 |                 0.7409 |                  0.9132 |                         0.9218 |                           0.7891 |                             0.6255 |                          0.1749 |
| Qwen/Qwen3.5-4B |      20 | 864 |            0.4484 |              0.4396 |              0.4271 |                     0.1424 |                   0.6507 |                 0.7719 |                  0.9001 |                         1.1394 |                           0.8262 |                             0.6940 |                          0.2320 |
| Qwen/Qwen3.5-4B |      25 | 864 |            0.4568 |              0.4130 |              0.4294 |                     0.1875 |                   0.6578 |                 0.7353 |                  0.8839 |                         1.2124 |                           0.7974 |                             0.7082 |                          0.2359 |
| Qwen/Qwen3.5-4B |      31 | 864 |            0.4743 |              0.4442 |              0.4537 |                     0.1759 |                   0.6905 |                 0.8014 |                  0.8953 |                         1.2848 |                           0.8477 |                             0.7355 |                          0.2680 |

### Qwen/Qwen3.5-9B

| model_id        |   layer |   n |   mean_position_t |   median_position_t |   fraction_new_side |   fraction_outside_segment |   mean_log_offaxis_ratio |   median_offaxis_ratio |   mean_offaxis_fraction |   mean_midpoint_distance_ratio |   median_midpoint_distance_ratio |   mean_log_midpoint_distance_ratio |   mean_endpoint_relative_hidden |
|:----------------|--------:|----:|------------------:|--------------------:|--------------------:|---------------------------:|-------------------------:|-----------------------:|------------------------:|-------------------------------:|---------------------------------:|-----------------------------------:|--------------------------------:|
| Qwen/Qwen3.5-9B |       1 | 864 |            0.5182 |              0.5301 |              0.5266 |                     0.1759 |                   0.8127 |                 1.1523 |                  0.9408 |                         1.4639 |                           1.2089 |                             0.8418 |                          0.0119 |
| Qwen/Qwen3.5-9B |       5 | 864 |            0.4128 |              0.3934 |              0.3681 |                     0.1227 |                   0.7227 |                 0.9939 |                  0.9323 |                         1.2033 |                           1.0509 |                             0.7542 |                          0.0375 |
| Qwen/Qwen3.5-9B |      10 | 864 |            0.4248 |              0.4014 |              0.3692 |                     0.0544 |                   0.6600 |                 0.8760 |                  0.9370 |                         1.0478 |                           0.9200 |                             0.6874 |                          0.0604 |
| Qwen/Qwen3.5-9B |      15 | 864 |            0.4876 |              0.4715 |              0.4606 |                     0.0475 |                   0.5700 |                 0.6726 |                  0.9202 |                         0.8994 |                           0.7060 |                             0.6018 |                          0.1505 |
| Qwen/Qwen3.5-9B |      20 | 864 |            0.5203 |              0.5153 |              0.5127 |                     0.1065 |                   0.6808 |                 0.8132 |                  0.9157 |                         1.1883 |                           0.8612 |                             0.7162 |                          0.2152 |
| Qwen/Qwen3.5-9B |      25 | 864 |            0.5191 |              0.4875 |              0.4861 |                     0.1377 |                   0.7089 |                 0.7989 |                  0.9055 |                         1.3345 |                           0.8502 |                             0.7490 |                          0.2268 |
| Qwen/Qwen3.5-9B |      31 | 864 |            0.4953 |              0.4562 |              0.4456 |                     0.1586 |                   0.7422 |                 0.8407 |                  0.9125 |                         1.4436 |                           0.8823 |                             0.7796 |                          0.2701 |

## Behavior 1 versus behavior 3

| model_id        | stage        | metric                      | group                        |   n |      mean |    ci_low |   ci_high |   mannwhitney_u |   p_value |
|:----------------|:-------------|:----------------------------|:-----------------------------|----:|----------:|----------:|----------:|----------------:|----------:|
| Qwen/Qwen3.5-4B | early_00_09  | position_t                  | direct_no_ack                | 733 |   0.43030 |   0.41158 |   0.45043 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | position_t                  | self_resolves                | 125 |   0.46376 |   0.41750 |   0.51275 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | position_t                  | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     42341.00000 |   0.17532 |
| Qwen/Qwen3.5-4B | early_00_09  | oriented_position           | direct_no_ack                | 733 |   0.54501 |   0.52513 |   0.56539 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | oriented_position           | self_resolves                | 125 |   0.51987 |   0.47335 |   0.56769 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | oriented_position           | direct_vs_self_resolves_test | 789 | nan       | nan       | nan       |     42076.00000 |   0.33462 |
| Qwen/Qwen3.5-4B | early_00_09  | log_offaxis_ratio           | direct_no_ack                | 733 |   0.73838 |   0.71631 |   0.76182 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | log_offaxis_ratio           | self_resolves                | 125 |   0.75979 |   0.71703 |   0.80403 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | log_offaxis_ratio           | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     41987.00000 |   0.13529 |
| Qwen/Qwen3.5-4B | early_00_09  | offaxis_fraction            | direct_no_ack                | 733 |   0.93271 |   0.92638 |   0.93883 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | offaxis_fraction            | self_resolves                | 125 |   0.94253 |   0.92592 |   0.95702 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | offaxis_fraction            | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     40784.00000 |   0.04961 |
| Qwen/Qwen3.5-4B | early_00_09  | log_midpoint_distance_ratio | direct_no_ack                | 733 |   0.76882 |   0.74850 |   0.79023 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | log_midpoint_distance_ratio | self_resolves                | 125 |   0.78625 |   0.74406 |   0.82977 |       nan       | nan       |
| Qwen/Qwen3.5-4B | early_00_09  | log_midpoint_distance_ratio | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     42378.00000 |   0.17996 |
| Qwen/Qwen3.5-4B | late_22_31   | position_t                  | direct_no_ack                | 733 |   0.43470 |   0.40204 |   0.46541 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | position_t                  | self_resolves                | 125 |   0.59449 |   0.51411 |   0.68157 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | position_t                  | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     36476.00000 |   0.00027 |
| Qwen/Qwen3.5-4B | late_22_31   | oriented_position           | direct_no_ack                | 733 |   0.54579 |   0.51252 |   0.57832 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | oriented_position           | self_resolves                | 125 |   0.63931 |   0.56120 |   0.72798 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | oriented_position           | direct_vs_self_resolves_test | 789 | nan       | nan       | nan       |     35370.00000 |   0.04979 |
| Qwen/Qwen3.5-4B | late_22_31   | log_offaxis_ratio           | direct_no_ack                | 733 |   0.66562 |   0.63927 |   0.69390 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | log_offaxis_ratio           | self_resolves                | 125 |   0.64342 |   0.57542 |   0.71264 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | log_offaxis_ratio           | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     47493.00000 |   0.51183 |
| Qwen/Qwen3.5-4B | late_22_31   | offaxis_fraction            | direct_no_ack                | 733 |   0.88828 |   0.87767 |   0.89931 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | offaxis_fraction            | self_resolves                | 125 |   0.88302 |   0.85507 |   0.90989 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | offaxis_fraction            | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     46650.00000 |   0.74380 |
| Qwen/Qwen3.5-4B | late_22_31   | log_midpoint_distance_ratio | direct_no_ack                | 733 |   0.71340 |   0.68795 |   0.74045 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | log_midpoint_distance_ratio | self_resolves                | 125 |   0.69573 |   0.63593 |   0.75832 |       nan       | nan       |
| Qwen/Qwen3.5-4B | late_22_31   | log_midpoint_distance_ratio | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     46880.00000 |   0.67695 |
| Qwen/Qwen3.5-4B | middle_10_21 | position_t                  | direct_no_ack                | 733 |   0.41755 |   0.39806 |   0.43721 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | position_t                  | self_resolves                | 125 |   0.50283 |   0.45375 |   0.55407 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | position_t                  | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     37472.00000 |   0.00113 |
| Qwen/Qwen3.5-4B | middle_10_21 | oriented_position           | direct_no_ack                | 733 |   0.53353 |   0.51209 |   0.55503 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | oriented_position           | self_resolves                | 125 |   0.55997 |   0.51362 |   0.60700 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | oriented_position           | direct_vs_self_resolves_test | 789 | nan       | nan       | nan       |     38087.00000 |   0.43784 |
| Qwen/Qwen3.5-4B | middle_10_21 | log_offaxis_ratio           | direct_no_ack                | 733 |   0.63652 |   0.61851 |   0.65530 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | log_offaxis_ratio           | self_resolves                | 125 |   0.66810 |   0.62369 |   0.71742 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | log_offaxis_ratio           | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     42759.00000 |   0.23322 |
| Qwen/Qwen3.5-4B | middle_10_21 | offaxis_fraction            | direct_no_ack                | 733 |   0.91523 |   0.90728 |   0.92250 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | offaxis_fraction            | self_resolves                | 125 |   0.93189 |   0.91332 |   0.94702 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | offaxis_fraction            | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     40793.00000 |   0.05002 |
| Qwen/Qwen3.5-4B | middle_10_21 | log_midpoint_distance_ratio | direct_no_ack                | 733 |   0.67237 |   0.65601 |   0.68940 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | log_midpoint_distance_ratio | self_resolves                | 125 |   0.69832 |   0.65373 |   0.74422 |       nan       | nan       |
| Qwen/Qwen3.5-4B | middle_10_21 | log_midpoint_distance_ratio | direct_vs_self_resolves_test | 858 | nan       | nan       | nan       |     43211.00000 |   0.30981 |
| Qwen/Qwen3.5-9B | early_00_09  | position_t                  | direct_no_ack                | 717 |   0.42986 |   0.40842 |   0.45116 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | position_t                  | self_resolves                | 139 |   0.50178 |   0.45784 |   0.54595 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | position_t                  | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     41821.00000 |   0.00268 |
| Qwen/Qwen3.5-9B | early_00_09  | oriented_position           | direct_no_ack                | 717 |   0.54950 |   0.52746 |   0.57068 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | oriented_position           | self_resolves                | 139 |   0.51057 |   0.46540 |   0.55280 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | oriented_position           | direct_vs_self_resolves_test | 800 | nan       | nan       | nan       |     48145.00000 |   0.18335 |
| Qwen/Qwen3.5-9B | early_00_09  | log_offaxis_ratio           | direct_no_ack                | 717 |   0.73128 |   0.71015 |   0.75363 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | log_offaxis_ratio           | self_resolves                | 139 |   0.72755 |   0.68720 |   0.76871 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | log_offaxis_ratio           | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     48836.00000 |   0.70918 |
| Qwen/Qwen3.5-9B | early_00_09  | offaxis_fraction            | direct_no_ack                | 717 |   0.93102 |   0.92412 |   0.93728 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | offaxis_fraction            | self_resolves                | 139 |   0.94104 |   0.92497 |   0.95467 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | offaxis_fraction            | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     44751.00000 |   0.05689 |
| Qwen/Qwen3.5-9B | early_00_09  | log_midpoint_distance_ratio | direct_no_ack                | 717 |   0.76315 |   0.74368 |   0.78446 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | log_midpoint_distance_ratio | self_resolves                | 139 |   0.75492 |   0.71829 |   0.79474 |       nan       | nan       |
| Qwen/Qwen3.5-9B | early_00_09  | log_midpoint_distance_ratio | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     49641.00000 |   0.94322 |
| Qwen/Qwen3.5-9B | late_22_31   | position_t                  | direct_no_ack                | 717 |   0.49033 |   0.45954 |   0.52109 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | position_t                  | self_resolves                | 139 |   0.59771 |   0.53132 |   0.66565 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | position_t                  | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     41266.00000 |   0.00133 |
| Qwen/Qwen3.5-9B | late_22_31   | oriented_position           | direct_no_ack                | 717 |   0.53571 |   0.50369 |   0.56756 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | oriented_position           | self_resolves                | 139 |   0.54661 |   0.47227 |   0.61473 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | oriented_position           | direct_vs_self_resolves_test | 800 | nan       | nan       | nan       |     45150.00000 |   0.91477 |
| Qwen/Qwen3.5-9B | late_22_31   | log_offaxis_ratio           | direct_no_ack                | 717 |   0.72282 |   0.69273 |   0.75426 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | log_offaxis_ratio           | self_resolves                | 139 |   0.68269 |   0.61509 |   0.75619 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | log_offaxis_ratio           | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     54115.00000 |   0.10841 |
| Qwen/Qwen3.5-9B | late_22_31   | offaxis_fraction            | direct_no_ack                | 717 |   0.91162 |   0.90158 |   0.92067 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | offaxis_fraction            | self_resolves                | 139 |   0.89484 |   0.86962 |   0.91846 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | offaxis_fraction            | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     52645.00000 |   0.29170 |
| Qwen/Qwen3.5-9B | late_22_31   | log_midpoint_distance_ratio | direct_no_ack                | 717 |   0.76029 |   0.73099 |   0.78957 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | log_midpoint_distance_ratio | self_resolves                | 139 |   0.72766 |   0.65936 |   0.79857 |       nan       | nan       |
| Qwen/Qwen3.5-9B | late_22_31   | log_midpoint_distance_ratio | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     53967.00000 |   0.12116 |
| Qwen/Qwen3.5-9B | middle_10_21 | position_t                  | direct_no_ack                | 717 |   0.48440 |   0.46554 |   0.50328 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | position_t                  | self_resolves                | 139 |   0.55246 |   0.51136 |   0.59612 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | position_t                  | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     42664.00000 |   0.00722 |
| Qwen/Qwen3.5-9B | middle_10_21 | oriented_position           | direct_no_ack                | 717 |   0.51118 |   0.49197 |   0.53072 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | oriented_position           | self_resolves                | 139 |   0.50510 |   0.46096 |   0.54979 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | oriented_position           | direct_vs_self_resolves_test | 800 | nan       | nan       | nan       |     45609.00000 |   0.76835 |
| Qwen/Qwen3.5-9B | middle_10_21 | log_offaxis_ratio           | direct_no_ack                | 717 |   0.63669 |   0.61744 |   0.65632 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | log_offaxis_ratio           | self_resolves                | 139 |   0.63617 |   0.59000 |   0.68526 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | log_offaxis_ratio           | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     50606.00000 |   0.77173 |
| Qwen/Qwen3.5-9B | middle_10_21 | offaxis_fraction            | direct_no_ack                | 717 |   0.92649 |   0.91903 |   0.93349 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | offaxis_fraction            | self_resolves                | 139 |   0.92524 |   0.90888 |   0.94112 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | offaxis_fraction            | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     50169.00000 |   0.89948 |
| Qwen/Qwen3.5-9B | middle_10_21 | log_midpoint_distance_ratio | direct_no_ack                | 717 |   0.66701 |   0.64858 |   0.68503 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | log_midpoint_distance_ratio | self_resolves                | 139 |   0.66775 |   0.62539 |   0.71654 |       nan       | nan       |
| Qwen/Qwen3.5-9B | middle_10_21 | log_midpoint_distance_ratio | direct_vs_self_resolves_test | 856 | nan       | nan       | nan       |     50276.00000 |   0.86782 |

## Behavior effects after controlling conflict type and chosen side

| model_id        | stage        | metric                      |   n_direct |   n_self_resolves |   direct_mean_centered |   self_resolves_mean_centered |   self_resolves_minus_direct |   mannwhitney_u |   p_value |   p_value_bh |
|:----------------|:-------------|:----------------------------|-----------:|------------------:|-----------------------:|------------------------------:|-----------------------------:|----------------:|----------:|-------------:|
| Qwen/Qwen3.5-4B | early_00_09  | position_t                  |        670 |               119 |                0.00461 |                      -0.02594 |                     -0.03055 |     41997.00000 |   0.35218 |      0.65019 |
| Qwen/Qwen3.5-4B | late_22_31   | position_t                  |        670 |               119 |               -0.00447 |                       0.02518 |                      0.02965 |     39097.00000 |   0.73763 |      0.96904 |
| Qwen/Qwen3.5-4B | middle_10_21 | position_t                  |        670 |               119 |                0.00094 |                      -0.00531 |                     -0.00625 |     39711.00000 |   0.94658 |      0.96904 |
| Qwen/Qwen3.5-9B | early_00_09  | position_t                  |        665 |               135 |                0.00108 |                      -0.00531 |                     -0.00639 |     44424.00000 |   0.84999 |      0.96904 |
| Qwen/Qwen3.5-9B | late_22_31   | position_t                  |        665 |               135 |               -0.00412 |                       0.02029 |                      0.02441 |     43219.00000 |   0.49563 |      0.74344 |
| Qwen/Qwen3.5-9B | middle_10_21 | position_t                  |        665 |               135 |               -0.00275 |                       0.01353 |                      0.01628 |     43062.00000 |   0.45596 |      0.72953 |
| Qwen/Qwen3.5-4B | early_00_09  | log_offaxis_ratio           |        670 |               119 |               -0.00331 |                       0.01863 |                      0.02194 |     36163.00000 |   0.10617 |      0.52133 |
| Qwen/Qwen3.5-4B | late_22_31   | log_offaxis_ratio           |        670 |               119 |               -0.00322 |                       0.01810 |                      0.02132 |     37694.00000 |   0.34344 |      0.65019 |
| Qwen/Qwen3.5-4B | middle_10_21 | log_offaxis_ratio           |        670 |               119 |               -0.00406 |                       0.02287 |                      0.02693 |     36564.00000 |   0.14970 |      0.52133 |
| Qwen/Qwen3.5-9B | early_00_09  | log_offaxis_ratio           |        665 |               135 |               -0.00298 |                       0.01469 |                      0.01768 |     41338.00000 |   0.14712 |      0.52133 |
| Qwen/Qwen3.5-9B | late_22_31   | log_offaxis_ratio           |        665 |               135 |               -0.00192 |                       0.00948 |                      0.01141 |     44792.00000 |   0.96904 |      0.96904 |
| Qwen/Qwen3.5-9B | middle_10_21 | log_offaxis_ratio           |        665 |               135 |               -0.00036 |                       0.00176 |                      0.00212 |     44363.00000 |   0.83050 |      0.96904 |
| Qwen/Qwen3.5-4B | early_00_09  | offaxis_fraction            |        670 |               119 |               -0.00193 |                       0.01085 |                      0.01277 |     30999.00000 |   0.00011 |      0.00262 |
| Qwen/Qwen3.5-4B | late_22_31   | offaxis_fraction            |        670 |               119 |               -0.00069 |                       0.00390 |                      0.00460 |     37972.00000 |   0.40878 |      0.70077 |
| Qwen/Qwen3.5-4B | middle_10_21 | offaxis_fraction            |        670 |               119 |               -0.00233 |                       0.01314 |                      0.01547 |     33914.00000 |   0.00940 |      0.07517 |
| Qwen/Qwen3.5-9B | early_00_09  | offaxis_fraction            |        665 |               135 |               -0.00178 |                       0.00875 |                      0.01052 |     37340.00000 |   0.00205 |      0.02459 |
| Qwen/Qwen3.5-9B | late_22_31   | offaxis_fraction            |        665 |               135 |                0.00164 |                      -0.00809 |                     -0.00973 |     45278.00000 |   0.87342 |      0.96904 |
| Qwen/Qwen3.5-9B | middle_10_21 | offaxis_fraction            |        665 |               135 |               -0.00037 |                       0.00183 |                      0.00221 |     41905.00000 |   0.22316 |      0.62095 |
| Qwen/Qwen3.5-4B | early_00_09  | log_midpoint_distance_ratio |        670 |               119 |               -0.00258 |                       0.01453 |                      0.01712 |     36583.00000 |   0.15205 |      0.52133 |
| Qwen/Qwen3.5-4B | late_22_31   | log_midpoint_distance_ratio |        670 |               119 |               -0.00304 |                       0.01714 |                      0.02018 |     37413.00000 |   0.28460 |      0.62095 |
| Qwen/Qwen3.5-4B | middle_10_21 | log_midpoint_distance_ratio |        670 |               119 |               -0.00319 |                       0.01797 |                      0.02116 |     37348.00000 |   0.27203 |      0.62095 |
| Qwen/Qwen3.5-9B | early_00_09  | log_midpoint_distance_ratio |        665 |               135 |               -0.00230 |                       0.01131 |                      0.01360 |     42207.00000 |   0.27361 |      0.62095 |
| Qwen/Qwen3.5-9B | late_22_31   | log_midpoint_distance_ratio |        665 |               135 |               -0.00258 |                       0.01271 |                      0.01529 |     44609.00000 |   0.90958 |      0.96904 |
| Qwen/Qwen3.5-9B | middle_10_21 | log_midpoint_distance_ratio |        665 |               135 |               -0.00026 |                       0.00126 |                      0.00152 |     44331.00000 |   0.82032 |      0.96904 |

## Within-conflict-type output-choice AUC

| model_id        | stage        | metric            |   mean |   median |    std |
|:----------------|:-------------|:------------------|-------:|---------:|-------:|
| Qwen/Qwen3.5-4B | early_00_09  | log_offaxis_ratio | 0.5350 |   0.5410 | 0.1310 |
| Qwen/Qwen3.5-4B | early_00_09  | offaxis_fraction  | 0.5168 |   0.5129 | 0.0973 |
| Qwen/Qwen3.5-4B | early_00_09  | position_t        | 0.5752 |   0.5836 | 0.0730 |
| Qwen/Qwen3.5-4B | late_22_31   | log_offaxis_ratio | 0.5277 |   0.5359 | 0.0902 |
| Qwen/Qwen3.5-4B | late_22_31   | offaxis_fraction  | 0.5051 |   0.4885 | 0.0759 |
| Qwen/Qwen3.5-4B | late_22_31   | position_t        | 0.5366 |   0.4793 | 0.1235 |
| Qwen/Qwen3.5-4B | middle_10_21 | log_offaxis_ratio | 0.5543 |   0.5247 | 0.1075 |
| Qwen/Qwen3.5-4B | middle_10_21 | offaxis_fraction  | 0.5497 |   0.5677 | 0.0873 |
| Qwen/Qwen3.5-4B | middle_10_21 | position_t        | 0.5432 |   0.5513 | 0.0874 |
| Qwen/Qwen3.5-9B | early_00_09  | log_offaxis_ratio | 0.5395 |   0.5209 | 0.1362 |
| Qwen/Qwen3.5-9B | early_00_09  | offaxis_fraction  | 0.5328 |   0.5417 | 0.0644 |
| Qwen/Qwen3.5-9B | early_00_09  | position_t        | 0.5627 |   0.5719 | 0.0675 |
| Qwen/Qwen3.5-9B | late_22_31   | log_offaxis_ratio | 0.5337 |   0.5117 | 0.1170 |
| Qwen/Qwen3.5-9B | late_22_31   | offaxis_fraction  | 0.5181 |   0.5085 | 0.0472 |
| Qwen/Qwen3.5-9B | late_22_31   | position_t        | 0.5231 |   0.4627 | 0.1626 |
| Qwen/Qwen3.5-9B | middle_10_21 | log_offaxis_ratio | 0.5355 |   0.5727 | 0.0836 |
| Qwen/Qwen3.5-9B | middle_10_21 | offaxis_fraction  | 0.5407 |   0.5219 | 0.0789 |
| Qwen/Qwen3.5-9B | middle_10_21 | position_t        | 0.5107 |   0.5568 | 0.1451 |

## Grouped-CV prediction

| model_id        | target          | feature_group           |   n_samples |   positive_rate |   balanced_accuracy_mean |   balanced_accuracy_std |   roc_auc_mean |   roc_auc_std |
|:----------------|:----------------|:------------------------|------------:|----------------:|-------------------------:|------------------------:|---------------:|--------------:|
| Qwen/Qwen3.5-4B | pair_choice     | conflict_type_only      |         792 |          0.4811 |                   0.6648 |                  0.0381 |         0.7281 |        0.0370 |
| Qwen/Qwen3.5-4B | pair_choice     | axis_trajectory         |         792 |          0.4811 |                   0.5919 |                  0.0385 |         0.6314 |        0.0477 |
| Qwen/Qwen3.5-4B | pair_choice     | offaxis_trajectory      |         792 |          0.4811 |                   0.5833 |                  0.0203 |         0.6036 |        0.0214 |
| Qwen/Qwen3.5-4B | pair_choice     | full_geometry           |         792 |          0.4811 |                   0.6173 |                  0.0362 |         0.6524 |        0.0280 |
| Qwen/Qwen3.5-4B | pair_choice     | full_geometry_plus_type |         792 |          0.4811 |                   0.6717 |                  0.0448 |         0.7220 |        0.0288 |
| Qwen/Qwen3.5-4B | behavior_1_vs_3 | conflict_type_only      |         858 |          0.1457 |                   0.7223 |                  0.0523 |         0.7774 |        0.0314 |
| Qwen/Qwen3.5-4B | behavior_1_vs_3 | axis_trajectory         |         858 |          0.1457 |                   0.6109 |                  0.0408 |         0.6501 |        0.0676 |
| Qwen/Qwen3.5-4B | behavior_1_vs_3 | offaxis_trajectory      |         858 |          0.1457 |                   0.6050 |                  0.0651 |         0.6182 |        0.0741 |
| Qwen/Qwen3.5-4B | behavior_1_vs_3 | full_geometry           |         858 |          0.1457 |                   0.6241 |                  0.0536 |         0.6931 |        0.0446 |
| Qwen/Qwen3.5-4B | behavior_1_vs_3 | full_geometry_plus_type |         858 |          0.1457 |                   0.6731 |                  0.0447 |         0.7350 |        0.0221 |
| Qwen/Qwen3.5-9B | pair_choice     | conflict_type_only      |         804 |          0.4888 |                   0.6635 |                  0.0352 |         0.7414 |        0.0313 |
| Qwen/Qwen3.5-9B | pair_choice     | axis_trajectory         |         804 |          0.4888 |                   0.6400 |                  0.0290 |         0.6677 |        0.0477 |
| Qwen/Qwen3.5-9B | pair_choice     | offaxis_trajectory      |         804 |          0.4888 |                   0.5973 |                  0.0275 |         0.6300 |        0.0225 |
| Qwen/Qwen3.5-9B | pair_choice     | full_geometry           |         804 |          0.4888 |                   0.6449 |                  0.0526 |         0.7015 |        0.0440 |
| Qwen/Qwen3.5-9B | pair_choice     | full_geometry_plus_type |         804 |          0.4888 |                   0.6839 |                  0.0183 |         0.7571 |        0.0200 |
| Qwen/Qwen3.5-9B | behavior_1_vs_3 | conflict_type_only      |         856 |          0.1624 |                   0.6929 |                  0.0369 |         0.7892 |        0.0401 |
| Qwen/Qwen3.5-9B | behavior_1_vs_3 | axis_trajectory         |         856 |          0.1624 |                   0.6253 |                  0.0520 |         0.6515 |        0.0516 |
| Qwen/Qwen3.5-9B | behavior_1_vs_3 | offaxis_trajectory      |         856 |          0.1624 |                   0.5760 |                  0.0267 |         0.5985 |        0.0484 |
| Qwen/Qwen3.5-9B | behavior_1_vs_3 | full_geometry           |         856 |          0.1624 |                   0.6118 |                  0.0256 |         0.6690 |        0.0280 |
| Qwen/Qwen3.5-9B | behavior_1_vs_3 | full_geometry_plus_type |         856 |          0.1624 |                   0.6625 |                  0.0277 |         0.7249 |        0.0515 |

## Leave-one-conflict-type-out prediction

| ('model_id', '')   | ('target', '')   | ('feature_group', '')   |   ('balanced_accuracy', 'mean') |   ('balanced_accuracy', 'std') |   ('roc_auc', 'mean') |   ('roc_auc', 'std') |
|:-------------------|:-----------------|:------------------------|--------------------------------:|-------------------------------:|----------------------:|---------------------:|
| Qwen/Qwen3.5-4B    | behavior_1_vs_3  | axis_trajectory         |                          0.5169 |                         0.0881 |                0.5216 |               0.1256 |
| Qwen/Qwen3.5-4B    | behavior_1_vs_3  | full_geometry           |                          0.5425 |                         0.1086 |                0.5624 |               0.0893 |
| Qwen/Qwen3.5-4B    | behavior_1_vs_3  | full_geometry_plus_type |                          0.5425 |                         0.1086 |                0.5624 |               0.0893 |
| Qwen/Qwen3.5-4B    | pair_choice      | axis_trajectory         |                          0.5439 |                         0.1213 |                0.5637 |               0.1113 |
| Qwen/Qwen3.5-4B    | pair_choice      | full_geometry           |                          0.5234 |                         0.0665 |                0.5031 |               0.0873 |
| Qwen/Qwen3.5-4B    | pair_choice      | full_geometry_plus_type |                          0.5234 |                         0.0665 |                0.5031 |               0.0873 |
| Qwen/Qwen3.5-9B    | behavior_1_vs_3  | axis_trajectory         |                          0.4927 |                         0.0870 |                0.5359 |               0.1068 |
| Qwen/Qwen3.5-9B    | behavior_1_vs_3  | full_geometry           |                          0.5086 |                         0.1153 |                0.5261 |               0.1500 |
| Qwen/Qwen3.5-9B    | behavior_1_vs_3  | full_geometry_plus_type |                          0.5086 |                         0.1153 |                0.5261 |               0.1500 |
| Qwen/Qwen3.5-9B    | pair_choice      | axis_trajectory         |                          0.5668 |                         0.0591 |                0.5756 |               0.0639 |
| Qwen/Qwen3.5-9B    | pair_choice      | full_geometry           |                          0.5224 |                         0.0453 |                0.5375 |               0.0772 |
| Qwen/Qwen3.5-9B    | pair_choice      | full_geometry_plus_type |                          0.5224 |                         0.0453 |                0.5375 |               0.0772 |

## Cross-model normalized-geometry consistency

| metric                   |   mean |    min |    max |
|:-------------------------|-------:|-------:|-------:|
| endpoint_relative_hidden | 0.8662 | 0.6615 | 0.9391 |
| log_offaxis_ratio        | 0.8280 | 0.6806 | 0.9031 |
| offaxis_fraction         | 0.4774 | 0.2716 | 0.5957 |
| position_t               | 0.6399 | 0.5064 | 0.7115 |
| signed_margin            | 0.6399 | 0.5064 | 0.7115 |

| comparison      |   n |   agreement |
|:----------------|----:|------------:|
| pair_org_vs_new | 755 |      0.8146 |
| behavior_full   | 864 |      0.8623 |

## Interpretation limits

- Hidden states are from the final prompt position before generation, not answer-prefix or generated-token states.
- There is no neutral seed hidden state in the existing extraction, so the analysis cannot identify two independently estimated concept vectors.
- Off-axis displacement can reflect conflict interaction, prompt length, or composition of two constraints. A compatible two-constraint control is required before calling it conflict-specific.
- Evaluation labels are model-judge outputs and contain judge noise.
- Every current conflict prompt places the new constraint last, so the existing runs cannot identify a causal recency effect.