[1mdiff --git a/two_signalers.py b/two_signalers.py[m
[1mindex 31ec601..86612dd 100644[m
[1m--- a/two_signalers.py[m
[1m+++ b/two_signalers.py[m
[36m@@ -11,8 +11,8 @@[m [mfrom typing import Tuple, Callable[m
 Report_States = ["SS", "SI", "IS", "II"][m
 [m
 # The underlying probability whether a path is secure S (q_S) or insecure I (1 - q_S)[m
[31m-Q = {0: .99,[m
[31m-     1: .98}[m
[32m+[m[32mQ = {0: .9,[m
[32m+[m[32m     1: .9}[m
 [m
 # p_state_path is the probability with which the signaler will truthfully report the security[m
 #     state of the path when the path is in specified state[m
[36m@@ -27,8 +27,8 @@[m [mReport_Schedule = {0: {'S': .5,[m
 # The security cost is the cost associated with the security state of a path being insecure[m
 # The key is the path j and the value is the security cost[m
 # You can set these values to anything you like as I've clamped the equilibrium values accordingly.[m
[31m-Security_Costs = {0: .5,[m
[31m-                  1: 20}[m
[32m+[m[32mSecurity_Costs = {0: 2,[m
[32m+[m[32m                  1: 1.8}[m
 [m
 # If you wish to 'zoom in' to any part of the heat map, you can change these limit values. Keep in mind that[m
 # the start values cannot be less OR EQUAL TO zero. They cannot be equal to zero as this will put a zero in the[m
[36m@@ -184,7 +184,7 @@[m [mdef get_data(report_states, c: dict[int: float], P: dict[int, Probability],[m
             # This sets probability of truthfulness when path 1 is insecure[m
             # and is the y-dimension of the heat map[m
             y_index(j, y[j])[m
[31m-            Z[j, i] = data_func(report_states, c, P)[m
[32m+[m[32m            Z[i, j] = data_func(report_states, c, P)[m
             # Z[i, j] = exp_sec_cost(report_states, c, P)[m
     return X, Y, Z[m
 [m
