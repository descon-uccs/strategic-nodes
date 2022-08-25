from __future__ import annotations
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Tuple, Callable
# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from colorspacious import cspace_converter

# The state space for reports
Report_States = ["SS", "SI", "IS", "II"]

# The underlying probability whether a path is secure S (q_S) or insecure I (1 - q_S)
Q = {0: .9,
     1: .9}

# p_state_path is the probability with which the signaler will truthfully report the security
#     state of the path when the path is in specified state
# {path : {state : p_}}
# e.g., if path 1 is secure, then there is a Report_Schedule[1]['S'] chance that the second letter in the report
#   will be 'S'
Report_Schedule = {0: {'S': .5,
                       'I': 1},
                   1: {'S': .5,
                       'I': 1}}

# The security cost is the cost associated with the security state of a path being insecure
# The key is the path j and the value is the security cost
# You can set these values to anything you like as I've clamped the equilibrium values accordingly.
Security_Costs = {0: 2,
                  1: 1.8}

# If you wish to 'zoom in' to any part of the heat map, you can change these limit values. Keep in mind that
# the start values cannot be less OR EQUAL TO zero. They cannot be equal to zero as this will put a zero in the
# denominator of the bayesian equation.
axis_lims = {'x': {'start': 0.5,
                   'stop': 1.0},
             'y': {'start': 0.5,
                   'stop': 1.0}}

# You can turn this down to 50 for performance, and up to about 300 or 400 for better looks
ax_len = 100

# These are set later when the output function set-up runs
plot_title = " "
x_label = " "
y_label = " "


# The probability class contains the probability parameters for a single path. Each path will have a
# corresponding instantiation of the Probability class stored in a dictionary keyed on the path's index
class Probability:
    def __init__(self, security_probability, report_schedule):
        self.q = security_probability
        self.rs = report_schedule

    def prob_is_insecure(self, given_r=None) -> float:
        # If not report is given, then this simply returns the probability that this path is insecure, or
        # 1 - q_S
        if not given_r:
            return 1 - self.q
        else:
            assert(len(given_r) == 1)
            # We define the following statements:
            #  A - The path is insecure
            #  B - The report is the state given by r
            # The following is the implementation of P[A | B] = P[B | A] * P[A] / P[B]
            return (self.prob_report_is(given_r, given_theta='I')
                    * self.prob_is_insecure())\
                   / self.prob_report_is(given_r)

    def prob_report_is(self, report_state: str, given_theta=None) -> float:
        assert(len(report_state) == 1)
        assert(report_state == 'S' or report_state == 'I')
        default_value: float
        if not given_theta:
            # This defaults to:
            #  The report r is secure S
            #  so (the probability it's secure) * (how often we truthfully report when secure)
            #      + (probability it's insecure) * (how often we falsely report secure when it's insecure)
            default_value = self.q * self.rs['S'] + ((1 - self.q) * (1 - self.rs['I']))
            if report_state == 'S':
                return default_value
            else:
                return 1.0 - default_value
        else:
            assert(len(given_theta) == 1)
            assert(given_theta == 'S' or given_theta == 'I')
            # theta is the true security state of the path. We report truthfully according to p_theta.
            # So we return p_theta when the report is the same as the real state. Otherwise, we return
            # 1 - p_theta
            default_value = self.rs[given_theta]
            if given_theta == report_state:
                return default_value
            else:
                return 1 - default_value


def get_equilibrium_flow(report_state: str, c: dict[int, float], P: dict[int, Probability]) -> Tuple[float, float]:
    # equilibrium is reached when X_0 + P[I | R=r_0] * c_0 = X_1 + P[I | R=r_1] * c_1
    # Unfortunately, this is hard-coded for two paths
    # These values are calculated for a specific report (e.g., 'SS') within the report space
    X_0 = .5 * (1
                - c[0] * P[0].prob_is_insecure(given_r=report_state[0])
                + c[1] * P[1].prob_is_insecure(given_r=report_state[1]))
    # Clamp the output
    if X_0 > 1.0:
        X_0 = 1.0
    elif X_0 < 0.0:
        X_0 = 0.0

    return X_0, 1 - X_0


def prob_of_report(report_state: str, P: dict[int, Probability]) -> float:
    # This calculates the probability of a given report state. It assumes the probabilities of these
    # are independent
    total = 1.0
    for j in range(len(report_state)):
        total *= P[j].prob_report_is(report_state[j])
    return total


def exp_latency(report_states: list[str], c: dict[int: float], P: dict[int, Probability]):
    # Calculates network latency for each possible report state, then aggregates this to arrive at
    # expected latency across the report state space
    exp_lat = 0.0
    for report_state in report_states:
        network_latency = 0.0
        X = get_equilibrium_flow(report_state, c, P)
        for flow_rate in X:
            network_latency += pow(flow_rate, 2)
        exp_lat += prob_of_report(report_state, P) * network_latency
    return exp_lat


def network_security_cost(report_state: str, c: dict[int: float], P: dict[int, Probability]):
    # Calculates the total security cost of the network for a give report state
    total = 0.0
    X = get_equilibrium_flow(report_state, c, P)
    for j in range(len(report_state)):
        total += P[j].prob_is_insecure(report_state[j]) * c[j] * X[j]
    return total


def exp_sec_cost(report_states, c: dict[int: float], P: dict[int, Probability]):
    total = 0.0
    for report_state in report_states:
        total += prob_of_report(report_state, P) * network_security_cost(report_state, c, P)
    return total


def set_p_s(index: int, value: float):
    Report_Schedule[1]['S'] = value


def set_p_i(index: int, value: float):
    Report_Schedule[1]['I'] = value


def get_data(report_states, c: dict[int: float], P: dict[int, Probability],
             data_func: Callable[[list[str], dict[int: float], dict[int: Probability]], float],
             x_index: Callable[[int, float], None],
             y_index: Callable[[int, float], None]):
    x = np.linspace(axis_lims['x']['start'], axis_lims['x']['stop'],  ax_len)
    y = np.linspace(axis_lims['y']['start'], axis_lims['y']['stop'],  ax_len)
    X, Y = np.meshgrid(x, y)
    Z = np.empty((ax_len, ax_len))
    for i in range(ax_len):
        # This sets probability of truthfulness when path 1 is secure
        # and is the x-dimension of the heat map
        x_index(i, x[i])
        for j in range(ax_len):
            # This sets probability of truthfulness when path 1 is insecure
            # and is the y-dimension of the heat map
            y_index(j, y[j])
            Z[i, j] = data_func(report_states, c, P)
            # Z[i, j] = exp_sec_cost(report_states, c, P)
    return X, Y, Z


def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name=name,
                                             colors=[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


def plot_data(X, Y, Z):
    custom_map = custom_div_cmap(11, mincol='g', midcol='0.9', maxcol='CornflowerBlue')
    plt.pcolormesh(X, Y, Z, cmap=custom_map, shading='auto')
    plt.axis([X[0, 0], X[ax_len - 1, ax_len - 1], Y[0, 0], Y[ax_len - 1, ax_len - 1]])
    plt.colorbar()
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def demo(P: dict[int, Probability]):
    print(P[0].prob_is_insecure("S"))
    print(get_equilibrium_flow("SI", Security_Costs, P))
    print(P[0].prob_is_insecure("S"))
    print(get_equilibrium_flow("SI", Security_Costs, P))
    print(exp_latency(Report_States, Security_Costs, P))
    print(exp_sec_cost(Report_States, Security_Costs, P))


def set_up_latency():
    # Create plot labels
    global plot_title, x_label, y_label
    plot_title = "Expected Network Latency"
    x_label = r'$p_{S_{1}}$'
    y_label = r'$p_{I_{1}}$'

    return exp_latency, set_p_s, set_p_i


def set_up_security():
    # Create plot labels
    global plot_title, x_label, y_label
    plot_title = "Expected Security Cost"
    x_label = r'$p_{S_{1}}$'
    y_label = r'$p_{I_{1}}$'

    return exp_sec_cost, set_p_s, set_p_i


def do_heat_map(P: dict[int, Probability]):
    # Imitate the following with a custom function of your own to output to the heat map
    output = set_up_latency()
    # output = set_up_security()

    x, y, z = get_data(Report_States, Security_Costs, P, *output)
    plot_data(x, y, z)


def do_br_plot(P: dict[int, Probability]):
    x = np.linspace(axis_lims['x']['start'], axis_lims['x']['stop'],  ax_len)
    br = np.empty((2, ax_len), float)
    for i in range(ax_len):
        for j in range(2):
            # e.g., j is 0, so the opponent is 1. This means we must fix our own reporting, which, in this context
            # (see the above explanation of Report_Schedule), means that we must
            opponent = (j + 1) % 2
            # Set the 'opponent' report schedule to the x-axis value
            Report_Schedule[j]['S'] = x[i]
            # Find current player's best response to this updated report schedule value by varying the reporting on
            # the opponent and trying to minimize their traffic
            br[j, i] = best_response_to(opponent, P)

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, br[0], color='blue', label=r'$B_0(p_{S_1})$')
    ax.plot(x, br[1], color='red', label=r'$B_1(p_{S_0})$')
    plt.xlabel(r'$p_S$')
    plt.title("Best Response")
    plt.legend()
    plt.show()


def best_response_to(path: int, P: dict[int, Probability]):
    responses = np.linspace(0.5, 1.0, ax_len)
    traffic = np.empty(ax_len)
    for i in range(ax_len):
        Report_Schedule[path]['S'] = responses[i]
        exp_flow = 0.0
        for report in Report_States:
            flow = get_equilibrium_flow(report, Security_Costs, P)
            exp_flow += prob_of_report(report, P) * flow[path]
        traffic[i] = exp_flow
    # We want to minimize the traffic on our opponent's path
    best_response = np.argmin(traffic - [0.00000001*i for i in range(len(traffic))]) # when best reponse is ambiguous, pick the most informative signal
    return responses[best_response]


def main():
    do_demo = False
    # print("hello, world")
    P = dict()
    for i in range(2):
        P[i] = Probability(Q[i], Report_Schedule[i])
    if do_demo:
        demo(P)
    # do_heat_map(P)
    do_br_plot(P)


if __name__ == '__main__':
    main()
