# import sys
# import os
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#
# from plot_utils import generate_full_report
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# RESULTS_DIR = os.path.join(BASE_DIR, 'results')
#
# def main():
#     trees_number_file = os.path.join(RESULTS_DIR, "trees_number")
#     epsilon_file = os.path.join(RESULTS_DIR, "epsilon")
#     generate_full_report(trees_number_file, 1, y_range=(0.8, 1.0))
#     generate_full_report(epsilon_file, 1, y_range=(0.8, 1.0))
#
# if __name__ == "__main__":
#     main()
from plot_utils import generate_full_report

generate_full_report("trees_number", 1, y_range=(0.1, 1.0))
generate_full_report("epsilon", 1, y_range=(0.1, 1.0))
generate_full_report("tree_percentage", 1, y_range=(0.1, 1.0))
generate_full_report("max_features", 1, y_range=(0.1, 1.0))