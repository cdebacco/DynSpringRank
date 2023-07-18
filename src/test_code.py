import pickle as pkl

with open('./data/output/dsr_A_synt_cos_beta2-0_c0-5_output.pkl','rb') as file:
    results =pkl.load(file)

print(results)

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-s','--save',action='store_true',help='Saves the output of model')
# parser.add_argument('-v','--verbose',action='store_true',help='prints details of internal executions of model')
# args = parser.parse_args()

# if args.verbose:
#     print(args.verbose)
#     print('verbosity is on')
# if args.save:
#     print(args.save)
#     print('output saved')
