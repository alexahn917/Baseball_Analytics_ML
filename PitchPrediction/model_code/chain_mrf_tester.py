######################################################################
# DO NOT MODIFY THIS FILE. YOU MUST SUBMIT IT WITHOUT CHANGES
######################################################################


import sys
from chain_mrf import ChainMRFPotentials, SumProduct, MaxSum

def find_potentials(data_file):
    p = ChainMRFPotentials(data_file)
    sp = SumProduct(p)
    for i in range(1, p.chain_length() + 1):
        marginal = sp.marginal_probability(i)
        if len(marginal) - 1 != p.num_x_values(): # take off 1 for 0 index which is not used
            raise Exception("length of probability distribution is incorrect: " + str(len(marginal)) + " != " + str(p.num_x_values()))
        print "marginal probability distribution for node " + str(i) + " is:"
        sum_prob = 0.0
        for k in range(1, p.num_x_values() + 1):
            if marginal[k] < 0.0 or marginal[k] > 1.0:
                raise Exception("illegal probability for x_" + str(i))
            print "\tP(x_" + str(i) + " = " + str(k) + ") = " + str(marginal[k])
            sum_prob += marginal[k]
        err = 1e-5
        if sum_prob < 1.0 - err or sum_prob > 1.0 + err:
            raise Exception("marginal probability distribution for x_" + str(i) + " doesn't sum to 1")
    
    ms = MaxSum(p)
    for i in range(1, p.chain_length() + 1):
        max_prob = ms.max_probability(i)
        print "MaxLogProbability=" + str(max_prob)
        assignments = ms.get_assignments()
        for j in range(1, len(assignments)):
            print "\tx_" + str(j) + "=" + str(assignments[j])


def main():
    if len(sys.argv) == 1:
        print "missing MRF potential data file"
        return
    for data_file in sys.argv[1:]:
        print "[main] testing potentials for data file: " + data_file
        find_potentials(data_file)
    return


if __name__ == "__main__":
    main()


######################################################################                                                                       
# DO NOT MODIFY THIS FILE. YOU MUST SUBMIT IT WITHOUT CHANGES                                                                                
###################################################################### 
