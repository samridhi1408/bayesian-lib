class Node:
    def __init__(self, name, parents, prob_table):
        self.name = name
        self.parents = parents
        self.prob_table = prob_table

    def get_prob(self, value, parents_states):
        prob_true = self.prob_table[parents_states]
        if value == True:
            return prob_true
        else:
            return 1.0 - prob_true

#building the network

#no parents
node = Node("Gene", [], {
    () :0.10 
})

#Biomarker (Parent: Gene)
biomarker_node = Node("Biomarker", ["Gene"], {
    (True,): 0.80,
    (False,): 0.05
})

# Disease (Parent: Biomarker)
disease_node = Node("Disease", ["Biomarker"], {
    (True,): 0.90,   
    (False,): 0.10  
}) 

network = {
    "Gene": node,
    "Biomarker": biomarker_node,
    "Disease": disease_node
}

def calc_joint_prob(scenario, network):
    total_prob = 1.0

    for name, state in scenario.items():
        node = network[name]

        parent_states_list = []
        for parent in node.parents:
            parent_states_list.append(scenario[parent])

        parent_states = tuple(parent_states_list)

        p = node.get_prob(state, parent_states)

        total_prob *= p

    return total_prob

patient_a = {"Gene": True, "Biomarker": True, "Disease": True}
prob_a = calc_joint_prob(patient_a, network)

print(f"Probability of Patient A's scenario: {prob_a:.4f} (or {prob_a * 100:.2f}%)")