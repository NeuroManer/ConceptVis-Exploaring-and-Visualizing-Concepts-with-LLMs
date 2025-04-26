
import re 
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_chat import message
import openai
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from keyword_to_image import fetch_image_url,api_key
from time import sleep
import re
import numpy as np

class Converastion:
    # Initializer / Instance Attributes
    def __init__(self, user_prompt, gpt_response,activity='SCAMPER',concept_key='ToyCar'):
        # node type, 1 = idea node, 2 = design information nodes

        self.user_prompt = user_prompt
        self.gpt_response = gpt_response
        self.design_activity = activity
        # concept key is the parent node
        self.concept_key = concept_key

        print(self.design_activity)

    def Fetch_Activity(self):
        # recognize the design activities
        if activity is not None:
            return self.design_activity
        else:
            print('No related activity detected')
            activtiy = None
            return self.design_activity


    def Get_GPTResponse(self):
    # Get dictionary {idea:description}
        output_dict = {}

        # Create a regular expression pattern that matches a number, a name, and a description.
        pattern = re.compile(r'(\d+)\.\s+(.+?):\s+(.+)')

        # Find all matches in the input string.
        matches = pattern.findall(self.gpt_response)

        for match in matches:
            # Each match is a tuple of three strings: the number, the name, and the description.
            # We ignore the number and add the name and description to the dictionary.
            _, name, description = match
            output_dict[name] = description

        # print(output_dict)
        return output_dict

    def Get_ConceptPairs(self):

        design_info = self.Get_GPTResponse()
        concept_pairs = []
        keyword_info_pairs = []

        if ((self.design_activity == 'Generate') or (self.design_activity == 'generation')) :
            # print(info)
            parent_node = self.concept_key
            
            for key_word,info in design_info.items():
                concept_pairs.append([parent_node,key_word])
                keyword_info_pairs.append([key_word,info])

            return concept_pairs,keyword_info_pairs

        elif self.design_activity == 'SCAMPER':
            parent_node = self.concept_key
            concept_pairs = []
            for key_word, info in design_info.items():
                # later update it to process the type of nodes using a dictionary instead of []
                # concept_pairs.append({parent_node:'idea',key_word})
                concept_pairs.append([parent_node,key_word])
                concept_pairs.append([key_word,info])
            return concept_pairs,keyword_info_pairs
        elif self.design_activity == 'functional decomposition':
            parent_node = self.concept_key 
            concept_pairs =[]
            print(design_info.items())
            for function,description in design_info.items():
                concept_pairs.append([parent_node,function])
                keyword_info_pairs.append([function,description])
            return concept_pairs,keyword_info_pairs
        elif self.design_activity == 'implementation methods':
            parent_node = self.concept_key 
            concept_pairs =[]
            print(design_info.items())
            for method,description in design_info.items():
                concept_pairs.append([parent_node,method])
                keyword_info_pairs.append([method,description])

            return concept_pairs,keyword_info_pairs
                # concept_pairs.append({parent_node:'idea',key_word})


    def Create_ConceptDict(self):
        data = self.concept_pairs
        def insert_into_tree(tree, parent, child):
            if parent not in tree:
                tree[parent] = {}
            if child not in tree[parent]:
                tree[parent][child] = {}

        tree_dict = {}
        for pair in data:
            insert_into_tree(tree_dict, *pair)

        self.concept_dict = tree_dict

    def Get_Conceptdict():
        return self.concept_dict
        
        # return tree_dict

    # def Genearate_Graph

    def Keyword_extraction():
        # Extract the keyword from user prompt and the GPT response, the keyword include the father nodes, 
        # and the information nodes
        return None

    def MemoryConnection():
        # If there is not obvirous keyword, use the long term memory to get the connection
        # For example, use ask "give me more ideas"
        # For example, I want to compare it with out door toys

        return None
        
def FatherNodeSearch():
	return None
	# Can we use longchain to get the fathers nodes

def ResponseExtraction(gpt_response):
	# Initialize an empty dictionary.
	output_dict = {}

	# Create a regular expression pattern that matches a number, a name, and a description.
	pattern = re.compile(r'(\d+)\.\s+(.+?):\s+(.+)')

	# Find all matches in the input string.
	matches = pattern.findall(gpt_response)

	for match in matches:
		# Each match is a tuple of three strings: the number, the name, and the description.
		# We ignore the number and add the name and description to the dictionary.
		_, name, description = match
		output_dict[name] = description

	# print(output_dict)
	return output_dict

def GetConceptPairs(GPT_response):
	design_info = ResponseExtraction(GPT_response)
	# print(info)
	parent_node = 'ToyCar'
	concept_pairs = []
	for key_word in design_info:
		concept_pairs.append([parent_node,key_word])

	return concept_pairs


# PROCESSING CONVERSATIONS(prompts and reponses) DICTIONARY


def find_depth(mind_map):

    if not isinstance(mind_map, dict) or not mind_map:
        return 0

    max_depth = 0
    for child_node in mind_map.values():
        depth = find_depth(child_node)
        max_depth = max(max_depth, depth)

    return max_depth + 1


def get_node_names(dictionary):
    queue = deque([(dictionary, 0)])  # start with the root at level 0
    levels = {}

    while queue:
        node, level = queue.popleft()  # remove the next node and its level
        if isinstance(node, dict):
            for key, value in node.items():
                if level in levels:
                    levels[level].append(key)
                else:
                    levels[level] = [key]
                queue.append((value, level + 1))  # add children to queue at next level
    result = []
    for level, names in sorted(levels.items()):
        result.append(f'Layer {level}: {names}')
    return result

def find_path(dictionary, target, path=[]):

    for key, value in dictionary.items():
        new_path = path + [key]
        if key == target:
            return new_path
        if isinstance(value, dict):
            result = find_path(value, target, new_path)
            if result:
                return result
    return None

def create_tree(data):
    def insert_into_tree(tree, parent, child):
        if parent not in tree:
            tree[parent] = {}
        if child not in tree[parent]:
            tree[parent][child] = {}

    tree_dict = {}
    for pair in data:
        insert_into_tree(tree_dict, *pair)
        
    return tree_dict

def insert_at_node(tree, path, new_data):
    """
    Insert new_data into tree at a specific path.
    
    Parameters:
    - tree: The current tree (nested dictionary).
    - path: List of keys specifying the path to insertion point.
    - new_data: New dictionary to insert.

    Returns:
    - Modified tree with new_data inserted.
    """
    
    # Navigate the tree according to the given path

    node = tree
    for key in path[:-1]:
        node = node.get(key, {})
    if path[-1] in node:
        node[path[-1]].update(new_data)
    else:
        node[path[-1]] = new_data
    return tree
# # # Example to use the backend
# # # THE FOLLOWING CODE SHOULD BE PLACED IN THE coAIdeator_interface.py

# def GraphGenerator_ConceptPairs(parentnode, childnode):
#     # Generate concept map by parent and child node pairs.

#     parentnode_exist = 0

#     parent_label = parentnode
#     child_label = childnode

#     for node in nodes:
#         if node.label == parentnode:
#             # dumy_node = node
#             parentnode_exist = 1
#             break

#     if parentnode_exist == 0:
#     # if parent node not exist, generate parent node
#     # In this case, it show up a caution that the user generate a new idea. This should be included in our 
#     # design space
#         nodes.append(Node(id=parent_label, 
#                     label=parent_label, 
#                     size=40,
#                     shape="circularImage",
#                     # The circularImage, the differen shape should represent different meaning
#                     image = fetch_image_url(parent_label,api_key)
#                     ))

#     # Add childnode
#     sleep(2)
#     nodes.append(Node(id=childnode, 
#                     label=childnode, 
#                     size=30,
#                     shape="circularImage",
#                     image = fetch_image_url(childnode,api_key)
#                     ))

#     # Add edge
#     edges.append(Edge(source=parent_label, 
                    # label=parent_label+' '+child_label, 
                    # label='',
                    # target=child_label))

# def GraphGenerator_ConceptMap(concept_pairs):

# 	for pair in concept_pairs:
# 		parent_concept = pair[0]
# 		child_concept = pair[1]
# 		print('call GraphGenerator_ConceptMap')
# 		GraphGenerator_ConceptPairs(parent_concept,child_concept)

# def sidebar():
# 	with st.sidebar:
# 		st.title("CoAIdeator")

# 		input_text = user_input_handler()

# 		generate_button = st.button('Generate Response')
# 		# st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# 		if input_text and generate_button:
# 			output = Conversation.run(input=input_text)  
# 			# print(output)
# 			# information = ResponseExtraction(output)
# 			# print('gpt_information',information)


# 			# graph_display_area.empty()
#               concept_pairs = backend.GetConceptPairs(output)   
# 			# GraphGenerator_ConceptMap(concept_pairs)
# 			
			

# test_conversation = backend.Converastion(input_text,output)
# concept_pairs = test_conversation.Get_ConceptPairs()
# GraphGenerator_ConceptMap(concept_pairs)

			
# 			# generate_nodes_for_concepts(generate_graph,generate_graph)
# 			with graph_display_area.container():
# 				agraph(nodes,edges,config)

# 			st.session_state.past.append(input_text)
# 			st.session_state.generated.append(output)
			
# 		if st.session_state['generated']:
			
# 			for i in range(len(st.session_state['generated'])-1, -1, -1):
# 				message(st.session_state["generated"][i], key=str(i))
# 				message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

