import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_chat import message
from streamlit_tags import st_tags_sidebar
import openai
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
from time import sleep
from keyword_to_image import fetch_image_url,api_key
import coAIdeator_backend as backend

if 'tags_desplayed' not in st.session_state:
    st.session_state.tags_displayed =False
if 'keyword_concepts' not in st.session_state:
    st.session_state.keyword_concepts =[]
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if "input" not in st.session_state:
    st.session_state['input'] = [] 
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "gpt_prompt" not in st.session_state:
     st.session_state.gpt_prompt = None
if 'nodes' not in st.session_state:
    st.session_state.nodes = []
if 'edges' not in st.session_state:
    st.session_state.edges = []
if 'clicked_node' not in st.session_state:
     st.session_state.clicked_node = None
if 'comp1' not in st.session_state:
    st.session_state.comp1 = None
if 'comp2' not in st.session_state:
    st.session_state.comp2 = None
if 'connect1' not in st.session_state:
    st.session_state.connect1 = None
if 'connect2' not in st.session_state:
    st.session_state.connect2 = None
if 'selected_button' not in st.session_state:
        st.session_state.selected_button = 'None'
if 'editable_textbox' not in st.session_state:
        st.session_state.editable_textbox = None
if "current_design_activity" not in st.session_state:
    st.session_state.current_design_activity = None
if "functional_decomposition_node" not in st.session_state:
    st.session_state.functional_decomposition_node = None

def ischild():
    ischild = False
    for edge in edges:
        if(((edge.source == st.session_state.connect1) and (edge.to == st.session_state.connect2)) or ((edge.source == st.session_state.connect2) and (edge.to == st.session_state.connect1))):
            ischild =True
    return ischild

def render_buttons():
    # Function to handle the button logic
    def handle_button_click(button_name):
        st.session_state.selected_button = button_name

    # Check if the selected_button state variable exists, if not, create it

    # Create two rows with 6 columns each for the buttons
    
    button_names = ['Generate','Explore','Compare','Critique','SCAMPER','Brainstorm','Functional Decomposition']

    # row1[0].button('Generate',on_click=generate_button)
    # Display buttons in the first row
    st.sidebar.write("**Design Activities**")
    row1 = st.sidebar.columns(4)

    for index, button_name in enumerate(button_names[:4]):
        if row1[index].button(button_name):
            handle_button_click(button_name)
    st.sidebar.write("**Design methods**")
    row2 =st.sidebar.columns(3)
    # Display buttons in the second row
    for index, button_name in enumerate(button_names[4:]):
        if row2[index].button(button_name):
            handle_button_click(button_name)

config = Config(width=800,
                height=900,
                directed=False, 
                physics=False, 
                hierarchical=False,
                nodeSpacing = 150,
                parentCentralization = False,
                direction = "LR"
    
                # **kwargs
                 
                )   
API_key = ""
# add your api key here
llm = ChatOpenAI(temperature=0,
                openai_api_key=API_key, 
                model_name='gpt-3.5-turbo', 
                verbose=False)

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=7 )

nodes = st.session_state.nodes
edges = st.session_state.edges

def prompt_creator(keyword_concept, constraints):
    # Placeholder for your backend function
    # You can replace this with your actual implementation\
    if (st.session_state.selected_button == "SCAMPER"):
        st.session_state.editable_textbox = f"I want to apply SCAMPER to the concept of {keyword_concept} "
        st.session_state.selected_button = None
        st.session_state.current_design_activity = "SCAMPER"

    if((st.session_state.selected_button == "Generate")):
        if(constraints ==""):
            st.session_state.editable_textbox = f"Can you list more ideas related to the concept of {keyword_concept}"
        else:
            st.session_state.editable_textbox = f"Can you list more ideas related to the concept of {keyword_concept} subject to the following customer requirements.\n{constraints}"
        st.session_state.selected_button = None
        st.session_state.current_design_activity = "Generate"
    if((st.session_state.selected_button == "custom")):
        st.session_state.selected_button = None
        st.session_state.current_design_activity = "custom"
    

def generate_response(prompt):
    if prompt:		
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=[{"role": "user", "content": prompt}],max_tokens=193,temperature=0.5,)
        response_gpt = response['choices'][0]['message']['content']		
    else:
        response_gpt = "Please give some input text"
    return response_gpt 


def node_to_add_id_check(node_to_add):
    is_node_present =False
    for node in nodes:
        if node.id == node_to_add.id:
                is_node_present =True
    return is_node_present      
def append_node_if_not_exists(node_to_add):
    # print(node_to_add.id)
    if len(nodes) == 0:
          st.session_state.nodes.append(node_to_add)
    elif(node_to_add_id_check(node_to_add) == False):
        #   print(f'{node_to_add} is not in nodes')
          st.session_state.nodes.append(node_to_add)
    else:
        # print('in else')
        pass

def edge_to_add_source_target_check(edge_to_add):
    is_edge_present =False
    for edge in edges:
        if ((edge_to_add.source == edge.source) and (edge_to_add.to == edge.to)):
                is_edge_present =True
    return is_edge_present        
def append_edge_if_not_exists(edge_to_add):
    # print(edge_to_add.source,edge_to_add.to)
    if len(edges) == 0:
          st.session_state.edges.append(edge_to_add)
    elif(edge_to_add_source_target_check(edge_to_add) == False):
        #   print(f'{node_to_add} is not in nodes')
          st.session_state.edges.append(edge_to_add)
    else:
        # print('in else')
        pass

def generation_function():
    print("Running the generation function")
    st.info(f"You selected generation on the node {st.session_state.clicked_node}")
    # Add your generation-specific logic here
    # For example: st.write("This is the generation option.")
    
    if st.button('generate on node'):
        print('changing the session state value')
        st.session_state.editable_textbox = f"Can you generate more ideas related to {st.session_state.clicked_node}"	
        # generate_nodes_for_concepts(st.session_state.clicked_node,['magnetic maze game','marble roller coaster','build it yourself robot kit','interactive musical instrument','balance challenge game','mechanical puzzle box','diy rube goldberg machine kit','kinetic sand playset','steampunk themed automation','pull back race cars'])

def functional_decomposition_function():
    st.info(f"You selected functional decomposition on {st.session_state.clicked_node}")
    if(st.button("decompose")):
        st.session_state.sidebar_input = f"Can you perform functional decomposition on {st.session_state.clicked_node}"


def comparison_function():
    st.info("You selected 'comparison'")
    if((st.session_state.comp1 != None) and (st.session_state.comp2 != None)):
        st.session_state.comp1 =None
        st.session_state.comp2 =None
        
    if(st.session_state.comp1 == None):
        st.session_state.comp1 = st.session_state.clicked_node
    st.write('What would you like to compare it to ?')
    if((st.session_state.comp2 == None) and (st.session_state.clicked_node != st.session_state.comp1)):
            st.session_state.comp2 = st.session_state.clicked_node

    st.write(f'comp1 is {st.session_state.comp1} and comp2 is {st.session_state.comp2}')

       
    # Add your comparison-specific logic here
    # For example: st.write("This is the comparison option.")

def critiquing_function():
    st.info("You selected 'critiquing'")
    # Add your critiquing-specific logic here
    # For example: st.write("This is the critiquing option.")

def space_exploration_function():
     st.info("You selected 'space exploration'")
    # Add your space exploration-specific logic here
    # For example: st.write("This is the space exploration option.")

def manual_connection_function():
    if((st.session_state.connect1 != None) and (st.session_state.connect2 != None)):
        st.session_state.connect1 =None
        st.session_state.connect2 =None
    # Capture the id's of the 2 nodes to connect
    st.info("You selected 'manual_connection. Please select the 2 nodes to be connected")
    if(st.session_state.connect1 == None):
        st.session_state.connect1 = st.session_state.clicked_node
        st.write('What would you like to compare it to ?')
    if((st.session_state.connect2 == None) and (st.session_state.clicked_node != st.session_state.connect1)):
        st.session_state.connect2 = st.session_state.clicked_node
    st.write(f'You have selected that you want to connect {st.session_state.connect1} and  {st.session_state.connect2}')
    # Create a corresponding edge
    if(st.button('Connect')):
        if(ischild()):
            pass
        else:
            append_edge_if_not_exists(Edge(source=st.session_state.connect1, 
            target=st.session_state.connect2, 
            # **kwargs
            ) )
        # Add the edge to the graph
    
def GraphGenerator_ConceptPairs(parentnode, childnode,childnode_info):
    # Generate concept map by parent and child node pairs.
    if(st.session_state.current_design_activity == "SCAMPER"):
        parentnode_exist = 0

        parent_label = parentnode
        child_label = childnode

        for node in nodes:
            if node.label == parentnode:
                # dumy_node = node
                parentnode_exist = 1
                break

        if parentnode_exist == 0:
        # if parent node not exist, generate parent node

            append_node_if_not_exists(Node(id=parent_label, 
            label=parent_label, 
            size=40,
            shape="circularImage",
            # The circularImage, the differen shape should represent different meaning
            image = fetch_image_url(parent_label,api_key)
            ))
            
        # sleep(1)
        # if dumy_node is None:
        #     raise ValueError(f"Base node with label '{parent_label}' not found.")



        # Add childnode
        append_node_if_not_exists(Node(id=childnode,
                        target=childnode, 
                        size=30,
                        shape="circularImage",
                        image = fetch_image_url(childnode,api_key)
                        ))
        # sleep(1)
        # Add edge
        append_edge_if_not_exists(Edge(source=parent_label,  
                        target=child_label))
    elif(st.session_state.current_design_activity == "functional decomposition"):
        parentnode_exist = 0

        parent_label = parentnode
        child_label = childnode

        for node in nodes:
            if node.id == parentnode:
                # dumy_node = node
                parentnode_exist = 1
                break

        if parentnode_exist == 0:
        # if parent node not exist, generate parent node
        # In this case, it show up a caution that the user generate a new idea. This should be included in our 
        # design space
            append_node_if_not_exists(Node(id=parent_label, 
            label=parent_label, 
            size=40,
            font = {'color': '#343434',
                            'size': 18,
                            'face': 'helvetica',
                            'background': "white",
                            'strokeWidth': 0,
                            'strokeColor': '#ffffff',
                            'align': 'center',
                            'vadjust': 0,
                            'multi': False},
            shape="circularImage",
            # The circularImage, the differen shape should represent different meaning
            image = fetch_image_url(parent_label,api_key)
            ))
        # Add childnode
        append_node_if_not_exists(Node(id=childnode,
                        label=childnode,
                        title = f"{childnode}:\n{childnode_info}",
                        borderWidth=8,
                           color = "#3FD6D8",
                        font = {'color': '#343434',
                            'size': 16,
                            'face': 'arial',
                            'background': "white",
                            'strokeWidth': 0,
                            'strokeColor': '#ffffff',
                            'align': 'center',
                            'vadjust': 0,
                            'multi': False},
                        size=30,
                        shape="circularImage",
                        image = fetch_image_url(childnode,api_key)
                        ))
        # sleep(1)
        # Add edge
        append_edge_if_not_exists(Edge(source=parent_label,  
                          width = 3,
                        color="#3FD6D8",
                        target=child_label))
    elif(st.session_state.current_design_activity == "implementation methods"):
        parentnode_exist = 0

        parent_label = parentnode
        child_label = childnode

        for node in nodes:
            if node.id == parentnode:
                # dumy_node = node
                parentnode_exist = 1
                break

        if parentnode_exist == 0:
        # if parent node not exist, generate parent node
        # In this case, it show up a caution that the user generate a new idea. This should be included in our 
        # design space
            append_node_if_not_exists(Node(id=parent_label, 
            label=parent_label, 
            size=40,
            font = {'color': '#343434',
                            'size': 18,
                            'face': 'helvetica',
                            'background': "white",
                            'strokeWidth': 0,
                            'strokeColor': '#ffffff',
                            'align': 'center',
                            'vadjust': 0,
                            'multi': False},
            shape="circularImage",
            # The circularImage, the differen shape should represent different meaning
            image = fetch_image_url(parent_label,api_key)
            ))
        # Add childnode
        append_node_if_not_exists(Node(id=childnode,
                        label=childnode,
                        title = f"{childnode}:\n{childnode_info}",
                        borderWidth=8,
                           color = "#2ded33",
                        font = {'color': '#343434',
                            'size': 16,
                            'face': 'arial',
                            'background': "white",
                            'strokeWidth': 0,
                            'strokeColor': '#ffffff',
                            'align': 'center',
                            'vadjust': 0,
                            'multi': False},
                        size=30,
                        shape="circularImage",
                        image = fetch_image_url(childnode,api_key)
                        ))
        # sleep(1)
        # Add edge
        append_edge_if_not_exists(Edge(source=parent_label,  
                          width = 3,
                        color="#2ded33",
                        target=child_label))
    else:
        parentnode_exist = 0

        parent_label = parentnode
        child_label = childnode

        for node in nodes:
            if node.id == parentnode:
                # dumy_node = node
                parentnode_exist = 1
                break

        if parentnode_exist == 0:
        # if parent node not exist, generate parent node
        # In this case, it show up a caution that the user generate a new idea. This should be included in our 
        # design space
            append_node_if_not_exists(Node(id=parent_label, 
            label=parent_label, 
            size=40,
            font = {'color': '#343434',
                            'size': 18,
                            'face': 'helvetica',
                            'background': "white",
                            'strokeWidth': 0,
                            'strokeColor': '#ffffff',
                            'align': 'center',
                            'vadjust': 0,
                            'multi': False},
            shape="circularImage",
            # The circularImage, the differen shape should represent different meaning
            image = fetch_image_url(parent_label,api_key)
            ))
            
        # Add childnode
        append_node_if_not_exists(Node(id=childnode,
                        label=childnode,
                        title = f"{childnode}:\n{childnode_info}",
                        borderWidth=8,
                        color = "#FE5212",
                        font = {'color': '#343434',
                            'size': 16,
                            'face': 'arial',
                            'background': "white",
                            'strokeWidth': 0,
                            'strokeColor': '#ffffff',
                            'align': 'center',
                            'vadjust': 0,
                            'multi': False},
                        size=30,
                        shape="circularImage",
                        image = fetch_image_url(childnode,api_key)
                        ))
        # sleep(1)
        # Add edge
        append_edge_if_not_exists(Edge(source=parent_label,  
                          width = 3,
                           color="#FE5212",
                        target=child_label))

def GraphGenerator_ConceptMap(concept_pairs,keyword_info_pairs):

    for i in range(len(concept_pairs)):
        pair = concept_pairs[i]
        parent_concept = pair[0]
        child_concept = pair[1]
        if(keyword_info_pairs!=[]):
            child_concept_info = keyword_info_pairs[i][1]
        print('call GraphGenerator_ConceptMap')
        GraphGenerator_ConceptPairs(parent_concept,child_concept,child_concept_info)

def comparedropdown_callback():
    st.session_state.editable_textbox = f"Can you compare the two concepts {st.session_state.comp1} and {st.session_state.comp2}"
    st.session_state.current_design_activity = "compare"
def generate_dropdown_callback():
    st.session_state.editable_textbox = f"Can you generate more ideas related to {st.session_state.clicked_node}"
    st.session_state.current_design_activity = 'generation'
def funcional_decomposition_dropdown_callback():
    st.session_state.editable_textbox = f"Can you perform functional decomposition on {st.session_state.clicked_node} and list the corresponding functions ?"
    st.session_state.current_design_activity = 'functional decomposition'
    st.session_state.functional_decomposition_node = st.session_state.clicked_node
 
def find_parent(child_node_id):
    for edge in edges:
        if (edge.to == child_node_id):
            return edge.source

               
# def generate_button():
# 	st.session_state.editable_textbox = f"Can you generate more ideas related to {st.session_state.clicked_node}"
# 	st.session_state.selected_button = None
 
def main():
    
    st.set_page_config(layout="wide")
    st.title("Co-AIdeator system")

    st.sidebar.title("**Sidebar**")
    render_buttons()
    st.sidebar.title("Design Goals")
    keyword_concept = st.sidebar.text_input("**Please enter your keyword concept:**")

    constraints = st.sidebar.text_input("**Please enter the customer requirements**")
    font_size = 20  # you can adjust this as needed
    st.sidebar.markdown(f"<div style='font-size: {font_size}px;font-weight: bold; '>Prompt Suggestions</div>", unsafe_allow_html=True)
    st.sidebar.write("")
    prompt_creator(keyword_concept, constraints)
    prompt_editable = st.sidebar.text_area("**Generated Prompt:**", st.session_state.editable_textbox)
    st.session_state.gpt_prompt = prompt_editable    
    if((keyword_concept != "") and (st.session_state.keyword_concepts == [])):
        st.session_state.keyword_concepts.append(keyword_concept)
    if((keyword_concept != "") and (keyword_concept not in st.session_state.keyword_concepts)):
        st.session_state.keyword_concepts.append(keyword_concept)
    st.sidebar.write(
        f"<script>document.getElementById('keyword_input').value = '';</script>",
        unsafe_allow_html=True
    )
        # Create the ConversationChain object with the specified configuration

    Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
    )
        
# applying scamper
    row3 = st.sidebar.columns(2)
    if ((row3[0].button('Generate Response'))):
        if((st.session_state.current_design_activity == 'custom') or (st.session_state.current_design_activity == 'compare') or (st.session_state.current_design_activity == 'critique')):
            input_text = st.session_state.gpt_prompt
            output = Conversation.run(input=input_text)
            st.session_state.past.append(input_text)
            st.session_state.generated.append(output)
        else:
            input_text = st.session_state.gpt_prompt
            output = Conversation.run(input=input_text)
            st.session_state.past.append(input_text)
            st.session_state.generated.append(output)
            if(st.session_state.clicked_node == None):
                print('I need to run this part of the code')
                test_conversation = backend.Converastion(input_text,output,st.session_state.current_design_activity,keyword_concept)
            else:
                print('I am now in this part of the code, design activity is not generate')
                test_conversation = backend.Converastion(input_text,output,st.session_state.current_design_activity,st.session_state.clicked_node)
            concept_pairs,keyword_info_pairs = test_conversation.Get_ConceptPairs()
            print(f"concept pairs is {concept_pairs}")
            GraphGenerator_ConceptMap(concept_pairs,keyword_info_pairs)
    row3[1].button('Download Map')
    st.sidebar.title("Design History and Gallery")
    
    
    st.session_state.tags_displayed = True
    st_tags_sidebar(
    label='**List of Keyword concepts**',
    text='Press enter to add more',
    value=['action toys'],
    maxtags = 4,
    key='2')
    st_tags_sidebar(
    label='**User selected nodes**',
    text='Press enter to add more',
    value=['action toys'],
    maxtags = 4,
    key='3')
    st_tags_sidebar(
        label='**User explored nodes**',
        text='Press enter to add more',
        value=['action toys'],
        maxtags = 4,
        key='4')
    
    if st.session_state['generated']:
        with st.sidebar:
            st.sidebar.title("Information History")
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i])
                message(st.session_state['past'][i], is_user=True)




    col1,col2 = st.columns(spec=[0.75,0.25])
 
 
    with col1:
        print("running column1")
        
        st.session_state.clicked_node = agraph(nodes=st.session_state.nodes, edges=st.session_state.edges, config=config)
        append_node_if_not_exists(Node(id = "Tricycles",label = "Tricycles",color="#FE5212",shape = "circularImage",borderWidth=8,size = 30,image = fetch_image_url("Tricycle",api_key)))
        clicked_node = st.session_state.clicked_node
        if clicked_node:
            with col2:
            # Define the options for the dropdown menu
                dropdown_options = ['generation','functional decomposition', 'comparison', 'critiquing','Manual Connection','explore implementation methods for function']

            # Create the dropdown menu
                selected_option = st.selectbox("Select an option:", dropdown_options)

            

                if(selected_option == "generation"):
                    st.info(f"You selected generation on the node {st.session_state.clicked_node}")
                    st.button('generate on node',on_click=generate_dropdown_callback)
                if(selected_option == "functional decomposition"):
                    st.info(f"You functional decomposition on the node {st.session_state.clicked_node}")
                    st.button('decompose',on_click=funcional_decomposition_dropdown_callback)
                
                if(selected_option == "comparison"):
                    st.info("You selected 'comparison'")
                    if(st.session_state.comp1 ==None):
                        st.session_state.comp1= clicked_node
                    if((clicked_node != st.session_state.comp1)):
                        st.session_state.comp2 = clicked_node
                        st.session_state.comp1 = None

                    
                    st.write('What would you like to compare it to ?')
                    
                    st.write(f'comp1 is {st.session_state.comp1} and comp2 is {st.session_state.comp2}')
     
                    st.button('Compare concepts',on_click=comparedropdown_callback)
                    
                if(selected_option == "critiquing"):
                    st.info(f"Can you describe the advantages and disadvantages of {st.session_state.clicked_node}")
                    st.session_state.editable_textbox = f"Can you critique the concept {st.session_state.clicked_node}"
                    st.session_state.current_design_activity = "critique"

                if(selected_option == 'explore implementation methods for function'):
                    st.info(f"List five ways in which the {st.session_state.clicked_node} function can be implemented within the {find_parent(st.session_state.clicked_node)} concept. ")
                    st.session_state.editable_textbox = f"List five ways in which the {st.session_state.clicked_node} function can be implemented within the {find_parent(st.session_state.clicked_node)} concept. "
                    st.session_state.current_design_activity = "implementation methods"
                if(selected_option == 'Manual Connection'):
                    manual_connection_function()
                for node in nodes:
                    if(clicked_node == node.id):
                        st.write(node.title)
    
if __name__ == "__main__":
    main()





# append_node_if_not_exists(Node(id="parent_label", 
        # 			label="test", 
        # 			size=40,
         # 			color= '#6DDC2D',
        # 			shape="circularImage",
        # 			borderWidth=4,
        # 			# The circularImage, the differen shape should represent different meaning
        # 			image = fetch_image_url("apple",api_key)
        # 			))