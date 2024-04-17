import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re
import xml.etree.ElementTree as ET 

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("hams2/fullserv")
    return tokenizer,model

tokenizer,model = get_model()
trade_msg = st.text_area('Trade Message')
sgw_op = st.text_area('SGW Operation')
button = st.button("Predict")

if (trade_msg and sgw_op) and button:
    trade_msg1 = trade_msg.replace('/n','')
    sgw_op1 = sgw_op.replace('/n','')
    input_seq = trade_msg1+' '+sgw_op1
    pattern = r'[<\"/]'
    cleaned_seq1 = re.sub(pattern, '', input_seq)
    seq = cleaned_seq1.replace('>',' ')
    model.eval()
    input_ids = tokenizer.encode(seq, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    output_sequence = decoded_output[len(seq):].strip()
    root1 = ET.fromstring(trade_msg)
    root2 = ET.fromstring(output_sequence)
    match_id_value1 = root1.attrib.get('MtchID')
    match_id_value2 = root2.attrib.get('MtchID')
    if match_id_value1 != match_id_value2:
       root2.set('MtchID', match_id_value1)
    trd_id_value1 = root1.attrib.get('TrdID')
    trd_id_value2 = root2.attrib.get('TrdID')
    if trd_id_value1 != trd_id_value2:
       root2.set('TrdID', trd_id_value1)
    root3 = ET.fromstring(sgw_op)
    pty_id_value = 0
    for pty_element in root3.findall(".//Pty[@R='24']"):
       pty_id_value = pty_element.attrib.get('ID')
    if pty_id_value != 0:
       for pty_element in root2.findall(".//Pty[@R='24']"):
          pty_element.attrib['ID'] = pty_id_value
    modified_ccp_message = ET.tostring(root2, encoding='unicode')
    st.write("CCP Message: ",modified_ccp_message)
