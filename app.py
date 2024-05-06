import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import xmltodict

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("hams2/fullserv")
    return tokenizer,model

tokenizer,model = get_model()
st.title('Full Service')
trade_msg = st.text_area('Trade Message')
sgw_op = st.text_area('SGW Operation')
button = st.button("Predict")

if (trade_msg and sgw_op) and button:
    trade_msg1 = trade_msg.replace('\n','')
    sgw_op1 = sgw_op.replace('\n','')
    input_seq = trade_msg1+' '+sgw_op1
    pattern = r'[<\"/]'
    cleaned_seq1 = re.sub(pattern, '', input_seq)
    seq = cleaned_seq1.replace('>',' ')
    model.eval()
    input_ids = tokenizer.encode(seq, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    output_sequence = decoded_output[len(seq):].strip()
    trade_msg_dict = xmltodict.parse(trade_msg)
    output_sequence_dict = xmltodict.parse(output_sequence)
    match_id_value1 = trade_msg_dict['TrdCaptRpt']['@MtchID']
    match_id_value2 = output_sequence_dict['TrdCaptRpt']['@MtchID']
    if match_id_value1 != match_id_value2:
        output_sequence_dict['TrdCaptRpt']['@MtchID'] = match_id_value1
    trd_id_value1 = trade_msg_dict['TrdCaptRpt']['@TrdID']
    trd_id_value2 = output_sequence_dict['TrdCaptRpt']['@TrdID']
    if trd_id_value1 != trd_id_value2:
        output_sequence_dict['TrdCaptRpt']['@TrdID'] = trd_id_value1
    modified_ccp_message = xmltodict.unparse(output_sequence_dict)
    if modified_ccp_message.startswith("<?xml"):
        modified_ccp_message = modified_ccp_message.split("?>", 1)[1].strip()
    modified_ccp_message = modified_ccp_message.replace("\n", "").replace("\t", "")
    st.write("CCP Message: ",modified_ccp_message)
