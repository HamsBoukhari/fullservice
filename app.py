# Importing necessary libraries
import streamlit as st  # Streamlit for creating a web-based interface
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Libraries for tokenizing and GPT-2 model
import torch  # PyTorch for machine learning model operations
import re  # Regular expressions for pattern matching and text manipulation
import xmltodict  # For parsing and un-parsing XML data

# This function initializes and caches the tokenizer and model, allowing mutations to the cached data.
@st.cache(allow_output_mutation=True)
def get_model():
    # Load the tokenizer for the distilgpt2 model
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    # Set pad_token to be the end-of-sequence (EOS) token
    tokenizer.pad_token = tokenizer.eos_token
    # Load a pre-trained GPT2 model from a specific path (hams2/fullserv)
    model = GPT2LMHeadModel.from_pretrained("hams2/fullserv")
    return tokenizer, model

# Retrieve the cached tokenizer and model
tokenizer, model = get_model()

# Title for the Streamlit app
st.title('Full Service')

# User input areas for trade message and SGW operation 
trade_msg = st.text_area('Trade Message')  # Text area for trade message input
sgw_op = st.text_area('SGW Operation')  # Text area for SGW operation input
# Button to trigger the prediction
button = st.button("Predict")

# Check if both text areas have input and if the button was pressed
if (trade_msg and sgw_op) and button:
    try:
        # Replace newlines in the inputs with empty strings
        trade_msg1 = trade_msg.replace('\n', '')
        sgw_op1 = sgw_op.replace('\n', '')
        # Concatenate the two cleaned strings into a single sequence
        input_seq = trade_msg1 + ' ' + sgw_op1
        
        # Remove certain special characters (<, \ ,", /) from the sequence
        pattern = r'[<\"/]'
        cleaned_seq1 = re.sub(pattern, '', input_seq)
        # Replace ">" with a space
        seq = cleaned_seq1.replace('>', ' ')
        
        # Put the model into evaluation mode
        model.eval()
        
        # Encode the sequence into input IDs for the pre-trained GPT-2 model
        input_ids = tokenizer.encode(seq, return_tensors='pt')
        
        # Generate output from the pre-trained GPT-2 model, setting a maximum length and specifying padding
        output = model.generate(
            input_ids, 
            max_length=1000, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id  # Pad with the EOS token
        )
        
        # Decode the generated output into text, skipping special tokens
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the original input sequence from the generated output
        output_sequence = decoded_output[len(seq):].strip()
        
        # Parse the trade message and the generated output into dictionaries using xmltodict
        trade_msg_dict = xmltodict.parse(trade_msg)  # Original trade message
        output_sequence_dict = xmltodict.parse(output_sequence)  # Generated output
        
        # Extract and compare specific fields from the trade message and the generated output
        # If the 'MtchID' values differ, update the generated output to match the original
        match_id_value1 = trade_msg_dict['TrdCaptRpt']['@MtchID']
        match_id_value2 = output_sequence_dict['TrdCaptRpt']['@MtchID']
        if match_id_value1 != match_id_value2:
            output_sequence_dict['TrdCaptRpt']['@MtchID'] = match_id_value1
        
        # If the 'TrdID' values differ, update the generated output to match the original
        trd_id_value1 = trade_msg_dict['TrdCaptRpt']['@TrdID']
        trd_id_value2 = output_sequence_dict['TrdCaptRpt']['@TrdID']
        if trd_id_value1 != trd_id_value2:
            output_sequence_dict['TrdCaptRpt']['@TrdID'] = trd_id_value2
        
        # Convert the updated dictionary back into XML format
        modified_ccp_message = xmltodict.unparse(output_sequence_dict)
        
    # If the XML message starts with an XML declaration, remove it
        if modified_ccp_message.startswith("<?xml"):
             modified_ccp_message = modified_ccp_message.split("?>", 1)[1].strip()
        
        # Remove unnecessary newline and tab characters from the XML message
        modified_ccp_message = modified_ccp_message.replace("\n", "").replace("\t", "")
        
        # Display the final modified message in the Streamlit app
        st.write("CCP Message: ", modified_ccp_message)  # Output the final message to the user
        
    except Exception as e:
        # If an error occurs, print an error message
        st.write("There is an error in generating CCP Message. Please verify the Trade Message and the SGW Operation.")
