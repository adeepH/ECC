import streamlit as st
from PIL import Image
import subprocess
#import SessionState

st.set_page_config(page_title="Banking System", page_icon=":money_with_wings:")

 
page_bg = """
<style>
body {
background-image: linear-gradient(to bottom, #ffffff, #cccccc);
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
 
st.markdown("""
    <style>
        /* CSS styles for the login form */
        form {
            max-width: 300px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }
        input[type=text], input[type=password] {
            width: 100%;
            padding: 12px 20px;
            margin: 8px 0;
            display: inline-block;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button[type=submit] {
            background-color: #4CAF50;
            color: white;
            padding: 14px 20px;
            margin: 8px 0;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .logo {
            display: flex;
            align-items: center;
        }
        .logo img {
            width: 70px;
            margin-right: 10px;
        }
        .logo h1 {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            margin: 0;
            padding: 0;
        }
        .header {
            background-color: #F2F2F2;
            display: flex;
            align-items: center;
            padding: 10px;
            margin-bottom: 20px;
        }
        .header p {
            margin: 0;
            padding: 0;
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# Define the logo and header text
logo = '<h1>Welcome to Velocity Banking</h1>'


header_text = '<p>Login to Your Account</p>'

#@st.cache_resource()
def login():
    # Display the header and login form
    st.write(f'<div class="header">{logo}</div>', unsafe_allow_html=True)
    st.empty()
    st.write(f'<div class="header">{header_text}</div>', unsafe_allow_html=True)
    form = st.form(key='login-form')
    username = form.text_input('Username')
    password = form.text_input('Password', type='password')
    submit_button = form.form_submit_button(label='Log In')
    
    return username, password, submit_button
# Define user credentials
users = {
    'Adeep': 'AdeepH',
    'Dene': 'DeneA',
    'Kumar': 'KumarV'
}

# Check if the provided username and password match a valid user
def authenticate(username, password):
    if username in users and users[username] == password:
        st.write('xcvx')
        return True
    else:
        return st.error('Incorrect username or password')
    
if __name__ == "__main__":

    username, password, button = login()

    st.session_state['username'] = username
    st.session_state['password'] = password 
    st.session_state['user_dict'] = users
     
    # If authentication is successful, redirect to the home page
    if st.session_state.username in users and st.session_state.password in users[st.session_state.username]:
         
        #if authenticate(username, password):
            #st.write(authenticate(username, password))
        st.success('Logged in as {}'.format(username))
        st.write('Head over to the infer page to get fraudulent activity detection')
            #st.experimental_rerun()
            #st.stop()
    elif st.session_state.username == "" or st.session_state.password == "":
        st.stop()
    else:
        st.warning('please login to continue')

