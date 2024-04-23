import streamlit as st
import subprocess

st.set_page_config(layout="wide")

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)

html = """
<body>
    <div class="container">
        <div class="left">
            <h1>F E E L I N G O</h1>
            <div class="recomend">
                <h2>Made for you</h2>
                <p><b>Based on your emotion</b></p>
                <button class="btn"><b>Hit the Webcam!</b></button>
                <button class="btn1"><b>Speak Out!</b></button>
            </div>
        </div>
        <div class="right">
            <ul>
                <li><b>About</b></li>
                <li><b>Settings</b></li>
                <li><b>Support</b></li>
                <li><b>Profile</b></li>
            </ul>
        </div>
    </div>
</body>
"""

css = """
body{
    color: white;
}
.right{
    float: right;
    width: 50%;
    height: 90vh;
    background: url(https://tidal.com/_nuxt/img/devices2.c334f8d.png);
    background-size: contain;
    background-color: #202022;
}
.left{
    float: left;
    background-color: #202022;
    width: 50%;
    height: 90vh;
}
.left h1{
    padding: 10% 10%;
    font-family: Arial, Helvetica, sans-serif;
    color: white;
}
.left .search-container {
  float: right;
}

.left input[type=text] {
  padding: 3px;
  font-size: 17px;
  border: none;
}

.right ul{
    display: flex;
    padding: 5%;
    margin-left: 15%;
    list-style: none;
    color: white;
}
.right ul li{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 20px;
    padding: 5% 2%;
    margin-right: 10%;
    width: 8%;
    list-style: none
}
.recomend{
    padding-top:15%;
    width: 100%;
}
.recomend h2{
    
    font-size: 90px;
    color:white;
    text-align:center; 
}
.recomend p{
    font-size: 23px;
    color:white;
    position: absolute;
    top:65%;
    left: 7%;
}
.recomend .btn{
    display:block;
    font-size: 15px;
    position: absolute;
    top:74%;
    left: 7%;
    color:#0a0a23;
    background-color: #fff;
    width: 15%;
    height: 5%;
    border-radius: 5px;
}
.recomend .btn1{
    display:block;
    font-size: 15px;
    position: absolute;
    top:80%;
    left: 7%;
    color:#0a0a23;
    background-color: #fff;
    width: 15%;
    height: 5%;
    border-radius: 5px;
}
"""

# display the HTML and CSS code
st.markdown(html, unsafe_allow_html=True)
st.write(f"<style>{css}</style>", unsafe_allow_html=True)

if st.button('Give it a try'):
    subprocess.Popen(['streamlit', 'run', 'Facial_Recognition.py'])
if st.button('Speak out'):
    subprocess.Popen(['streamlit', 'run', 'Verbal_Recognition.py'])