import streamlit as st
from io import StringIO
from ultralytics import YOLO
from PIL import Image
import requests
import os , time
import platform
print(platform.system())
st.header('car accident detection system', divider='rainbow')
st.header(':blue[import car image] :sunglasses:')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    st.image(uploaded_file)
    img = Image.open(uploaded_file)
    model = YOLO("best.pt")
    results = model(img)
    names = model.names

    
    clas = []
    for r in results:
        for c in r.boxes.cls:
            clas.append(names[int(c)])
            print(names[int(c)])
    res = [clas[i] for i in range(len(clas)) if i == clas.index(clas[i]) ]
    
    print(res)
    
    
    API_TOKEN="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"
    API_URL = "https://api-inference.huggingface.co/models/opiljain/autotrain-cardamage-3762299975"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query():
        
        data = uploaded_file.getbuffer()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    output = query()
    
    print(output)
    
    if res and output:
        st.info('calculate fixing Cost estimation', icon="ℹ️")
        if output[0]["label"] == "Door" or output[0]["label"] == "hood" :
            st.write(output[0]["label"] + " has "+ res[0])
            col1, col2, col3 ,col4 ,col5 = st.columns(5)
            cost = 0
            with col1:
                x=number = st.number_input('hour')

            with col2:
                y=number = st.number_input('cost of hour')

            with col3:
                z=number = st.number_input('cost of new '+output[0]["label"])
            with col4:
                if st.button('calculate'):
                    cost = x*y+z
            with col5:
                if cost:
                    st.success(str(cost))
                
                    
        else:
            st.warning("cann't detect problem", icon="⚠️")
            
    else:
        st.warning("cann't detect car problem", icon="⚠️")
        if st.button('add to dataset', type="primary"):
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success('Done!')
            
            
   
    
    
# from ultralytics import YOLO
# model = YOLO("best.pt")
# results = model("12.jpeg")

# names = model.names

# for r in results:
#     for c in r.boxes.cls:
#         print(names[int(c)])

