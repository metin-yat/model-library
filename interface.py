import streamlit as st

# Info About the Project

info_1 = st.Page(
    "info.py",
    title="Information About the Project",
    icon=":material/healing:",
)

# LLM Related Pages
llm_1 = st.Page(
    "llm\\chatbot.py",
    title="Chat with Llama3.2",
    icon=":material/healing:",
)

# Computer Vision Pages
cv_1 = st.Page(
    "computer_vision\\cv1\\yolo_detection.py",
    title="Object Detection with YOLOv9",
    icon=":material/computer:",
)

st.set_page_config(page_title="Model Library")

st.markdown("""
    <p style= 
            'position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 30px;
            padding: 10px;'>
            by Samed Metin YAT
    </p>
""", unsafe_allow_html=True)
st.header("Model Library", divider = True)

cv_pages, llm_pages, info_pages = [cv_1], [llm_1], [info_1]

page_dict = {}
page_dict["README"] = info_pages
page_dict["Computer Vision Projects"] = cv_pages
page_dict["LLM Related Projects"] = llm_pages

pg = st.navigation(page_dict)

pg.run()