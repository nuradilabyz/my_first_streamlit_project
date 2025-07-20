import streamlit as st
from transformers import pipeline

user_text = st.text_area("Мәтінді осы жерге еңгізіңіз:", height=150)
classifier = pipeline("text-generation",model = "gpt2")
if st.button("Жауап алу"):
    if user_text:
        with st.spinner("Талдау жүрігізілуде..."):
            results = classifier(user_text,num_return_sequences=3)
            st.write("---")
            st.subheader("Нәтиже:")
            for i, result in enumerate(results):
                with st.container():
                    st.markdown(
                        f"""
                        <div style='
                            background-color:blue;
                            padding:15px;
                            margin-bottom:15px;
                            border-left:5px solid #4CAF50;
                            border-radius:10px;
                        '>
                        <h4>Нұсқа {i + 1}</h4>
                        <p style='font-size:16px'>{result['generated_text']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.warning("Талдау мәтінін еңгізіңіз")