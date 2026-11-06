import streamlit as st

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center; font-size:44px;
    background: linear-gradient(90deg, #8B5CF6, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    Model Insights</h1>
    <p style='text-align:center; color:#A0A0B8; font-size:18px;'>
    How well does the model really perform?</p>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", "98.78%")
c2.metric("Test images", "10,000")
c3.metric("Mistakes", "122")

st.subheader("Confusion matrix")
st.write(
    "Each row is the real digit and each column is what the model predicted. "
    "A bright diagonal means correct answers."
)
st.image("confusion_matrix.png")

st.subheader("How the model was built")
st.write("- A convolutional neural network (CNN) with two convolution layers")
st.write("- Trained on 60,000 handwritten digits from the MNIST dataset")
st.write("- Tested on 10,000 images it had never seen during training")
