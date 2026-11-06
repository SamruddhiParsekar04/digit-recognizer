import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_konva import st_canvas
from tensorflow import keras

st.set_page_config(page_title="Digit Recognizer", page_icon="✏️", layout="centered")


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.keras")


model = load_model()


def preprocess(canvas_rgba):
    # Take one color channel: white strokes on black background
    gray = canvas_rgba[:, :, 0].astype("uint8")

    # Find the box around the drawing
    ys, xs = np.where(gray > 20)
    if len(xs) == 0:
        return None
    gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Shrink so the longer side is 20 pixels (like MNIST digits)
    img = Image.fromarray(gray)
    w, h = img.size
    scale = 20 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    # Paste it in the center of a 28x28 black square
    square = Image.new("L", (28, 28), 0)
    square.paste(img, ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2))

    arr = np.array(square).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


st.markdown(
    """
    <h1 style='text-align:center; font-size:48px;
    background: linear-gradient(90deg, #8B5CF6, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    Digit Recognizer</h1>
    <p style='text-align:center; color:#A0A0B8; font-size:18px;'>
    Draw a digit from 0 to 9 and watch the AI guess it</p>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    canvas = st_canvas(
        fill_color="black",
        stroke_width=14,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        enable_viewport_controls=False,
        key="canvas",
    )

with right:
    if canvas.image_data is not None:
        x = preprocess(canvas.image_data)
        if x is None:
            st.info("Draw a digit to get a prediction.")
        else:
            probs = model.predict(x, verbose=0)[0]
            digit = int(np.argmax(probs))
            st.markdown(
                f"""
                <div style='text-align:center; padding:20px; border-radius:20px;
                background:#1A1A2E; border:1px solid #8B5CF6;'>
                    <p style='color:#A0A0B8; margin:0;'>I think it's a</p>
                    <h1 style='font-size:110px; margin:0;
                    background: linear-gradient(90deg, #8B5CF6, #06B6D4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;'>{digit}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"Confidence: {probs[digit] * 100:.1f}%")
            st.progress(float(probs[digit]))
            st.bar_chart(probs)