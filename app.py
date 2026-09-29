import torch
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st

from Networks import FFNet


# -----------------------------------
# 1. Page setup
# -----------------------------------

st.set_page_config(
    page_title="Crowd Counting - FFNet",
    page_icon="👥"
)

st.title("👥 Crowd Counting using FFNet")
st.write("Upload a crowd image and the model will estimate the number of people.")


# -----------------------------------
# 2. Load model
# -----------------------------------

@st.cache_resource
def load_model():

    device = torch.device("cpu")

    model = FFNet.FFNet()
    model = model.to(device)

    model.load_state_dict(
        torch.load(
            "./SHB_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model, device


model, device = load_model()


# -----------------------------------
# 3. Image upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a crowd image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------
# 4. Process image
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Same preprocessing used by dataset
    

    # -----------------------------------
    # 4. Prepare image for FFNet
    # -----------------------------------

    # FFNet was trained using 512 x 512 image patches.
    # So we resize the uploaded image to 512 x 512
    # before giving it to the model.

    image = image.resize((512, 512), Image.BICUBIC)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    img = transform(image)

    # Add batch dimension
    img = img.unsqueeze(0)

    img = img.to(device)


    # -----------------------------------
    # 5. Model prediction
    # -----------------------------------

    with torch.no_grad():

        output = model(img)

        density_map = output[0]

        predicted_count = density_map.sum().item()


    # -----------------------------------
    # 6. Display result
    # -----------------------------------

    st.success(
        f"Estimated people: {predicted_count:.0f}"
    )

    st.write(
        f"Raw model prediction: {predicted_count:.2f}"
    )