# Для запуска: "streamlit run .\web.py" (запустится на localhost)
# Загружается изображение, выделяется маска, итоговую маску можно скачать

'''
Если фронтенд не нужен, закомментируйте часть после комментария "Код для Streamlit",
уберите import streamlit и добавьте 

import matplotlib.pyplot as plt
img = dlib.load_rgb_image("face1.jpg")
result = run(img)

plt.imshow(result)
plt.axis("off")
plt.show()

в конце кода
'''

import dlib
import cv2
import numpy as np
import streamlit as st

predictor_path = "shape_predictor_68_face_landmarks.dat"

# Наложение маски на лицо
def fill_face(img, shape):
    points = []

    # нижняя часть лица
    for i in range(17):

        x = shape.part(i).x
        y = shape.part(i).y

        points.append((x, y))

    face_height = shape.part(8).y - shape.part(19).y

    # верхняя часть (лоб)
    for i in range(26, 16, -1):

        x = shape.part(i).x
        y = shape.part(i).y - int(face_height*1/3) # чтобы 

        points.append((x, y))

    points = np.array(points, dtype=np.int32)

    # пустая маска
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    return mask, points


# Вырезание глаз, рта и бровей из маски
def cut_parts(img, shape, mask):
    # левый глаз
    left_eye = []

    for i in range(36, 42):

        x = shape.part(i).x
        y = shape.part(i).y

        left_eye.append((x, y))

    left_eye = np.array(left_eye, dtype=np.int32)

    cv2.fillPoly(mask, [left_eye], 0)


    # правый глаз
    right_eye = []

    for i in range(42, 48):

        x = shape.part(i).x
        y = shape.part(i).y

        right_eye.append((x, y))

    right_eye = np.array(right_eye, dtype=np.int32)

    cv2.fillPoly(mask, [right_eye], 0)


    # рот
    mouth = []

    for i in range(48, 60):  # только наружняя часть рта

        x = shape.part(i).x
        y = shape.part(i).y

        mouth.append((x, y))

    mouth = np.array(mouth, dtype=np.int32)

    cv2.fillPoly(mask, [mouth], 0)

    # левая бровь
    left_eyebrow = []

    for i in range(17, 22):
        x = shape.part(i).x
        y = shape.part(i).y

        left_eyebrow.append((x, y))

    left_eyebrow = np.array(left_eyebrow, dtype=np.int32)
    left_eyebrow = cv2.convexHull(left_eyebrow)

    cv2.fillPoly(mask, [left_eyebrow], 0)


    # правая бровь
    right_eyebrow = []

    for i in range(22, 27):
        x = shape.part(i).x
        y = shape.part(i).y

        right_eyebrow.append((x, y))

    right_eyebrow = np.array(right_eyebrow, dtype=np.int32)
    right_eyebrow = cv2.convexHull(right_eyebrow)

    cv2.fillPoly(mask, [right_eyebrow], 0)


# Основной код
def run(img):
    img_copy = img.copy()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        mask, points = fill_face(img_copy, shape)

        cv2.fillPoly(mask, [points], 255)

        cut_parts(img, shape, mask)

        result = cv2.bitwise_and(img, img, mask=mask)

        return result


# Код для Streamlit
st.title('Маска кожных покровов лица', anchor = False)

uploaded_file = st.file_uploader(label="Choose an image", accept_multiple_files=False, type=["png", "jpg"])

if st.button(label='Run', width="stretch"):
    if uploaded_file is None:
        st.warning("Please upload an image first")
    else:
        with st.spinner("Processing..."):
            png_jpg = uploaded_file.name.split(".")[-1].lower()
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = run(img)

            if png_jpg == "jpg":
                success, encoded_image = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                mime = "image/jpeg"
                out_name = "mask.jpg"
            
            elif png_jpg == "png":
                success, encoded_image = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                mime = "image/png"
                out_name = "mask.png"

            image_bytes = encoded_image.tobytes()
        
        st.download_button(
            label="Скачать маску",
            data=image_bytes,
            file_name=out_name,
            mime=mime
        )