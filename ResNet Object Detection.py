import cv2
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

# Use imagenet instead
model = ResNet152(weights='resnet152_weights_tf_dim_ordering_tf_kernels.h5')

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    frame = cv2.resize(frame, (224, 224))
    image = frame[..., ::-1]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    predictions = model.predict(image)
    name = decode_predictions(predictions, top=1)[0][0][1]
    cv2.putText(frame, name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) == 13:
        break

capture.release()
cv2.destroyAllWindows()
