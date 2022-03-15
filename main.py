import cv2
import reactions

# Capturing the Video Stream starts with frame 0
video_capture = cv2.VideoCapture(0)

# Creating the cascade objects and referencing xml files that detect face, and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


def main():
    # main loop
    while True:

        # Get individual frame
        _, frame = video_capture.read()
        # Covert the frame to grayscale
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all the faces in that frame
        detected_faces = face_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
        detected_eyes = eye_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)

        # check to see if there is a face detected
        if detected_faces != ():
            print("detected faces", detected_faces)
        elif detected_faces == detected_faces:
            print("No faces detected", detected_faces)

        # check to see if there is an eye detected
        if detected_eyes != ():
            print("Detected eyes:", detected_eyes)
        elif detected_eyes == detected_eyes:
            print("No detected eyes: ", detected_eyes)

        # Pass frame to draw_found_faces
        reactions.draw_found_faces(detected_faces, frame, (0, 0, 255))
        reactions.draw_found_faces(detected_eyes, frame, (0, 255, 0))

        # Display the updated frame as a video stream
        cv2.imshow('Webcam Face Detection', frame)

        # Press the ESC key to exit the loop
        # 27 is the code for the ESC key
        if cv2.waitKey(1) == 27:
            break

    # Releasing the webcam resource
    video_capture.release()
    # Destroy the window that was showing the video stream
    cv2.destroyAllWindows()


main()
