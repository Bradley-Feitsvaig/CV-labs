import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Directory containing images
image_dir = 'images'
output_dir = 'output_faces'
hat_image_path = "hat.jpg"  # Path to the new hat image

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load the new birthday hat image
hat = cv2.imread(hat_image_path, cv2.IMREAD_UNCHANGED)

# Create a mask for the hat
hat_gray = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(hat_gray, 240, 255, cv2.THRESH_BINARY_INV)

# Ensure the hat image has an alpha channel
if hat.shape[2] == 3:
    hat = cv2.cvtColor(hat, cv2.COLOR_BGR2BGRA)

# Add the mask to the alpha channel of the hat image
hat[:, :, 3] = mask

# Function to process and extract face from an image
def process_image(imagePath):
    img = cv2.imread(imagePath)

    if img is None:
        print(f"Error: Image {imagePath} not loaded correctly.")
        return

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    face_count = 0

    for (x, y, w, h) in faces:
        face_count += 1
        # Create a mask with a white background
        face = img[y:y + h, x:x + w]
        mask = np.ones_like(face, dtype=np.uint8) * 255

        # Calculate the center and radius for the circle
        center = (w // 2, h // 2)
        radius = min(center[0], center[1], w - center[0], h - center[1])

        # Draw a filled circle in the mask (black circle on white background)
        cv2.circle(mask, center, radius, (0, 0, 0), -1)

        # Apply the mask to the face region to keep the circle and make the rest white
        face_with_circle = np.where(mask == 0, face, mask)

        # Resize the hat to be a bit larger than the face width
        scale_factor = 1  # Increase the size of the hat by 80%
        hat_resized = cv2.resize(hat, (int(w * scale_factor), int(hat.shape[0] * w * scale_factor / hat.shape[1])), interpolation=cv2.INTER_AREA)
        hat_resized_h, hat_resized_w, _ = hat_resized.shape

        # Create an output image with extra space for the hat
        output_image = np.ones((h + hat_resized_h, w, 4), dtype=np.uint8) * 255
        output_image[hat_resized_h:, :w, :3] = face_with_circle
        output_image[hat_resized_h:, :w, 3] = 255

        # Calculate position to overlay the hat
        y_offset = 100  # Default y_offset value
        if os.path.basename(imagePath) == "2.jpg":
            y_offset += 50  # Increase y_offset if the image is "2.jpg"

        x_offset = (output_image.shape[1] - hat_resized_w) // 2

        # Overlay the hat on the output image
        for i in range(hat_resized_h):
            for j in range(hat_resized_w):
                if y_offset + i >= output_image.shape[0] or x_offset + j >= output_image.shape[1]:
                    continue
                alpha = hat_resized[i, j, 3] / 255.0
                output_image[y_offset + i, x_offset + j, :3] = alpha * hat_resized[i, j, :3] + (
                            1 - alpha) * output_image[y_offset + i, x_offset + j, :3]

        # Crop the output image to remove excess space
        final_output_image = output_image[:h + hat_resized_h, :, :]

        # Save the face image with the circle and hat
        face_filename = os.path.join(output_dir,
                                     f"{os.path.splitext(os.path.basename(imagePath))[0]}_face{face_count}.png")
        cv2.imwrite(face_filename, final_output_image)

        # Display the face image with the circle and hat
        face_rgb = cv2.cvtColor(final_output_image, cv2.COLOR_BGRA2RGBA)
        plt.figure(figsize=(5, 5))
        plt.imshow(face_rgb)
        plt.axis('off')
        plt.show()

# Loop over all images in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        imagePath = os.path.join(image_dir, filename)
        process_image(imagePath)