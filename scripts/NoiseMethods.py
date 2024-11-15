import cv2
import numpy as np

class NoiseMethods:
    def __init__(self):
        pass

    def add_gaussian_noise(self, image):
        noise = np.random.normal(0, 10, image.shape).astype('float32')
        noisy_image = cv2.addWeighted(image.astype('float32'), 0.9, noise, 0.1, 0)
        return noisy_image

    def add_salt_and_pepper_noise(self, image, salt_prob=0.005, pepper_prob=0.005):
        noisy_image = np.copy(image)
        num_salt = int(np.ceil(salt_prob * image.size))
        num_pepper = int(np.ceil(pepper_prob * image.size))

        # Salt noise
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 255

        # Pepper noise
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1]] = 0

        return noisy_image
