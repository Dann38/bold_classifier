from ..cnn_bold_classifier import CNNBoldClassifier
import numpy as np
from .model import get_font_emb_model,get_font_bold_model
import cv2, torch
HEIGHT = 40
WIDTH = 40
class FontEmbClassifier(CNNBoldClassifier):
    def __init__(self):
        super().__init__()
        self.font_emb_model = get_font_emb_model("/home/daniil/project/bold_classifier/bold_classifier/cnn_bold_classifier/font_emb_classifier/model1.pt")
        self.font_bold_model = get_font_bold_model("/home/daniil/project/bold_classifier/bold_classifier/cnn_bold_classifier/font_emb_classifier/model_bold.pt")

    def resize(self, word_image):
        
        # Получаем текущие размеры изображения
        (original_height, original_width) = word_image.shape[:2]

        # Вычисляем новую ширину, сохраняя пропорции
        aspect_ratio = original_width / original_height
        new_width = int(HEIGHT * aspect_ratio)

        # Масштабируем изображение
        resized_image = cv2.resize(word_image, (new_width, HEIGHT), interpolation=cv2.INTER_LANCZOS4)

        return resized_image


    def get_word_evaluation(self, word_image: np.ndarray) -> float:
      
        resized_image = self.resize(word_image)
        gray_image = resized_image/255.0
        
        chars = [[gray_image[:, i*WIDTH:(i+1)*WIDTH]] for i in range(gray_image.shape[1]//WIDTH)]
        if len(chars) == 0:
            return 0.0
        data = torch.Tensor(chars)
        rez = self.font_emb_model(data)
        rez = self.font_bold_model(rez)
        rez = torch.sigmoid(rez)
        print(rez)
        return torch.mean(rez, 0)