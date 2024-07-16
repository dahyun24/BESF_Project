import os
import sys
from lime import lime_image
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QStackedWidget, QDialog, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, QRect,pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QColor, QFontDatabase, QImage, QPainter,QMovie
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from PIL import Image  # Import Image module from PIL
from rembg import remove
import matplotlib.pyplot as plt
#import qdarktheme
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
bp_dense_model = load_model('sv_BP_CAM_DenseNet_AUG_100_0422.h5') #0(해방풍), 1(식방풍), 2(방풍)
bp_loaded_model = load_model('sv_BP_CAM_ResNet_AUG_96.77_0421.h5') #0(해방풍), 1(식방풍), 2(방풍)
sj_loaded_model = load_model('sv_SJ_CAM_ResNet_AUG_95?_0411.h5') #0(산조인),1(면조인)
from BP_gradcam_algorithm import BP_activate
from SJ_gradcam_algorithm import SJ_activate

#확정버튼 클릭 후 모델 선택
class ImageConfirmationWindow(QDialog):
    def __init__(self, parent=None):
        super(ImageConfirmationWindow, self).__init__(parent)
        self.classification_result = None

        # 모델 선택 버튼
        self.sanjo_button = QPushButton("산조인류 분류", self)
        self.sanjo_button.clicked.connect(self.sanjo_classification_clicked)

        self.bangpung_button = QPushButton("방풍류 분류", self)
        self.bangpung_button.clicked.connect(self.bangpung_classification_clicked)

        # Create a vertical layout for buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.sanjo_button)
        button_layout.addSpacing(50)  # Adjust the spacing as needed
        button_layout.addWidget(self.bangpung_button)

        # Set up the main layout
        layout = QVBoxLayout(self)
        layout.addLayout(button_layout)

        self.setWindowTitle("모델 선택")
        self.setGeometry(850, 350, 400, 300)

    def sanjo_classification_clicked(self):
        print("산조인류 분류 버튼을 클릭했습니다.")
        # Write to file
        file_path = "sanjo.txt"
        with open(file_path, "w") as file:
            file.write("   산조인류 모델로 분석 중입니다.")
        self.close()

    def bangpung_classification_clicked(self):
        print("방풍류 분류 버튼을 클릭했습니다.")
        # Write to file
        file_path = "bangpung.txt"
        with open(file_path, "w") as file:
            file.write("    방풍류 모델로 분석 중입니다.")
        self.close()

class VideoCaptureWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("촬영")
        self.parent_widget = parent  # 부모 위젯 저장

        #메인layout에 왼쪽오른쪽layout을 행으로 QHBoxLayout를 사용해 만들기
        #각각의 layout에 QVBoxLayout를 활용하여 수직으로 text상자-이미지상자-버튼까지 add하기

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(100, 50, 100, 80)  # Add space on all sides of the main layout

        # Left layout
        left_layout = QVBoxLayout()
        #left_layout.setAlignment(Qt.AlignVCenter)

        self.video_label = QLabel()
        self.video_label.setFixedSize(480,400)  # 적절한 크기로 조절하세요.
        self.video_label.setScaledContents(True)  # 이미지 크기를 QLabel의 크기에 맞게 자동으로 조정
        left_layout.addWidget(self.video_label)

        self.button_1 = QPushButton("촬  영", self)
        self.button_1.setFixedWidth(480)
        self.button_1.setFixedHeight(80)
        self.button_1.setStyleSheet("border-radius: 10px; background-color: #FFA500; color: black; font-size: 20pt; font-weight: bold;")
        self.button_1.clicked.connect(self.capture_button)
        left_layout.addWidget(self.button_1)

        main_layout.addLayout(left_layout)

        main_layout.addSpacing(130)  

        # Right layout
        right_layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setFixedSize(480, 400)
        self.image_label.setScaledContents(True)
        right_layout.addWidget(self.image_label)

        self.button_2 = QPushButton("확  정", self)
        self.button_2.setFixedWidth(480)
        self.button_2.setFixedHeight(80)
        self.button_2.setStyleSheet("border-radius: 10px; background-color: #32CD32; color: black; font-size: 20pt; font-weight: bold;")
        self.button_2.clicked.connect(self.confirm_button)
        right_layout.addWidget(self.button_2)

        main_layout.addLayout(right_layout)

        # 카메라 캡처 시작
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.zoom_factor = 1  # 줌 배율 초기값
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  

        self.test_image = None  

    def keyPressEvent(self, event):
        key = event.key()
        print("Key pressed:", key)
        if key == Qt.Key_Plus:
            self.zoom_factor = min(self.zoom_factor * 1.1, 4)
            print("Zoom factor:", self.zoom_factor)

        elif key == Qt.Key_Minus:
            self.zoom_factor = max(self.zoom_factor / 1.1, 1)
            print("Zoom factor:", self.zoom_factor)

    def zoom_frame(self, frame, zoom_factor):
        height, width = frame.shape[:2]
        centerX, centerY = int(width / 2), int(height / 2)

        # 줌 배율에 따른 새로운 너비와 높이 계산
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)

        # 새로운 너비와 높이를 중심으로 프레임을 자릅니다
        left_x = max(centerX - new_width // 2, 0)
        right_x = min(centerX + new_width // 2, width)
        top_y = max(centerY - new_height // 2, 0)
        bottom_y = min(centerY + new_height // 2, height)

        cropped_frame = frame[top_y:bottom_y, left_x:right_x]

        # 자른 프레임을 원본 프레임 크기로 리사이즈
        return cv2.resize(cropped_frame, (width, height))
    
    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            if self.zoom_factor != 1:
                frame = self.zoom_frame(frame, self.zoom_factor)
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
  

    def capture_button(self, text):
        print("촬영 버튼을 클릭했습니다.")
            #이미지 전처리 함수
        def image_prepro(test_image):
            image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            image = remove(image) #배경 제거
            width, height = image.size
            square_size = max(width, height)
            background = Image.new('RGB', (square_size, square_size), (0, 0, 0)) #검은 배경 이미지 생성
            offset = ((square_size - width) // 2, (square_size - height) // 2)
            background.paste(image, offset) #중앙에 붙여넣기
            image = background.resize((299, 299)) # 300->299로
            image = image.convert('RGB') #RGB로 변환
            img_cv_rembg = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) #배경 제거 이미지 cv로 변환

            gray = cv2.cvtColor(img_cv_rembg, cv2.COLOR_BGR2GRAY) #흑백 이미지 생성
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 8, 15)
            pil_edge_array = np.array(Image.fromarray(edges)) #엣지 이미지 생성

            cv2.imwrite('image_rembg.jpg', img_cv_rembg)
            cv2.imwrite('image_gray.jpg', gray)
            Image.fromarray(pil_edge_array).save('image_edge.jpg')

        ret, frame = self.capture.read()
        if ret:
            if self.zoom_factor != 1:
                frame = self.zoom_frame(frame, self.zoom_factor)
            self.test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite("image_rgb.jpg", frame)

            pixmap = QPixmap.fromImage(QImage(self.test_image, self.test_image.shape[1], self.test_image.shape[0], self.test_image.shape[1] * 3, QImage.Format_RGB888))
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            
            image_prepro(frame) # 이미지 전처리 함수
    

    def confirm_button(self, text):
        print("확정 버튼을 클릭했습니다.")
        # Add your right button action here
        confirmation_window = ImageConfirmationWindow()
        confirmation_window.exec_()

        self.parent_widget.show_prediction()

 
class PredictionWindow(QWidget):        
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("예측")
        self.main_window = parent

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(100,50,100,50)
        main_layout.setAlignment(Qt.AlignCenter)

        # Image Label 생성
        self.image_label = QLabel()

        # 이미지 파일 로드 및 Image Label에 설정
        pixmap = QPixmap("image_rembg.jpg")  # 이미지 파일 경로에 맞게 수정하세요
        self.image_label.setPixmap(pixmap)
        self.image_label.setStyleSheet("border: none")
        self.image_label.setFixedSize(400,400) 
        self.image_label.setScaledContents(True)

        # main_layout에 Image Label 추가
        main_layout.addWidget(self.image_label)

        main_layout.addSpacing(70)

        # "촬영된 이미지" 라벨 추가
        self.text_1 = QLabel("모델 분류 중")
        self.text_1.setStyleSheet("border: none; font-size: 20pt; font-weight: bold;")  # 테두리 없음, 글자 크기 키우기, bold체 설정

        # Check if sanjo.txt exists
        if os.path.exists("sanjo.txt"):
            with open("sanjo.txt", "r") as file:
                content = file.read()
            self.text_1.setText(content)
        # Check if bangpung.txt exists
        elif os.path.exists("bangpung.txt"):
            with open("bangpung.txt", "r") as file:
                content = file.read()
            self.text_1.setText(content)

        main_layout.addWidget(self.text_1)

        main_layout.addSpacing(30)

        gif_path = 'loading_gre.gif'
        self.movie = QMovie(gif_path)

        self.label = QLabel()
        self.label.setMovie(self.movie)
        self.label.setStyleSheet("border: none;")
        self.movie.start()

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addSpacing(100)
        horizontal_layout.addWidget(self.label)
        main_layout.addLayout(horizontal_layout)
        
        # 타이머 설정 (5초 후에 마지막 결과 창으로 자동 전환)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.switch_to_analysis)
        self.timer.start(1000)  # 1초 후에 timeout 이벤트 발생


    def switch_to_analysis(self):
        # 분석 화면으로 자동 전환
        self.main_window.show_analysis()

class AnalysisWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("분석")
        self.main_window = parent

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(100, 50, 100, 50)  # Add space on all sides of the main layout

        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        main_layout.addSpacing(30) #

        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        self.image = Image.open("image_rembg.jpg")
        self.image_array = keras_image.img_to_array(self.image)
        self.image_array = np.expand_dims(self.image_array, axis=0)
        self.image_array /= 255.

        model_name = None
        layer_name = None
        resize_dim = None
        if os.path.exists("sanjo.txt"):
            model_name = sj_loaded_model
            layer_name = 'conv2d_15'
            resize_dim = (143, 143)
        elif os.path.exists("bangpung.txt"):
            model_name = bp_loaded_model
            layer_name = 'conv2d_5'
            resize_dim = (295, 295)

        preds = model_name.predict(self.image_array)
        predicted_class_index = np.argmax(preds[0])

        def make_lime_image(model_name, image):           
            def segment_image_with_slic(image, n_segments=150, compactness=10, sigma=1):
                segments_slic = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
                return segments_slic
            
            image_array_lime = np.array(image)
            
            explainer = lime_image.LimeImageExplainer()
            # LIME 설명 생성
            explanation = explainer.explain_instance(image_array_lime.astype('double'), 
                                                     lambda x: model_name.predict(x), 
                                                     top_labels=5, hide_color=0, num_samples=250,
                                                     segmentation_fn=lambda x: segment_image_with_slic(x, n_segments=150, compactness=10, sigma=1))

            # 히트맵을 원본 이미지 위에 나타내기
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
            #배경 부분 비활성화
            background = np.all(image_array_lime == 0, axis=-1) # 모든 채널이 0인 픽셀 위치 찾기
            mask[background] = False # 배경에 해당하는 마스크 부분을 0으로 설정
            plt.imshow(mark_boundaries(image_array_lime, mask, color=(0, 0, 1)))
            plt.axis('off')
            save_path = f'lime_heatmap.jpg'  # 저장할 파일명 지정
            plt.savefig(save_path, format='jpg', bbox_inches = 'tight', pad_inches = 0)  # jpg 파일로 저장
            plt.close()

        make_lime_image(model_name, self.image)
        
        def make_gradcam_heatmap(img_array, model_name, layer_name, pred_index):
            grad_model = tf.keras.models.Model([model_name.inputs], [model_name.get_layer(layer_name).output, model_name.output])
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]
            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
        
        #방풍류 conv2d_5 295,295        
        self.heatmap = make_gradcam_heatmap(self.image_array, model_name, layer_name, predicted_class_index)
        img = self.image.resize(resize_dim)
        #산조인류 conv2d_15 143,143

        img_array = keras_image.img_to_array(img)
        gray_img_array = img_array[:, :, 0]
        mask = gray_img_array == 0
        self.heatmap[mask] = 0

        cmap = plt.cm.gist_ncar
        cmap.set_under(color='black')
        non_zero_pixels = self.heatmap[self.heatmap != 0]

        #vmin, vmax 계산
        Q1 = np.percentile(non_zero_pixels, 25)
        Q3 = np.percentile(non_zero_pixels, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + IQR
        vmin = lower_bound
        vmin = 0.1
        vmax = upper_bound

        plt.imshow(self.heatmap, cmap=cmap, vmin=vmin, vmax = vmax)
        plt.colorbar()
        plt.axis('off')
        plt.savefig('gradcam_heatmap.jpg', bbox_inches='tight')
        plt.show()
        plt.close()
        
        self.image_array *= 255.
        self.model = self.model_load()


        # 이미지 레이블 추가 함수
        def add_image_label(image_path):
            # Image Label 생성
            image_label = QLabel()

            # 이미지 파일 로드 및 Image Label에 설정
            pixmap = QPixmap(image_path)  # 이미지 파일 경로에 맞게 수정하세요
            image_label.setPixmap(pixmap)
            image_label.setStyleSheet("border: none")
            image_label.setFixedSize(300,300)
            image_label.setScaledContents(True)

            # main_layout에 Image Label 추가
            top_layout.addWidget(image_label)
            top_layout.addSpacing(70)

        # 3개의 이미지를 추가합니다.
        for image_path in ["image_rgb.jpg", "image_gray.jpg", "image_edge.jpg"]:
            add_image_label(image_path)

        # Image Label 생성
        self.image_label_lime_image = QLabel()

        # 이미지 파일 로드 및 Image Label에 설정
        pixmap = QPixmap("lime_heatmap.jpg")  # 이미지 파일 경로에 맞게 수정하세요
        self.image_label_lime_image.setPixmap(pixmap)
        self.image_label_lime_image.setStyleSheet("border: none")
        self.image_label_lime_image.setFixedSize(340,320) 
        self.image_label_lime_image.setScaledContents(True)

        # main_layout에 Image Label 추가
        bottom_layout.addWidget(self.image_label_lime_image)
        bottom_layout.addSpacing(50)


        # Image Label 생성
        self.image_label_gradcam_image = QLabel()

        # 이미지 파일 로드 및 Image Label에 설정
        pixmap_grad = QPixmap("gradcam_heatmap.jpg")  # 이미지 파일 경로에 맞게 수정하세요
        self.image_label_gradcam_image.setPixmap(pixmap_grad)
        self.image_label_gradcam_image.setStyleSheet("border: none")
        self.image_label_gradcam_image.setFixedSize(320,320) 
        self.image_label_gradcam_image.setScaledContents(True)

        # main_layout에 Image Label 추가
        bottom_layout.addWidget(self.image_label_gradcam_image)
        bottom_layout.addSpacing(40)


        self.text = QLabel("         잠시후 결과창으로 이동합니다.")
        self.text.setStyleSheet("border: none; font-size: 15pt;")  # 테두리 없음, 글자 크기 키우기, bold체 설정
        bottom_layout.addWidget(self.text)


        # 타이머 설정 (5초 후에 마지막 결과 창으로 자동 전환)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.switch_to_result_window)
        self.timer.start(7000)  # 10초 후에 timeout 이벤트 발생

    def model_load(self):
        # Check if sanjo.txt exists
        if os.path.exists("sanjo.txt"):
            os.remove("sanjo.txt")
            return self.load_sanjo_model()
        # Check if bangpung.txt exists
        elif os.path.exists("bangpung.txt"):
            os.remove("bangpung.txt")
            return self.load_bangpung_model()
        else:
            # Handle error condition, invalid model number
            return None

    def load_sanjo_model(self):
        center_mean_ratio, normalized_std_deviation = SJ_activate(self.heatmap)
        if center_mean_ratio > 0.6: #수정 필요
            if normalized_std_deviation >= 0.6:
                print("center+line")
                index = 0
            else:
                print("center+noline")
                index = 1
        else:
            print("edge")
            index = 2
                
            #산조인류 모델 예측 함수
        def SJ_model_predict(image_array):
            predictions = sj_loaded_model.predict(image_array)
            print(predictions.shape)
            predicted_class = np.argmax(predictions)
            return predicted_class, predictions

        predicted_class, predictions = SJ_model_predict(self.image_array)
        with open("explain.txt", "w") as file:
            if predicted_class == 1:
                file.write("이 이미지가 '면조인'일 확률은 {:.2f}% 입니다.\n면조인(전자조 Ziziphus mauritiana)\n".format(predictions[0,1] * 100))
                if index == 0:
                    file.write('활성화 부위 : center + line\n1. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n2. 회황색을 띰\n3. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n4. 산조인에 비해 두께가 얇고 납작함\n5.흔히 비늘 모양 무늬가 있음\n')
                elif index == 1:
                    file.write('활성화 부위 : center\n1. 회황색을 띰\n2. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n3. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n4. 산조인에 비해 두께가 얇고 납작함\n5.흔히 비늘 모양 무늬가 있음\n')
                elif index == 2:
                    file.write('활성화 부위 : edge\n1. 산조인에 비해 두께가 얇고 납작함\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n4. 회황색을 띰\n5.흔히 비늘 모양 무늬가 있음\n')
            elif predicted_class == 0:
                file.write("이 이미지가 '산조인'일 확률은 {:.2f}% 입니다.\n산조인(산조 Ziziphus jujuba var. spinosa)\n".format(predictions[0,0] * 100))
                if index == 0:
                    file.write('활성화 부위 : center + line\n1. 자색 또는 자갈색을 띰\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n4. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음\n')
                elif index == 1:
                    file.write('활성화 부위 : center\n1. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n2. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음\n3. 자색 또는 자갈색을 띰\n4. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n')
                elif index == 2:
                    file.write('활성화 부위 : edge\n1. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 자색 또는 자갈색을 띰\n4. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음\n')
            else:
                file.write("기타\n")
        
        with open("result_sanjo.txt", "w") as file:
            file.write("그룹 : 산조인류\n\n")
            file.write(f"면조인 판정 확률 : {predictions[0,1]*100:.2f}%\n")
            file.write(f"산조인 판정 확률 : {predictions[0,0]*100:.2f}%\n")
        predictions = np.round(predictions, 3)
        print(predictions)

    def load_bangpung_model(self):
        center_mean_ratio, edges_above_mean = BP_activate(self.heatmap)
        if center_mean_ratio > 0.6: #센터 활성화 중
            if edges_above_mean >= 2:
                print("center+edge")
                index = 0
            else:
                print("center")
                index = 1
        else:
                print("edge")
                index = 2
        #방풍류 모델 예측 함수
        def BP_model_predict(image_array):
            predictions = bp_dense_model.predict(image_array)
            print(predictions.shape)
            predicted_class = np.argmax(predictions)
            return predicted_class, predictions

        predicted_class, predictions = BP_model_predict(self.image_array)
        with open("explain.txt", "w") as file:
            if predicted_class == 0:
                file.write("이 이미지가 '해방풍'일 확률은 {:.2f}% 입니다.\n해방풍(갯방풍 Glehnia littoralis)\n".format(predictions[0,0] * 100))
                if index == 0:
                    file.write('활성화 부위 : center + edge\n1. 가늘고 긴 원기둥 모양\n2. 목부는 연한 노란색이며 조직이 치밀함\n3. 피부는 가루성이며, 갈색의 분비도가 작은 점으로 흩어져 있음\n4. 여러 개의 벌어진 틈이 있음\n')
                elif index == 1:
                    file.write('활성화 부위 : center\n1. 목부는 연한 노란색이며 조직이 치밀함\n2. 여러 개의 벌어진 틈이 있음\n3. 가늘고 긴 원기둥 모양\n4. 피부는 가루성이며, 갈색의 분비도가 작은 점으로 흩어져 있음\n')
                elif index == 2:
                    file.write('활성화 부위 : edge\n1. 피부는 가루성이며, 갈색의 분비도가 작은 점으로 흩어져 있음\n2. 가늘고 긴 원기둥 모양\n3. 여러 개의 벌어진 틈이 있음\n4. 목부는 연한 노란색이며 조직이 치밀함\n')
            elif predicted_class == 1:
                file.write("이 이미지가 '식방풍'일 확률은 {:.2f}% 입니다.\n식방풍(갯기름나물 Peucedanum japonicum)\n".format(predictions[0,1] * 100))
                if index == 0:
                    file.write('활성화 부위 : center + edge\n1. 식방풍은 피부와 목부의 색깔이 비슷하다\n2. 목부는 황갈색이며 치밀함\n3. 피부는 회갈색,연한 갈색이며 빈틈이 보이고 갈색의 형성층이 뚜렷함\n')
                elif index == 1:
                    file.write('활성화 부위 : center\n1. 목부는 황갈색이며 치밀함\n2. 피부는 회갈색,연한 갈색이며 빈틈이 보이고 갈색의 형성층이 뚜렷함\n3. 식방풍은 피부와 목부의 색깔이 비슷함\n')
                elif index == 2:
                    file.write('활성화 부위 : edge\n1. 피부는 회갈색,연한 갈색이며 빈틈이 보이고 갈색의 형성층이 뚜렷함\n2. 목부는 황갈색이며 치밀함\n3. 식방풍은 피부와 목부의 색깔이 비슷함\n')
            elif predicted_class == 2:
                file.write("이 이미지가 '방풍'일 확률은 {:.2f}% 입니다.\n방풍(Saposhnikovia divaricata)\n".format(predictions[0,2] * 100))
                if index == 0:
                    file.write('활성화 부위 : center + edge\n1. 목부는 연한 노란색임\n2. 피부는 연한 갈색이고 회갈색의 벌어진 틈이 여러개 있음\n3. 목부에 비해 피부의 색깔이 뚜렷하게 진함\n')
                elif index == 1:
                    file.write('활성화 부위 : center\n1. 목부는 연한 노란색임\n2. 피부에 비해 목부의 색깔이 뚜렷하게 연함\n3. 피부는 연한 갈색이고 회갈색의 벌어진 틈이 여러개 있음\n')
                elif index == 2:
                    file.write('활성화 부위 : edge\n1. 피부는 연한 갈색이고 회갈색의 벌어진 틈이 여러개 있음\n2. 목부에 비해 피부의 색깔이 뚜렷하게 진함\n3. 목부는 연한 노란색임\n')
            else:
                file.write("기타\n")

        with open("result_bangpung.txt", "w") as file:
            file.write("그룹 : 방풍류\n\n")
            file.write(f"해방풍 판정 확률 : {predictions[0,0]*100:.2f}%\n")
            file.write(f"식방풍 판정 확률 : {predictions[0,1]*100:.2f}%\n")
            file.write(f"방풍 판정 확률: {predictions[0,2]*100:.2f}%\n")

        predictions = np.round(predictions, 3)
        print(predictions)
    
    def switch_to_result_window(self):
        self.timer.stop()  # 타이머 중지

        self.main_window.show_result()

class ResultWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("결과")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50,50,50,50)  # Add space on all sides of the main layout


        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        main_layout.addSpacing(50)

        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)
        
        # 이미지 레이블 추가 함수
        def add_image_label(image_path):
            image_label = QLabel()

            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap)
            image_label.setStyleSheet("border: none")
            image_label.setFixedSize(300,300) 
            image_label.setScaledContents(True)

            top_layout.addWidget(image_label)
            top_layout.addSpacing(30) 

        for image_path in ["image_rgb.jpg", "lime_heatmap.jpg", "gradcam_heatmap.jpg"]:
            add_image_label(image_path)

        self.text_box = QTextEdit(self)
        self.text_box.setStyleSheet("font-size: 17pt; border: 2px solid orange; background-color: white; padding: 21px;")
        top_layout.addWidget(self.text_box)

        # Check if sanjo.txt exists
        if os.path.exists("result_sanjo.txt"):
            self.load_text_file("result_sanjo.txt",0)
            os.remove("result_sanjo.txt")
        # Check if bangpung.txt exists
        elif os.path.exists("result_bangpung.txt"):
            self.load_text_file("result_bangpung.txt",0)
            os.remove("result_bangpung.txt")

        self.text_box2 = QTextEdit(self)
        self.text_box2.setStyleSheet("font-size: 18pt; border: 4px solid grey; background-color: white; padding: 23px;")
        bottom_layout.addWidget(self.text_box2)

        # Load specific text file
        self.load_text_file("explain.txt",1)
        


    def load_text_file(self, file_name, num):
       with open(file_name, 'r', encoding='utf-8') as file:
           text = file.read()
           if num == 0:
               self.text_box.setText(text)
           elif num == 1:
               self.text_box2.setText(text)



class HomeButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("⌂")  # 홈모양 이모티콘 설정
        self.setStyleSheet("font-size: 35px;")  # 글자 크기 설정
        self.clicked.connect(self.go_to_home)  # 홈 버튼 클릭 시 이벤트 연결

    def go_to_home(self):
        self.parent().clear_content_area()  # 현재 메인 윈도우의 내용을 초기화
        self.parent().show_placeholder()  # 시작 화면 표시
        
        # 버튼들의 스타일을 원래 스타일로 변경
        button_list = [self.parent().button1, self.parent().button2, self.parent().button3, self.parent().button4]
        for button in button_list:
            button.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")
            button.setStyleSheet("""
                QPushButton {
                    background-color: #DCDCDC;
                    border: 2px solid black;
                }
                QPushButton:hover {
                    background-color: #FFFACD; /* 마우스 호버 시 연한 노란색 */
                }
            """)

class SecondPage(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("한약재 분류 프로그램")
        #self.showMaximized()

        #self.setGeometry(0,0,1860,1080)
        self.setStyleSheet("background-color: white;")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Grid Layout
        grid_layout = QGridLayout()

        home_button = HomeButton(self)
        grid_layout.addWidget(home_button, 0, 0, 3, 1)

        title_label = QLabel("실시간 한약재 분류")
        font = title_label.font()
        font.setPointSize(32)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(title_label, 0, 5, 3, 5)

        logo_label = QLabel()
        logo_pixmap = QPixmap("logo3.jpg").scaled(310, 120)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(logo_label, 1, 12, 2, 2)

        self.button1 = QPushButton(self)
        self.button2 = QPushButton(self)
        self.button3 = QPushButton(self)
        self.button4 = QPushButton(self)

        self.setup_button(self.button1, "시  작")
        self.setup_button(self.button2, "예  측")
        self.setup_button(self.button3, "분  석")
        self.setup_button(self.button4, "결  과")

        grid_layout.addWidget(self.button1, 3, 0, 3, 3)
        grid_layout.addWidget(self.button2, 6, 0, 3, 3)
        grid_layout.addWidget(self.button3, 9, 0, 3, 3)
        grid_layout.addWidget(self.button4, 12, 0, 3, 3)

        self.button1.clicked.connect(self.show_video_capture)
        self.button2.clicked.connect(self.show_prediction)
        self.button3.clicked.connect(self.show_analysis)
        self.button4.clicked.connect(self.show_result)

        self.content_area = QWidget(self)
        self.content_area.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")
        self.content_area_layout = QVBoxLayout(self.content_area)
        self.content_area_layout.setContentsMargins(30, 0, 30, 0)
        
        grid_layout.addWidget(self.content_area, 3, 4, 12, 10)
        self.layout.addLayout(grid_layout)

        self.show_placeholder()

    def setup_button(self, button, text):
        button.setText(text)
        button_font = button.font()
        button_font.setPointSize(22)
        button_font.setBold(True)
        button.setFont(button_font)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button_style = """
        QPushButton {
            background-color: #DCDCDC; /* 연한 회색 */
            border: 2px solid black; /* 테두리 두께 */
        }

        QPushButton:hover {
            background-color: #FFFACD; /* 마우스 호버 시 연한 노란색 */
        }

        QPushButton:checked {
            background-color: #FFFACD; /* 클릭 시에도 연한 노란색 유지 */
        }
        """
        button.setStyleSheet(button_style)

    def show_placeholder(self):
        placeholder_text = "실시간 한약재 분석 프로그램입니다.\n\n시작을 원하시면 왼쪽 위의 <시작> 버튼을 눌러주세요."
        placeholder_label = QLabel(placeholder_text, self)
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setFixedSize(1200, 300)  # 텍스트 상자의 크기 설정
        placeholder_label.setStyleSheet("font-size: 35px; font-weight: bold; background-color: white;")  # 글자 크기와 굵기 설정
        self.content_area_layout.addWidget(placeholder_label, alignment=Qt.AlignCenter)  # 가운데 정렬

    def show_video_capture(self):
        self.clear_content_area()
        self.content_area_layout.addWidget(VideoCaptureWindow(self))
        self.button1.setChecked(True)
        self.button2.setChecked(False)
        self.button3.setChecked(False)
        self.button4.setChecked(False)
        self.button1.setStyleSheet("background-color: #FFFACD; border: 2px solid black;")  # 촬영 버튼 노란색으로 스타일 변경
        self.button2.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")  # 나머지 버튼은 원래 스타일로 변경
        self.button3.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")
        self.button4.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")

    def show_prediction(self):
        self.clear_content_area()
        self.content_area_layout.addWidget(PredictionWindow(self))
        self.button1.setChecked(False)
        self.button2.setChecked(True)
        self.button3.setChecked(False)
        self.button4.setChecked(False)
        self.button2.setStyleSheet("background-color: #FFFACD; border: 2px solid black;")  # 예측 버튼 노란색으로 스타일 변경
        self.button1.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")  # 나머지 버튼은 원래 스타일로 변경
        self.button3.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")
        self.button4.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")

    def show_analysis(self):
        self.clear_content_area()
        self.content_area_layout.addWidget(AnalysisWindow(self))
        self.button1.setChecked(False)
        self.button2.setChecked(False)
        self.button3.setChecked(True)
        self.button4.setChecked(False)
        self.button3.setStyleSheet("background-color: #FFFACD; border: 2px solid black;")  # 분석 버튼 노란색으로 스타일 변경
        self.button1.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")  # 나머지 버튼은 원래 스타일로 변경
        self.button2.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")
        self.button4.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")

    def show_result(self):
        self.clear_content_area()
        self.content_area_layout.addWidget(ResultWindow(self))
        self.button1.setChecked(False)
        self.button2.setChecked(False)
        self.button3.setChecked(False)
        self.button4.setChecked(True)
        self.button4.setStyleSheet("background-color: #FFFACD; border: 2px solid black;")  # 결과 버튼 노란색으로 스타일 변경
        self.button1.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")  # 나머지 버튼은 원래 스타일로 변경
        self.button2.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")
        self.button3.setStyleSheet("background-color: #DCDCDC; border: 2px solid black;")   

    def clear_content_area(self):
        while self.content_area_layout.count():
            item = self.content_area_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

class FirstPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        # 가로 레이아웃 생성
        hbox = QHBoxLayout()

        # 레이블 추가
        layout.addSpacing(50)
        self.label = QLabel('한(생)약재 관능 검사 보조 XAI 프로그램')
        font = self.label.font()
        font.setPointSize(32)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hbox.addWidget(self.label)

        # 수직 레이아웃에 가로 레이아웃 추가
        layout.addLayout(hbox)
        #layout.addSpacing(100)  # 간격을 추가하여 hbox와 label_layout 사이의 간격을 늘립니다.
        
        # 버튼 추가
        label_layout = QVBoxLayout()
        self.label2 = QLabel('과제명: 인공지능 XAI 등을 활용한 한약(생약)재 관능검사 보조기술 개발 연구')
        font = self.label2.font()
        font.setPointSize(15)
        #font.setBold(True)
        self.label2.setFont(font)
        self.label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_layout.addWidget(self.label2)

        label_layout.addSpacing(50) #50->30

        label_text = "실시간 한약재 분석 프로그램입니다.시작을 원하시면 <시작> 버튼을 눌러주세요."
        label_1 = QLabel(label_text, self)
        label_1.setAlignment(Qt.AlignCenter)
        label_1.setStyleSheet("font-size: 30px; font-weight: bold; border: none;")  # 글자 크기와 굵기 설정
        label_layout.addWidget(label_1, alignment=Qt.AlignCenter)  # 가운데 정렬
        label_layout.addSpacing(50) #50->30

        # 이미지 추가
        image_label = QLabel(self)
        pixmap = QPixmap("screenshot_resize.png").scaled(450,400)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_layout.addWidget(image_label)
        label_layout.addSpacing(30)

        self.button = QPushButton('시작')
        self.button.setFixedSize(200, 80)  # 버튼의 크기를 조정합니다.
        self.button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 25px;")  # 배경색과 글자색을 설정합니다.
        label_layout.addWidget(self.button, alignment=Qt.AlignCenter)
        
        self.logo = QLabel(self)
        self.logo.setPixmap(QPixmap("logo3.jpg").scaled(290, 100))

        label_layout.addWidget(self.logo)
        layout.addLayout(label_layout)
        self.setLayout(layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('한약재 분류 프로그램')
        #self.setMinimumSize(1860,1080)
        self.setStyleSheet("background-color: white;")

        self.stackedWidget = QStackedWidget(self)

        self.page1 = FirstPage()
        self.page1.button.clicked.connect(self.showSecondPage)
        self.stackedWidget.addWidget(self.page1)

        self.page2 = SecondPage()
        #self.page2.home_button.clicked.connect(self.showFirstPage)
        self.stackedWidget.addWidget(self.page2)

        layout = QVBoxLayout()
        layout.addWidget(self.stackedWidget)
        self.setLayout(layout)
    

    def showSecondPage(self):
        self.stackedWidget.setCurrentIndex(1)

    def showFirstPage(self):
        self.stackedWidget.setCurrentIndex(0)
                

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #qdarktheme.setup_theme("light")
    # 사용할 한국어 폰트 선택
    korean_font = QFont("맑은 고딕")
    
    # 애플리케이션 전체의 폰트 설정
    app.setFont(korean_font)
    main_window = MainWindow()
    main_window.resize(1860, 1080)
    main_window.show()
    sys.exit(app.exec_())
