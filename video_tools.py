import tensorflow as tf
import tensorflow.keras.utils as np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, GRU, LSTM
from keras.metrics import top_k_categorical_accuracy
import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from PIL import Image
from IPython.display import display

import random
import mediapipe as mp
import sys
from glob import glob
import os
import time
import copy
import pickle
import numpy as np
import cv2
mp_drawing_styles = mp.solutions.drawing_styles

DEBUG_PATH = "C:/Users/jimmy/Documents/DEMO_VIDEOS/DEBUG_FRAMES/"

MAX_FRAMES_VID = 90
##Index for face landmarks
LANDMARK_IDX = [156, 70, 63, 105, 66, 107, 55, 193, 246, 161, 160, 159, 158, 157, 173, 33, 7, 163, 144, 145, 153, 154, 155, 133, 383, 300, 293, 334, 296, 336, 285, 417, 466, 388, 387, 386, 385, 384, 398, 263, 249, 390, 373, 374, 380, 381, 382, 362, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]
all_classes = np.load('NPY_SPLITS/ALL_CLASSES.npy',allow_pickle=True)

def get_all_classes():
    return all_classes

def crop_video_and_get_frames_90(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if num_frames > MAX_FRAMES_VID:
        remove_frames = (num_frames - MAX_FRAMES_VID) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, remove_frames)
        for i in range(MAX_FRAMES_VID):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        # while len(frames) > MAX_FRAMES_VID:
        #    frames.pop()
        # Release the video file
        cap.release()
    else:
        for i in range(MAX_FRAMES_VID):
            success, frame = cap.read()
            if not success:
                break
            else:
                frames.append(frame)
    return frames

def get_keypoints_mp_base(video_path):
    FRAMES = 90
    KEYPOINTS = 134
    frames = crop_video_and_get_frames_90(video_path)
    holistic_model = mp.solutions.holistic.Holistic()
    frames_video_list_0 = []
    frameNr = 0
    for frame in frames:
        hand_landmarks_left = np.zeros(shape=(21,2))
        hand_landmarks_right = np.zeros(shape=(21,2))
        pose_landmarks_list = np.zeros(shape=(25,2))
        frame_KPs = np.zeros(shape=(KEYPOINTS))
        BGR_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(BGR_Image)
        if results.left_hand_landmarks: 
            hand_landmarks_left = np.array([[lmk.x, lmk.y] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
        else:
            hand_landmarks_left = np.zeros(shape=(21,2))
        if results.right_hand_landmarks:
            hand_landmarks_right = np.array([[lmk.x, lmk.y] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
        else: 
            hand_landmarks_right = np.zeros(shape=(21,2))        
        if results.pose_landmarks:
            pose_landmarks_list = []
            pose_landmarks = results.pose_landmarks
            pose_landmarks_list = np.array([[lmk.x, lmk.y] for i, lmk in enumerate(results.pose_landmarks.landmark) if i < 25], dtype=np.float32)             
        frames_video_list_0.append(np.concatenate((hand_landmarks_left.flatten().tolist(),hand_landmarks_right.flatten(),pose_landmarks_list.flatten()), axis=None)) 
        frameNr = frameNr+1
    while frameNr < FRAMES:
        if len(frames_video_list_0) < FRAMES:
            frames_video_list_0.append(np.zeros(KEYPOINTS))        
        frameNr = frameNr+1
    video_kp_0 = np.stack(frames_video_list_0)
    video_kp_0 = video_kp_0.reshape(-1, FRAMES, KEYPOINTS)
    holistic_model.close()
    return video_kp_0


def get_keypoints_mp_pose_face(video_path):
    FRAMES = 90
    KEYPOINTS = 270
    frames = crop_video_and_get_frames_90(video_path)    
    holistic_model = mp.solutions.holistic.Holistic()
    frames_video_list_0 = []
    frameNr = 0
    for frame in frames:
        hand_landmarks_left = np.zeros(shape=(21,2))
        hand_landmarks_right = np.zeros(shape=(21,2))
        pose_landmarks_list = np.zeros(shape=(25,2))
        landmarks_face = np.zeros(shape=(68,2))
        frame_KPs = np.zeros(shape=(KEYPOINTS))
        BGR_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(BGR_Image)
        if results.left_hand_landmarks: 
            hand_landmarks_left = np.array([[lmk.x, lmk.y] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
        else:
            hand_landmarks_left = np.zeros(shape=(21,2))
        if results.right_hand_landmarks:
            hand_landmarks_right = np.array([[lmk.x, lmk.y] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
        else: 
            hand_landmarks_right = np.zeros(shape=(21,2))        
        if results.pose_landmarks:
            pose_landmarks_list = []
            pose_landmarks = results.pose_landmarks
            pose_landmarks_list = np.array([[lmk.x, lmk.y] for i, lmk in enumerate(results.pose_landmarks.landmark) if i < 25], dtype=np.float32)
        else: 
            pose_landmarks_list = np.zeros(shape=(25,2))                            
        if results.face_landmarks:
            landmarks_face = []
            for index in LANDMARK_IDX:
                landmark = results.face_landmarks.landmark[index]    
                landmarks_face.append(landmark.x)
                landmarks_face.append(landmark.y)
        else:
            landmarks_face = np.zeros(shape=(68,2))
            landmarks_face = landmarks_face.flatten().astype(str).tolist()
        frames_video_list_0.append(np.concatenate((hand_landmarks_left.flatten().tolist(),hand_landmarks_right.flatten(),pose_landmarks_list.flatten(),landmarks_face), axis=None)) 
        frameNr = frameNr+1
    while frameNr < FRAMES:
        if len(frames_video_list_0) < FRAMES:
            frames_video_list_0.append(np.zeros(KEYPOINTS))        
        frameNr = frameNr+1
    video_kp_0 = np.stack(frames_video_list_0)
    video_kp_0 = video_kp_0.reshape(-1, FRAMES, KEYPOINTS)
    holistic_model.close()
    return video_kp_0


def get_keypoints_mp_pose_face_PCA(video_path):
    FRAMES = 90
    KEYPOINTS = 202
    frames = crop_video_and_get_frames_90(video_path)
    holistic_model = mp.solutions.holistic.Holistic()
    pca = PCA(n_components=1)
    frames_video_list_0 = []
    frameNr = 0
    for frame in frames:
        hand_landmarks_left = np.zeros(shape=(21,2))
        hand_landmarks_right = np.zeros(shape=(21,2))
        pose_landmarks_list = np.zeros(shape=(25,2))
        landmarks_face = np.zeros(shape=(68))
        frame_KPs = np.zeros(shape=(KEYPOINTS))
        BGR_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(BGR_Image)
        if results.left_hand_landmarks: 
            hand_landmarks_left = np.array([[lmk.x, lmk.y] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
        else:
            hand_landmarks_left = np.zeros(shape=(21,2))
        if results.right_hand_landmarks:
            hand_landmarks_right = np.array([[lmk.x, lmk.y] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
        else: 
            hand_landmarks_right = np.zeros(shape=(21,2))        
        if results.pose_landmarks:
            pose_landmarks_list = []
            pose_landmarks = results.pose_landmarks
            pose_landmarks_list = np.array([[lmk.x, lmk.y] for i, lmk in enumerate(results.pose_landmarks.landmark) if i < 25], dtype=np.float32)
        else: 
            pose_landmarks_list = np.zeros(shape=(25,2))
        if results.face_landmarks:
            landmarks_pca = []
            for index in LANDMARK_IDX:
                landmark = results.face_landmarks.landmark[index]    
                landmarks_pca.append([landmark.x, landmark.y])
            pca.fit(landmarks_pca)
            landmarks_pca = pca.transform(landmarks_pca)
            landmarks_face = landmarks_pca.flatten()
        else:
            landmarks_face = np.zeros(shape=(68))
            landmarks_face = landmarks_face.flatten().astype(str).tolist()
        frames_video_list_0.append(np.concatenate((hand_landmarks_left.flatten().tolist(),hand_landmarks_right.flatten(),pose_landmarks_list.flatten(),landmarks_face), axis=None)) 
        frameNr = frameNr+1
    while frameNr < FRAMES:
        if len(frames_video_list_0) < FRAMES:
            frames_video_list_0.append(np.zeros(KEYPOINTS))        
        frameNr = frameNr+1
    video_kp_0 = np.stack(frames_video_list_0)
    video_kp_0 = video_kp_0.reshape(-1, FRAMES, KEYPOINTS)
    holistic_model.close()
    return video_kp_0



def get_keypoints_mp_pose_face_PCA_30(video_path):
    FRAMES = 30
    KEYPOINTS = 202
    frames = crop_video_and_get_frames_30(video_path)
    holistic_model = mp.solutions.holistic.Holistic()
    pca = PCA(n_components=1)
    frames_video_list_0 = []
    frameNr = 0
    for frame in frames:
        hand_landmarks_left = np.zeros(shape=(21,2))
        hand_landmarks_right = np.zeros(shape=(21,2))
        pose_landmarks_list = np.zeros(shape=(25,2))
        landmarks_face = np.zeros(shape=(68))
        frame_KPs = np.zeros(shape=(KEYPOINTS))
        BGR_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(BGR_Image)
        if results.left_hand_landmarks: 
            hand_landmarks_left = np.array([[lmk.x, lmk.y] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
        else:
            hand_landmarks_left = np.zeros(shape=(21,2))
        if results.right_hand_landmarks:
            hand_landmarks_right = np.array([[lmk.x, lmk.y] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
        else: 
            hand_landmarks_right = np.zeros(shape=(21,2))        
        if results.pose_landmarks:
            pose_landmarks_list = []
            pose_landmarks = results.pose_landmarks
            pose_landmarks_list = np.array([[lmk.x, lmk.y] for i, lmk in enumerate(results.pose_landmarks.landmark) if i < 25], dtype=np.float32)
        else: 
            pose_landmarks_list = np.zeros(shape=(25,2))
        if results.face_landmarks:
            landmarks_pca = []
            for index in LANDMARK_IDX:
                landmark = results.face_landmarks.landmark[index]    
                landmarks_pca.append([landmark.x, landmark.y])
            pca.fit(landmarks_pca)
            landmarks_pca = pca.transform(landmarks_pca)
            landmarks_face = landmarks_pca.flatten()
        else:
            landmarks_face = np.zeros(shape=(68))
            landmarks_face = landmarks_face.flatten().astype(str).tolist()
        frames_video_list_0.append(np.concatenate((hand_landmarks_left.flatten().tolist(),hand_landmarks_right.flatten(),pose_landmarks_list.flatten(),landmarks_face), axis=None)) 
        frameNr = frameNr+1
    while frameNr < FRAMES:
        if len(frames_video_list_0) < FRAMES:
            frames_video_list_0.append(np.zeros(KEYPOINTS))        
        frameNr = frameNr+1
    video_kp_0 = np.stack(frames_video_list_0)
    video_kp_0 = video_kp_0.reshape(-1, FRAMES, KEYPOINTS)
    holistic_model.close()
    return video_kp_0

def get_keypoints_mp_pose_face_30(video_path):
    FRAMES = 30
    KEYPOINTS = 270
    frames = crop_video_and_get_frames_30(video_path)    
    holistic_model = mp.solutions.holistic.Holistic()
    frames_video_list_0 = []
    frameNr = 0
    for frame in frames:
        hand_landmarks_left = np.zeros(shape=(21,2))
        hand_landmarks_right = np.zeros(shape=(21,2))
        pose_landmarks_list = np.zeros(shape=(25,2))
        landmarks_face = np.zeros(shape=(68,2))
        frame_KPs = np.zeros(shape=(KEYPOINTS))
        BGR_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(BGR_Image)
        if results.left_hand_landmarks: 
            hand_landmarks_left = np.array([[lmk.x, lmk.y] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
        else:
            hand_landmarks_left = np.zeros(shape=(21,2))
        if results.right_hand_landmarks:
            hand_landmarks_right = np.array([[lmk.x, lmk.y] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
        else: 
            hand_landmarks_right = np.zeros(shape=(21,2))        
        if results.pose_landmarks:
            pose_landmarks_list = []
            pose_landmarks = results.pose_landmarks
            pose_landmarks_list = np.array([[lmk.x, lmk.y] for i, lmk in enumerate(results.pose_landmarks.landmark) if i < 25], dtype=np.float32)
        else: 
            pose_landmarks_list = np.zeros(shape=(25,2))                            
        if results.face_landmarks:
            landmarks_face = []
            for index in LANDMARK_IDX:
                landmark = results.face_landmarks.landmark[index]    
                landmarks_face.append(landmark.x)
                landmarks_face.append(landmark.y)
        else:
            landmarks_face = np.zeros(shape=(68,2))
            landmarks_face = landmarks_face.flatten().astype(str).tolist()
        frames_video_list_0.append(np.concatenate((hand_landmarks_left.flatten().tolist(),hand_landmarks_right.flatten(),pose_landmarks_list.flatten(),landmarks_face), axis=None)) 
        frameNr = frameNr+1
    while frameNr < FRAMES:
        if len(frames_video_list_0) < FRAMES:
            frames_video_list_0.append(np.zeros(KEYPOINTS))        
        frameNr = frameNr+1
    video_kp_0 = np.stack(frames_video_list_0)
    video_kp_0 = video_kp_0.reshape(-1, FRAMES, KEYPOINTS)
    holistic_model.close()
    return video_kp_0



def get_keypoints_mp_base_30(video_path):
    FRAMES = 30
    KEYPOINTS = 134
    frames = crop_video_and_get_frames_30(video_path)
    holistic_model = mp.solutions.holistic.Holistic()
    frames_video_list_0 = []
    frameNr = 0
    for frame in frames:
        hand_landmarks_left = np.zeros(shape=(21,2))
        hand_landmarks_right = np.zeros(shape=(21,2))
        pose_landmarks_list = np.zeros(shape=(25,2))
        frame_KPs = np.zeros(shape=(KEYPOINTS))
        BGR_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(BGR_Image)
        if results.left_hand_landmarks: 
            hand_landmarks_left = np.array([[lmk.x, lmk.y] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
        else:
            hand_landmarks_left = np.zeros(shape=(21,2))
        if results.right_hand_landmarks:
            hand_landmarks_right = np.array([[lmk.x, lmk.y] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
        else: 
            hand_landmarks_right = np.zeros(shape=(21,2))        
        if results.pose_landmarks:
            pose_landmarks_list = []
            pose_landmarks = results.pose_landmarks
            pose_landmarks_list = np.array([[lmk.x, lmk.y] for i, lmk in enumerate(results.pose_landmarks.landmark) if i < 25], dtype=np.float32)             
        frames_video_list_0.append(np.concatenate((hand_landmarks_left.flatten().tolist(),hand_landmarks_right.flatten(),pose_landmarks_list.flatten()), axis=None)) 
        frameNr = frameNr+1
    while frameNr < FRAMES:
        if len(frames_video_list_0) < FRAMES:
            frames_video_list_0.append(np.zeros(KEYPOINTS))        
        frameNr = frameNr+1
    video_kp_0 = np.stack(frames_video_list_0)
    video_kp_0 = video_kp_0.reshape(-1, FRAMES, KEYPOINTS)
    holistic_model.close()
    return video_kp_0




def get_model_GRU(filepath_weights, FRAMES, KEYPOINTS):
    top2_acc = lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=2)
    model = keras.models.Sequential()
    model.add(Dense(units=1024, input_shape=(FRAMES, KEYPOINTS)))
    model.add(LayerNormalization())
    model.add(Dropout(0.6))
    model.add(GRU(units=2048, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(GRU(units=1024, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(GRU(units=512, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(GRU(units=256, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(GRU(units=128, return_sequences=False))
    model.add(LayerNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(units=20, activation='softmax'))
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy",top2_acc])
    model.load_weights(filepath_weights)
    return model

def get_model_LSTM(filepath_weights, FRAMES, KEYPOINTS):
    top2_acc = lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=2)
    model = keras.models.Sequential()
    model.add(Dense(units=1024, input_shape=(FRAMES, KEYPOINTS)))
    model.add(LayerNormalization())
    model.add(Dropout(0.6))
    model.add(LSTM(units=2048, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(units=1024, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(units=512, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(LayerNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(LayerNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(units=20, activation='softmax'))
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy",top2_acc])
    model.load_weights(filepath_weights)
    return model

def get_best_probs_90(predictions):
    top_probs = np.argsort(predictions.flatten())[::-1][:2]
    return top_probs

def get_class(idx):
    return all_classes[idx]

def predict_from_video_90(model, KPs):
    #vid1, vid2, vid3 = preprocess_video(video)
    predictions_0 = model.predict(KPs, verbose=0)
    #predictions_1 = loaded_model.predict(vid2)
   # predictions_2 = loaded_model.predict(vid3)
    top_probs = get_best_probs_90(predictions_0)
    return top_probs

def select_files_from_subfolders(video_path, N_FILES):
    selected_files = []
    for subfolder in os.listdir(video_path):
        subfolder_path = os.path.join(video_path, subfolder)
        files_in_subfolder = os.listdir(subfolder_path)
        num_files_to_select = min(N_FILES // len(os.listdir(video_path)), len(files_in_subfolder))
        selected_files_from_subfolder = random.sample(files_in_subfolder, num_files_to_select)
        selected_files += [os.path.join(subfolder_path, file) for file in selected_files_from_subfolder]
    return selected_files

def preprocess_video_30_face_PCA(video):
    kp1 = get_keypoints_mp_pose_face_PCA_30(video)
    return kp1

def crop_video_and_get_frames_30(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if num_frames > MAX_FRAMES_VID:
        remove_frames = (num_frames - MAX_FRAMES_VID) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, remove_frames)
        for i in range(MAX_FRAMES_VID):
            ret, frame = cap.read()
            if ret:
                if(i % 3 == 0):
                    frames.append(frame)
        cap.release()
    else:
        for i in range(MAX_FRAMES_VID):
            success, frame = cap.read()
            if not success:
                break
            else:
                if(i % 3 == 0):
                    frames.append(frame)
    return frames


def predict_sign_visual(model, video):
    FRAMES_SAMPLED = 30
    KEYPOINTS_BASELINE_FACE_PCA = 202
    kp1 = get_keypoints_mp_pose_face_PCA_30_visual(video)
    predictions = predict_from_video_90(model, kp1[0].reshape(-1, FRAMES_SAMPLED, KEYPOINTS_BASELINE_FACE_PCA))
    predicted_sign = get_class(predictions[0])
    frames_to_display = kp1[1]
    frames_to_display_white_bg = kp1[2]
    fig, axes = plt.subplots(1, len(frames_to_display), figsize=(15, 3))
    for i, array_data in enumerate(frames_to_display):
        video_name = os.path.basename(video)
        image_path = DEBUG_PATH + video_name[:-4] + f"_{i}.jpg"
        image = Image.fromarray(array_data)
        cv2.imwrite(image_path, array_data)
        resized_image = image.resize((512, 768))
        axes[i].imshow(resized_image)
        axes[i].axis('off')
    plt.text(0.5, 0.0, predicted_sign, ha='center', fontsize=28, transform=fig.transFigure)
    plt.show()
    for i, array_data_w in enumerate(frames_to_display_white_bg):
        white_image_path = os.path.basename(video)
        white_image_path = DEBUG_PATH + video_name[:-4] + f"_white_{i}.jpg"
        cv2.imwrite(white_image_path, array_data_w)
    return get_class(predictions[1]) 

def get_keypoints_mp_pose_face_PCA_30_visual(video_path):
    FRAMES = 30
    KEYPOINTS = 202
    frames = crop_video_and_get_frames_30(video_path)
    holistic_model = mp.solutions.holistic.Holistic()
    pca = PCA(n_components=1)
    frames_video_list_0 = []
    frames_to_display = []
    frames_to_display_white_bg = []
    frameNr = 0
    for frame in frames:
        hand_landmarks_left = np.zeros(shape=(21,2))
        hand_landmarks_right = np.zeros(shape=(21,2))
        pose_landmarks_list = np.zeros(shape=(25,2))
        landmarks_face = np.zeros(shape=(68))
        frame_KPs = np.zeros(shape=(KEYPOINTS))
        BGR_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(BGR_Image)
        if results.left_hand_landmarks: 
            hand_landmarks_left = np.array([[lmk.x, lmk.y] for lmk in results.left_hand_landmarks.landmark], dtype=np.float32)
        else:
            hand_landmarks_left = np.zeros(shape=(21,2))
        if results.right_hand_landmarks:
            hand_landmarks_right = np.array([[lmk.x, lmk.y] for lmk in results.right_hand_landmarks.landmark], dtype=np.float32)
        else: 
            hand_landmarks_right = np.zeros(shape=(21,2))        
        if results.pose_landmarks:
            pose_landmarks_list = []
            pose_landmarks = results.pose_landmarks
            pose_landmarks_list = np.array([[lmk.x, lmk.y] for i, lmk in enumerate(results.pose_landmarks.landmark) if i < 25], dtype=np.float32)
        else: 
            pose_landmarks_list = np.zeros(shape=(25,2))
        if results.face_landmarks:
            landmarks_pca = []
            for index in LANDMARK_IDX:
                landmark = results.face_landmarks.landmark[index]    
                landmarks_pca.append([landmark.x, landmark.y])
            pca.fit(landmarks_pca)
            landmarks_pca = pca.transform(landmarks_pca)
            landmarks_face = landmarks_pca.flatten()
        else:
            landmarks_face = np.zeros(shape=(68))
            landmarks_face = landmarks_face.flatten().astype(str).tolist()
        frames_video_list_0.append(np.concatenate((hand_landmarks_left.flatten().tolist(),hand_landmarks_right.flatten(),pose_landmarks_list.flatten(),landmarks_face), axis=None)) 
        if(frameNr % 5 == 0):
            #image = Image.fromarray(frame)
            image = draw_landmarks(frame, results)
            height, width, _ = frame.shape
            white_image = draw_landmarks(generate_white_bg(height, width), results)
            frames_to_display.append(image)
            frames_to_display_white_bg.append(white_image)
        frameNr = frameNr+1
    while frameNr < FRAMES:
        if len(frames_video_list_0) < FRAMES:
            frames_video_list_0.append(np.zeros(KEYPOINTS))        
        frameNr = frameNr+1
    video_kp_0 = np.stack(frames_video_list_0)
    video_kp_0 = video_kp_0.reshape(-1, FRAMES, KEYPOINTS)
    holistic_model.close()
    return video_kp_0, frames_to_display, frames_to_display_white_bg

def draw_landmarks(image, results):
    annotated_image = image.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style())
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style())
    if results.face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,  results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    return annotated_image

def generate_white_bg(h, w):
    channels = 3 
    white_image = np.full((h, w, channels), 255, dtype=np.uint8)
    return white_image