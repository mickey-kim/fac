#
오픈소스 library중 face-recognition, age-gender-estimation, Emotion-recognition 을 함께 사용할수 있도록 개발함.

#face-recognition 부분 참조
https://github.com/ukayzm/opencv/tree/master/face_recognition

#age-gender-estimation 부분 참조
https://github.com/yu4u/age-gender-estimation

#Emotion-recognition 
https://github.com/omar178/Emotion-recognition

#age-gender faster model 
https://github.com/Tony607/Keras_age_gender

각 기능별 redis로 카메라 인풋과 아웃풋을 관리

test_without_cam.py -> 기존 디렉토리에서 np.ndarray뽑아내던게 오래걸려서 redis에 업로드해놧다가 불러오는형식. 새로사람 추가될시 redis업로드
