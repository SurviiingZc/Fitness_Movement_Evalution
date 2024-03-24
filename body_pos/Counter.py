import  numpy as np

def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle

class Evaluate():
    def __init__(self):
        self. flag = 0
        self. sum_score = 0
        self. clear()

    def clear(self):
        self. score = [0, 0]

    def Angle_calc(self, keypoints):

        keypoints = np.array(keypoints)

        v1 = keypoints[5] - keypoints[6]
        v2 = keypoints[8] - keypoints[6]
        self. angle_right_arm = get_angle(v1, v2)

        # 计算左臂与水平方向的夹角
        v1 = keypoints[7] - keypoints[5]
        v2 = keypoints[6] - keypoints[5]
        self. angle_left_arm = get_angle(v1, v2)

        # 计算右肘的夹角
        v1 = keypoints[6] - keypoints[8]
        v2 = keypoints[10] - keypoints[8]
        self. angle_right_elbow = get_angle(v1, v2)

        # 计算左肘的夹角
        v1 = keypoints[5] - keypoints[7]
        v2 = keypoints[9] - keypoints[7]
        self. angle_left_elbow = get_angle(v1, v2)

        # 计算左大腿和左臂夹角
        v1 = keypoints[13] - keypoints[11]
        v2 = keypoints[7] - keypoints[5]
        self. angle_left_leg = get_angle(v1, v2)

        # 计算右大腿和右臂夹角
        v1 = keypoints[14] - keypoints[12]
        v2 = keypoints[8] - keypoints[6]
        self. angle_right_leg = get_angle(v1, v2)

        # 计算左大腿和左小腿夹角
        v1 = keypoints[11] - keypoints[13]
        v2 = keypoints[15] - keypoints[13]
        self. angle_left_knee = get_angle(v1, v2)

        # 推肩条件
        self. shoulder_push_begin = (self. angle_right_leg > -90 and self. angle_left_leg < 90)
        self. shoulder_push_finish = (self. angle_right_leg < -150 and self. angle_left_leg > 150)
        # 飞鸟条件
        self. flying_bird_begin = (self. angle_right_leg > -30 and self. angle_left_leg < 30)
        self. flying_bird_finish = (self. angle_right_leg < -60 and self. angle_left_leg > 60)
        # 深蹲条件
        self. squat_begin = (self. angle_left_knee < -120 or self. angle_left_knee > 0)
        self. squat_finish = (self. angle_left_knee > -70 and self. angle_left_knee < 0)
        # 二头弯举条件
        self. bend_begin = (self. angle_left_elbow < 180 and self. angle_left_elbow > 150)
        self. bend_finish = (self. angle_left_elbow < 45 and self. angle_left_elbow > 0)

    def Evaluate_update(self, type):
        Counter = 0
        # 分类计数
        if (type == "Shoulder_Push"):
            if (self. shoulder_push_begin):
                if(not self.flag):
                    self. sum_score += self. score[1]
                    self.clear()
                self. flag = 1
            elif (self. shoulder_push_finish ):
                self.Shoulder_Push_score()
                if (self.flag):
                    Counter = 1
                    self.flag = 0

        elif (type == "Flying_Bird"):
            if (self. flying_bird_begin):
                if(not self. flag):
                    self. sum_score += self. score[1]
                    self.clear()
                self. flag = 1
            elif (self. flying_bird_finish ):
                self. Flying_Bird_score()
                if(self. flag):
                    Counter = 1
                    self. flag = 0

        elif (type == "Squat"):
            if (self. squat_begin):
                if (not self.flag):
                    self.sum_score += self.score[1]
                    self.clear()
                self.flag = 1
            elif (self. squat_finish):
                self. Squat_score()
                if(self. flag):
                    Counter = 1
                    self. flag = 0

        elif (type == "Bend"):
            if (self. bend_begin):
                self. flag = 1
            elif (self. bend_finish and self. flag):
                Counter = 1
                self. flag = 0

        return Counter

    def Shoulder_Push_score(self):
        if(self.angle_left_leg == 180 or self.angle_left_leg < 0):
            self. score[0] = 180
        elif(self.angle_left_leg > self. score[0]):
            self.score[0] = self.angle_left_leg
        self. score[1] = 100 - (170 - self. score[0]) * 2

    def Flying_Bird_score(self):
        if(self.angle_left_leg > 90):
            self. score[0] = 90
        elif(self.angle_left_leg > self. score[0]):
            self.score[0] = self.angle_left_leg
        self. score[1] = 100 - (90 - self. score[0]) * (4/3)

    def Squat_score(self):
        #score[0]初值为0对某些角度判定有影响
        if(self.score[0] == 0):
            self.score[0] = self.angle_left_knee
        elif(self.angle_left_knee > -20):
            self. score[0] = -20
        elif(self.angle_left_knee > self. score[0]):
            self.score[0] = self.angle_left_knee
        self. score[1] = 100 - (-20 - self. score[0]) * (4/5)


#
# def train_counter(keypoints, type):
#     global flag
#     keypoints = np.array(keypoints)
#     Counter = 0
#     v1 = keypoints[5] - keypoints[6]
#     v2 = keypoints[8] - keypoints[6]
#     angle_right_arm = get_angle(v1, v2)
#
#     # 计算左臂与水平方向的夹角
#     v1 = keypoints[7] - keypoints[5]
#     v2 = keypoints[6] - keypoints[5]
#     angle_left_arm = get_angle(v1, v2)
#
#     # 计算右肘的夹角
#     v1 = keypoints[6] - keypoints[8]
#     v2 = keypoints[10] - keypoints[8]
#     angle_right_elbow = get_angle(v1, v2)
#
#     # 计算左肘的夹角
#     v1 = keypoints[5] - keypoints[7]
#     v2 = keypoints[9] - keypoints[7]
#     angle_left_elbow = get_angle(v1, v2)
#
#     #计算左大腿和左臂夹角
#     v1 = keypoints[13] - keypoints[11]
#     v2 = keypoints[7] - keypoints[5]
#     angle_left_leg = get_angle(v1, v2)
#
#     #计算右大腿和右臂夹角
#     v1 = keypoints[14] - keypoints[12]
#     v2 = keypoints[8] - keypoints[6]
#     angle_right_leg = get_angle(v1, v2)
#
#     #计算左大腿和左小腿夹角
#     v1 = keypoints[11] - keypoints[13]
#     v2 = keypoints[15] - keypoints[13]
#     angle_left_knee = get_angle(v1, v2)
#
#     #推肩条件
#     shoulder_push_begin= (angle_right_leg>-90 and angle_left_leg<90)
#     shoulder_push_finish = (angle_right_leg<-150 and angle_left_leg>150)
#     #飞鸟条件
#     flying_bird_begin = (angle_right_leg>-30 and angle_left_leg<30)
#     flying_bird_finish = (angle_right_leg<-60 and angle_left_leg>60)
#     #深蹲条件
#     squat_begin = (angle_left_knee<-120 or angle_left_knee>0)
#     squat_finish = (angle_left_knee>-70 and angle_left_knee<0)
#     #二头弯举条件
#     bend_begin = (angle_left_elbow<180 and angle_left_elbow>150)
#     bend_finish = (angle_left_elbow<45 and angle_left_elbow>0)
#
#     #分类计数
#     if(type == "Shoulder_Push"):
#         if( shoulder_push_begin):
#             flag = 1
#         elif( shoulder_push_finish and flag):
#             Counter = 1
#             flag = 0
#     elif(type == "Flying_Bird"):
#         if(flying_bird_begin):
#              flag = 1
#         elif(flying_bird_finish and flag):
#             Counter  = 1
#             flag = 0
#     elif(type == "Squat"):
#         if(squat_begin):
#             flag = 1
#         elif(squat_finish and flag):
#             Counter = 1
#             flag = 0
#     elif(type == "Bend"):
#         if(bend_begin):
#             flag = 1
#         elif(bend_finish):
#             Counter = 1
#             flag = 0
#
#     return Counter





