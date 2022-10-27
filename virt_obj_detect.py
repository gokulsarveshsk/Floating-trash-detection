import cv2
# Always import cv2 at the top
import time
import commons as cm
from threading import Thread

PATH_TO_MODEL = "C:/Users/Kabilan/tensorflow/workspace/output/saved_model"  # TODO
PATH_TO_LABELS = 'C:/Users/Kabilan/tensorflow/workspace/labels.txt'

tolerance = 0.1
x_deviation = 0
y_deviation = 0
threshold = 0.2
top_k = 5
arr_track_data = [0, 0, 0, 0, 0, 0]

arr_valid_objects = ['waste']

cap = cv2.VideoCapture(1)


detect_fn, labels = cm.load_model(PATH_TO_MODEL, PATH_TO_LABELS)


def track_object(objs, labels):

    # global delay
    global x_deviation, y_deviation, tolerance, arr_track_data

    if (len(objs) == 0):
        print("no objects to track")
        print("stop")
        arr_track_data = [0, 0, 0, 0, 0, 0]
        return

    # ut.head_lights("OFF")
    k = 0
    flag = 0
    for obj in objs:
        lbl = labels.get(obj.id, obj.id)
        k = arr_valid_objects.count(lbl)
        if (k > 0):
            x_min, y_min, x_max, y_max = list(obj.bbox)
            flag = 1
            break

    # print(x_min, y_min, x_max, y_max)
    if (flag == 0):
        print("selected object no present")
        return

    x_diff = x_max-x_min
    y_diff = y_max-y_min
    print("x_diff: ", round(x_diff, 5))
    print("y_diff: ", round(y_diff, 5))

    obj_x_center = x_min+(x_diff/2)
    obj_x_center = round(obj_x_center, 3)

    obj_y_center = y_min+(y_diff/2)
    obj_y_center = round(obj_y_center, 3)

    # print("[",obj_x_center, obj_y_center,"]")

    x_deviation = round(0.5-obj_x_center, 3)
    y_deviation = round(0.5-obj_y_center, 3)

    print("{", x_deviation, y_deviation, "}")

    # move_robot()
    thread = Thread(target=move_robot)
    thread.start()
    # thread.join()

    # print(cmd)

    arr_track_data[0] = obj_x_center
    arr_track_data[1] = obj_y_center
    arr_track_data[2] = x_deviation
    arr_track_data[3] = y_deviation


def move_robot():
    global x_deviation, y_deviation, tolerance, arr_track_data

    print("moving robot .............!!!!!!!!!!!!!!")
    print(x_deviation, y_deviation, tolerance, arr_track_data)

    if (abs(x_deviation) < tolerance and abs(y_deviation) < tolerance):
        cmd = "Stop"
        # delay1 = 0
        print(cmd)

    else:
        if (abs(x_deviation) > abs(y_deviation)):
            if (x_deviation >= tolerance):
                cmd = "Move Left"
                # delay1 = get_delay(x_deviation, 'l')

                print(cmd)
                # time.sleep(delay1)
                print("stop")

            if (x_deviation <= -1*tolerance):
                cmd = "Move Right"
                # delay1 = get_delay(x_deviation, 'r')

                print(cmd)
                # time.sleep(delay1)
                print("stop")
        else:

            if (y_deviation >= tolerance):
                cmd = "Move Forward"
                # delay1 = get_delay(y_deviation, 'f')

                print("forward")
                # time.sleep(delay1)
                print("stop")

            if (y_deviation <= -1*tolerance):
                cmd = "Move Backward"
                # delay1 = get_delay(y_deviation, 'b')

                print(cmd)
                # time.sleep(delay1)
                print("stop")

    arr_track_data[4] = cmd
    arr_track_data[5] = delay1


# def get_delay(deviation, direction):
#     deviation = abs(deviation)
#     if (direction == 'f' or direction == 'b'):
#         if (deviation >= 0.3):
#             d = 0.1
#         elif (deviation >= 0.2 and deviation < 0.30):
#             d = 0.075
#         elif (deviation >= 0.15 and deviation < 0.2):
#             d = 0.045
#         else:
#             d = 0.035
#     else:
#         if (deviation >= 0.4):
#             d = 0.080
#         elif (deviation >= 0.35 and deviation < 0.40):
#             d = 0.070
#         elif (deviation >= 0.30 and deviation < 0.35):
#             d = 0.060
#         elif (deviation >= 0.25 and deviation < 0.30):
#             d = 0.050
#         elif (deviation >= 0.20 and deviation < 0.25):
#             d = 0.040
#         else:
#             d = 0.030

#     return d


def main():

    arr_dur = [0, 0, 0]
    # while cap.isOpened():
    while True:
        # ----------------Capture Camera Frame-----------------
        start_t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        arr_dur[0] = time.time() - start_t0
        # ----------------------------------------------------

        # -------------------Inference---------------------------------
        start_t1 = time.time()
        detections = cm.predictions(detect_fn, frame, (640, 640))

        objs = cm.get_output(
            detections=detections, score_threshold=threshold, top_k=top_k)
        # print(objs)

        arr_dur[1] = time.time() - start_t1
        # cm.time_elapsed(start_t1,"inference")
        # ----------------------------------------------------

        # -----------------other------------------------------------
        start_t2 = time.time()
        track_object(objs, labels)  # tracking  <<<<<<<

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = draw_overlays(frame, objs, labels, arr_dur, arr_track_data)
        cv2.imshow('Object Tracking - TensosrFlow Lite', frame)
        end_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


def draw_overlays(cv2_im, objs, labels, arr_dur, arr_track_data):
    height, width, channels = cv2_im.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    global tolerance

    # draw black rectangle on top
    cv2_im = cv2.rectangle(cv2_im, (0, 0), (width, 24), (0, 0, 0), -1)

    # write processing durations
    cam = round(arr_dur[0]*1000, 0)
    inference = round(arr_dur[1]*1000, 0)
    other = round(arr_dur[2]*1000, 0)
    text_dur = 'Camera: {}ms   Inference: {}ms   other: {}ms'.format(
        cam, inference, other)
    cv2_im = cv2.putText(cv2_im, text_dur, (int(
        width/4)-30, 16), font, 0.4, (255, 255, 255), 1)

    # write FPS
    total_duration = cam+inference+other
    fps = round(1000/total_duration, 1)
    text1 = 'FPS: {}'.format(fps)
    cv2_im = cv2.putText(cv2_im, text1, (10, 20),
                         font, 0.7, (150, 150, 255), 2)

    # draw black rectangle at bottom
    cv2_im = cv2.rectangle(cv2_im, (0, height-24),
                           (width, height), (0, 0, 0), -1)

    # write deviations and tolerance
    str_tol = 'Tol : {}'.format(tolerance)
    cv2_im = cv2.putText(cv2_im, str_tol, (10, height-8),
                         font, 0.55, (150, 150, 255), 2)

    x_dev = arr_track_data[2]
    str_x = 'X: {}'.format(x_dev)
    if (abs(x_dev) < tolerance):
        color_x = (0, 255, 0)
    else:
        color_x = (0, 0, 255)
    cv2_im = cv2.putText(cv2_im, str_x, (110, height-8),
                         font, 0.55, color_x, 2)

    y_dev = arr_track_data[3]
    str_y = 'Y: {}'.format(y_dev)
    if (abs(y_dev) < tolerance):
        color_y = (0, 255, 0)
    else:
        color_y = (0, 0, 255)
    cv2_im = cv2.putText(cv2_im, str_y, (220, height-8),
                         font, 0.55, color_y, 2)

    # write direction, speed, tracking status
    cmd = arr_track_data[4]
    cv2_im = cv2.putText(cv2_im, str(cmd), (int(
        width/2) + 10, height-8), font, 0.68, (0, 255, 255), 2)

    delay1 = arr_track_data[5]
    str_sp = 'Speed: {}%'.format(round(delay1/(0.1)*100, 1))
    cv2_im = cv2.putText(cv2_im, str_sp, (int(width/2) +
                         185, height-8), font, 0.55, (150, 150, 255), 2)

    if (cmd == 0):
        str1 = "No object"
    elif (cmd == 'Stop'):
        str1 = 'Acquired'
    else:
        str1 = 'Tracking'
    cv2_im = cv2.putText(cv2_im, str1, (width-140, 18),
                         font, 0.7, (0, 255, 255), 2)

    # draw center cross lines
    cv2_im = cv2.rectangle(cv2_im, (0, int(height/2)-1),
                           (width, int(height/2)+1), (255, 0, 0), -1)
    cv2_im = cv2.rectangle(cv2_im, (int(width/2)-1, 0),
                           (int(width/2)+1, height), (255, 0, 0), -1)

    # draw the center red dot on the object
    cv2_im = cv2.circle(cv2_im, (int(
        arr_track_data[0]*width), int(arr_track_data[1]*height)), 7, (0, 0, 255), -1)

    # draw the tolerance box
    cv2_im = cv2.rectangle(cv2_im, (int(width/2-tolerance*width), int(height/2-tolerance*height)),
                           (int(width/2+tolerance*width), int(height/2+tolerance*height)), (0, 255, 0), 2)

    # draw bounding boxes
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(
            x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)

        box_color, text_color, thickness = (0, 150, 255), (0, 255, 0), 2
        cv2_im = cv2.rectangle(
            cv2_im, (x0, y0), (x1, y1), box_color, thickness)

        # text3 = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        # cv2_im = cv2.putText(cv2_im, text3, (x0, y1-5),font, 0.5, text_color, thickness)

    return cv2_im


main()
