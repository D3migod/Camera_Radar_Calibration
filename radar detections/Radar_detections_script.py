from __future__ import print_function
try:
    from yaml import CLoader as Loader
    print('using nice and fast CLoader')
except ImportError:
    from yaml import Loader
    print('using ugly and slow-as-fuck Loader')
import cv2
import math

import yaml


import gzip
import os.path
import numpy as np
import pandas as pd
import os
import os.path


def calculateCartesianCoordinates(coordinate):
    radius, azimuth = coordinate[0], coordinate[1]
    return np.array([radius * math.sin(math.radians(azimuth)), radius * math.cos(math.radians(azimuth)), 0])

def convertToCartesian(coordinates):
    return np.array(list(map(calculateCartesianCoordinates, coordinates)))


def ungzip(yaml_in_directory):
    # наши ямлы требуют небольшой ректификации перед использованием
    ungzipped = gzip.open(yaml_in_directory, 'rt')
    ungzipped.readline()
    ungzipped = ungzipped.read()
    ungzipped = ungzipped.replace(':', ': ')

    # собственно парсинг
    yml = yaml.load(ungzipped, Loader=Loader)
    return yml

def read_image_grabmsecs(yml_path):
    yml_data = ungzip(yml_path)
    image_frames = [sh['leftImage']
                    for sh in yml_data['shots'] if 'leftImage' in sh.keys()]
    
    data = np.zeros(shape=(len(image_frames), 1), dtype=int)
    i_real = 0
    for i_fr in image_frames:
        data[i_real, 0] = int(i_fr['grabMsec'])
        i_real += 1

    data = data[:i_real, :]
    return data




# Reading yml data

# Сhange this to where data is stored

    
def read_yml(yml_path):
    yml = yaml.load(yml_path, Loader=Loader)
    return yml

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    dt = np.dtype(mapping["dt"])
    mat = np.array(mapping["data"], dt)
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


# Creating Camera object for central projection 

class Camera:
    '''Pinhole camera projection'''
    def __init__(self, r, t, K, size):
        if  not (t.shape == (3,) and r.shape == (3,) and K.shape == (3,3)):
            raise ValueError('bad constructor args')
        self.r = r
        self.t = t
        self.K = K
        self.size = size
        R = rot_matrix(r)
        self.P = K.dot(np.hstack([R, t.reshape(t.shape[0], 1)]))
        pass
    
    def translate(self, points, t):
        return np.array([point + t for point in points])
    
    def rotate(self, points, rot_mat):
        return np.array([rot_mat.dot(point) for point in points])
    
    def transform(self, points):
        return self.rotate(points, rot_matrix([np.pi / 2.0, 0, 0]))
    
    def project(self, points):
        if points.shape[1] == 3:
            R = rot_matrix(r)
            translated = self.translate(points, self.t)
#             print("Translated: ", translated)
            rotated = self.rotate(translated, R.T)
#             print("Rotated: ", rotated)
            tranformed = self.transform(rotated)
#             print("Transformed: ", tranformed)
            homo_pts = self.K.dot(tranformed.T).T
#             print("homo_pts: ", homo_pts)
        elif points.shape[1] == 4:
            homo_pts = self.P.dot(points.T)
        else:
            raise ValueError('Incorrect points size for Camera.project: %s' % \
                             str(points.shape))
        return homo_pts[:, : 2] / homo_pts[:, 2].reshape(-1, 1)
    
def rot_matrix(angles):
    cos = [np.math.cos(a) for a in angles]
    sin = [np.math.sin(a) for a in angles]
    Rz = np.array([[cos[2], -sin[2], 0],
                   [sin[2], cos[2],  0],
                   [     0,      0,  1]])

    Ry = np.array([[ cos[1], 0, sin[1]],
                   [      0, 1,      0],
                   [-sin[1], 0, cos[1]]])

    Rx = np.array([[1,      0,       0],
                   [0, cos[0], -sin[0]],
                   [0, sin[0], cos[0]]])
    #Rs = [Rx, Ry, Rz]
    return Ry.dot(Rx).dot(Rz)
    


def frange(x, y, jump):
    while x < y:
        yield round(x, 1)
        x += jump

# UI sliders callback functions
def showText(frame):
    cv2.putText(frame, "x: " + str(round(cam.t[0], 3)) + " m", 
    topLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
    cv2.putText(frame, "y: " + str(round(cam.t[1], 3)) + " m", 
    (topLeftCornerOfText[0], topLeftCornerOfText[1]+1*diffPosition), 
    font, 
    fontScale,
    fontColor,
    lineType)
    cv2.putText(frame, "z: " + str(round(cam.t[2], 3)) + " m", 
    (topLeftCornerOfText[0], topLeftCornerOfText[1]+2*diffPosition), 
    font, 
    fontScale,
    fontColor,
    lineType)
    cv2.putText(frame, "Rx: " + str(round(cam.r[0], 3)) + " m", 
    (topLeftCornerOfText[0], topLeftCornerOfText[1]+3*diffPosition), 
    font, 
    fontScale,
    fontColor,
    lineType)
    cv2.putText(frame, "Ry: " + str(round(cam.r[1], 3)) + " m", 
    (topLeftCornerOfText[0], topLeftCornerOfText[1]+4*diffPosition), 
    font, 
    fontScale,
    fontColor,
    lineType)
    cv2.putText(frame, "Rz: " + str(round(cam.r[2], 3)) + " m", 
    (topLeftCornerOfText[0], topLeftCornerOfText[1]+5*diffPosition), 
    font, 
    fontScale,
    fontColor,
    lineType)

def changeTX(x):
    cam.t[0] = -(x - 10000)/1000.0
    
def changeTY(y):
    cam.t[1] = -(y - 10000)/1000.0
    
def changeTZ(z):
    cam.t[2] = -(z - 10000)/1000.0
    
def changeRX(x):
    cam.r[0] = (x - 10000)/1000.0/100.0
    
def changeRY(y):
    cam.r[1] = (y - 10000)/1000.0/100.0
    
def changeRZ(z):
    cam.r[2] = (z - 10000)/1000.0/100.0

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Reading a single tsv file by path
def read_table(path):
    try:
        return pd.read_csv(path, delimiter='\t', skiprows=[1])
    except pd.errors.EmptyDataError as ede:
        return []

# Reading all tsv files in a folder
def read_tables(folder):
    """reads all tables found in directory and concatenates them into one big
    table
    """
    common_table = pd.DataFrame()
    empty_files = []
    nparsed = 0

    for root, dirs, files in os.walk(folder):
        for f in [f for f in sorted(files) if os.path.splitext(f)[-1] == '.tsv']:
            # print('scanning file', f, '...', end=' ')
            try:
                # Основной вызов в чтении. delimiter='\t', чтобы считать табы разделителями,
                # skiprows=[1], чтобы пропустить строку с типами данных.
                table = pd.read_csv(os.path.join(root, f), delimiter='\t', skiprows=[1])
            except pd.errors.EmptyDataError as ede:
                # print('Empty file!!')
                empty_files.append(f)
            finally:
                # print(table.shape)
                common_table = common_table.append(table)

            nparsed += 1

    print('parsed', nparsed, 'files. Rows: ', common_table.shape[0], ' empty files number:', len(empty_files))
    print('empty files list: [', ', '.join(empty_files), ']\n')
    
    # чтобы установить сквозную нумерацию во всех прочитанных таблицах
    common_table = common_table.reset_index(drop=True)
    
    return common_table

if __name__ == '__main__':
    cv2.destroyAllWindows()
    # Reading yml.gz data
    # Сhange this to where data is stored
    data_dir = '/Users/bulatgaliev/GitHub/Masters_diploma/radar_calib_data'
    #'/prun/mipt/student_nirs/2018/galiev_bulat/radar_calib_data'
    # data_dir = '../../radar_calib_data'
    
    # Read grabmsecs
    grabmsecs = np.concatenate((read_image_grabmsecs(os.path.join(data_dir, 't24.305.026.info.yml.gz')).flatten(),
                      read_image_grabmsecs(os.path.join(data_dir, 't24.305.027.info.yml.gz')).flatten(),
                      read_image_grabmsecs(os.path.join(data_dir, 't24.305.028.info.yml.gz')).flatten(),
                      read_image_grabmsecs(os.path.join(data_dir, 't24.305.029.info.yml.gz')).flatten()))

    # Read yml
    camera_path = os.path.join(data_dir, 'calibs', 'camera.yml')
    stream = open(camera_path, 'r')
    stream.readline()
    stream = stream.read()
    stream = stream.replace(':', ': ')
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
    data = yaml.load(stream)

    #Create camera object
    K = data["K"]
    r = data["r"].flatten()
    t = data["t"].flatten()
    img_size = (data["sz"][0], data["sz"][1])
    print('K = ', K)
    print('r = ', r)
    print('t = ', t)
    print('Image size = ', img_size)
    cam = Camera(r=r, t=-t, K=K, size=img_size)

    # Create grid points
    cartesianGridPoints = []
    for x in frange(-10, 10, 0.2):
        if float(x).is_integer():
            for y in frange(0, 10, 0.2):
                cartesianGridPoints.append([x, y, 0])
        else:
            for y in frange(0, 10, 1):
                cartesianGridPoints.append([x, y, 0])
    gridPoints = cam.project(np.array(cartesianGridPoints))

    # Change this to where data is stored
    radar_data_dir = os.path.join(data_dir, 'radar_detections')

    # Change this to where data is stored
    video_dir = data_dir

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText    = (10,20)
    fontScale              = 0.5
    fontColor              = (0,0,0)
    lineType               = 2
    diffPosition           = 25


    # table = read_table(os.path.join(radar_data_dir, 't24.305.028.tsv')) # Reading t24.305.028.tsv radar points only
    table = read_tables(radar_data_dir) # Reading all points # TODO: Fix reading data
    selectedDataTable = table.iloc[:, [1, 3, 4]]
    uniqueDataTable = selectedDataTable.grabMsec.unique()
    
    # Read camera detections
    camera_data_dir = os.path.join(data_dir, 'camera_detections')
    
    # camera_table = read_table(os.path.join(camera_data_dir, 't24.305.028.left.avi.tsv'))
    camera_table = read_tables(camera_data_dir)
    # Read FrameNumber, x, y, w, h
    selectedCameraDataTable = camera_table.iloc[:, [2, 4, 5, 6, 7]]
    totalFrameNumber = 0 
    isFirstZero = False
    for index, row in selectedCameraDataTable.iterrows():
#         print(row)
        if row['FrameNumber'] == 0:
            if isFirstZero:
                print(currentValue)
                totalFrameNumber += currentValue
                isFirstZero = False
        elif not isFirstZero:
            isFirstZero = True    
        currentValue = row['FrameNumber']
        selectedCameraDataTable.at[index, 'FrameNumber'] = totalFrameNumber + currentValue
    
    cap = cv2.VideoCapture(os.path.join(video_dir, 't24.305.026-029.left.mov'))
    # Creating UI
    cv2.namedWindow('frame')
    cv2.createTrackbar('t_x', 'frame', 10159, 20000, changeTX)
    cv2.createTrackbar('t_y', 'frame', 8420, 20000, changeTY)
    cv2.createTrackbar('t_z', 'frame', 11257, 20000, changeTZ)
    cv2.createTrackbar('r_x', 'frame', 14164, 20000, changeRX)
    cv2.createTrackbar('r_y', 'frame', 7388, 20000, changeRY)
    cv2.createTrackbar('r_z', 'frame', 7797, 20000, changeRZ)
    
    pause = False
    # Change to what suits your system more
    key_space = 32
    
    # Current index of an element from yml.gz's grabmsec list
    grabmsecsIndex = 0
    # Current index of an element from radar tsv's grabmsec list
    radarDetectionsIndex = 0
    frame = None
    
    while(True):
        if not pause:
            ret, frame = cap.read()
            # Checking ret causes video to stop in the end
            #if ret is False:
            #    break
            
        if frame is None: # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            grabmsecsIndex = 0
            radarDetectionsIndex = 0
            continue
        if not pause:
            frame = rescale_frame(frame, percent=50)
        if grabmsecsIndex >= len(grabmsecs):
            # There are no more frames to show
            continue
            
        # Calculate the radar grabmsec index closest to the camera grabmsec index
        while radarDetectionsIndex+1 < len(uniqueDataTable) and uniqueDataTable[radarDetectionsIndex+1] < grabmsecs[grabmsecsIndex]:
            radarDetectionsIndex += 1
            
        if radarDetectionsIndex+1 >= len(uniqueDataTable):
            # There are no more radar detections to show. 
            grabmsecsIndex += 1
            # Show frame without any detections and continue
            cv2.imshow('frame', frame)
            continue
        # Get current grabmsecs of both camera and radar
        currentGrabmsec = grabmsecs[grabmsecsIndex]
        
        currentFrameDetections = selectedDataTable.loc[selectedDataTable['grabMsec'] == uniqueDataTable[radarDetectionsIndex]]
        cartesianPoints = convertToCartesian(currentFrameDetections.values[:,-2:])
        points = cam.project(cartesianPoints)
        
        # Camera detections
        # WARNING: FRAME NUMBER IS THE SAME IN ALL FILES 026, 027, 028 avi tsv
        
        
        currentFrameCameraDetections = selectedCameraDataTable.loc[selectedCameraDataTable['FrameNumber'] == grabmsecsIndex]
        tmp = currentFrameCameraDetections
        # To match radar detections we must take bottom-middle
        # (i.e. x + 0.5w, y+h ) point of camera detections!
        cartesianCameraPoints = np.hstack([(tmp.loc[:, 'x'] + 0.5 * tmp.loc[:, 'w']).as_matrix().reshape(-1, 1),
                                           (tmp.loc[:, 'y'] +       tmp.loc[:, 'h']).as_matrix().reshape(-1, 1), 
                                            np.zeros(shape=(tmp.shape[0], 1), dtype=np.float32)]
                                         )
        frame_todraw = frame.copy() #don't draw on original frame
        
        # Grid points should rotate with the Camera!
        gridPoints = cam.project(np.array(cartesianGridPoints))


        for point in gridPoints:
            x, y = int(point[0]*0.5), int(point[1]*0.5)
            if x < frame.shape[1] and y < frame.shape[0] and x > 0 and y > 0:
                cv2.circle(frame_todraw, (x, y), 2, (0,0,0), -1)
                
        for point in points:
            x, y = int(point[0]*0.5), int(point[1]*0.5)
            if x < frame.shape[1] and y < frame.shape[0] and x > 0 and y > 0:
                cv2.circle(frame_todraw, (x, y), 5, (0,255,0), -1)
                
        for point in cartesianCameraPoints:
            x, y = int(point[0]*0.5), int(point[1]*0.5)
            if x < frame.shape[1] and y < frame.shape[0] and x > 0 and y > 0:
                cv2.circle(frame_todraw, (x, y), 5, (193,0,32), -1)
        
        showText(frame_todraw)
        # Show frame with detections
        cv2.imshow('frame', frame_todraw)
        
        if not pause:
            # Go to the next camera grabmsec value
            grabmsecsIndex += 1
        # Checking if the user pressed q key to exit the video
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Press 'Q' key to exit
            break
        if key == key_space:
            pause = not pause
        
            
    # Destroing the video window
    cv2.destroyAllWindows()

