import cv2

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

marker = cv2.aruco.generateImageMarker(
    dictionary,
    23,
    1000
)

cv2.imwrite("aruco_marker.png", marker)

print("Marker generated: aruco_marker.png")