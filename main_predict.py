from config import *
from utils import *
from preprocess import *
from eda import *
from model import *
from training import *
from solve_sudoku import *

pathImage = IMG_PATH
heightImg = IMG_HEIGHT
widthImg = IMG_WIDTH
pathImage

#### PREPARE THE IMAGE
img = cv2.imread(pathImage)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)

#### FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

#### FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
print(biggest)

if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)

else:
    print("No Sudoku Found")

cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
pts1 = np.float32(biggest)
pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
imgWarpGrayed = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
imgDetectedDigits = imgBlank.copy()

#### SPLIT THE WARPED SUDOKU IMAGE AND PREDICT THE DIGITS
imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpGrayed)
print(boxes[0].shape) # := (50, 50)

numbers = getPrediction(boxes, model)
print(numbers)

imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
numbers = np.asarray(numbers)
posList = np.where(numbers > 0, 0, 1) # Set bit at blank spaces 

#### SOLVE
board = np.reshape(numbers, (9, 9))
print(board)

if(solveSudoku(board)):
    solvedBoard = board
else:
    print("Error")
print(solvedBoard)

solvedBoardFlattened = []
for subarr in solvedBoard:
    for item in subarr:
        solvedBoardFlattened.append(item)

solvedNumbers = solvedBoardFlattened*posList
imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers, (0, 255, 0))

#### OVERLAYING THE SOLUTION
pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
pts2 = np.float32(biggest)

imgInvWarped = img.copy()
invMatrix = cv2.getPerspectiveTransform(pts1, pts2)
imgInvWarped = cv2.warpPerspective(imgSolvedDigits, invMatrix, (widthImg, heightImg))
imgOverlayed = cv2.addWeighted(imgInvWarped, 1, img, 0.4, 1)

plt.imshow(imgOverlayed)

