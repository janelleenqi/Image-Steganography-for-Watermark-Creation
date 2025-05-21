import cv2 # OpenCV
import numpy as np # Arrays (1D, 2D, and matrices)
import matplotlib.pyplot as plt # Plots
import math
import os




class WatermarkProcessor:
    def __init__(self, carrierImgPath, watermarkImgPath):
        carrierImage = cv2.imread(carrierImgPath) #3 channel rgb image
        watermarkImage = cv2.imread(watermarkImgPath) #3 channel rgb image

        self.carrierImg = carrierImage
        self.carrierImgHeight, self.carrierImgWidth = None, None #initialised in preprocessing

        self.watermarkImg = watermarkImage  
        self.watermarkImgHeight, self.watermarkImgWidth = None, None #initialised in preprocessing

    
    def save_img(self, imagePath, imageDesc):
        cv2.imwrite(os.path.join(imagePath, str(f"{imageDesc}.png")), self.carrierImg)


    @staticmethod
    def matplot_visualise_img(img, title): # self.carrierImg or self.watermarkImg
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(title)
    
    @staticmethod
    def matplot_visualise_image_colour(img, title):
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)

    @staticmethod
    def preprocess_image(image):
        # format: png

        # check grayscale
        if image.ndim > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #1 channel grayscale image
            return gray
        else:
            return image

    def preprocess_carrier_image(self):
        preprocessedImg = self.preprocess_image(self.carrierImg)

        # update
        self.carrierImg = preprocessedImg
        self.carrierImgHeight, self.carrierImgWidth = self.carrierImg.shape # (y, x)


    @staticmethod
    def scale_down_7x7(img):
        size = 7 # must be odd number
        
        resizedImg = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return resizedImg

    @staticmethod  
    def gray_to_binary(img, threshold=128):
        # Apply binary threshold
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def preprocess_watermark_image(self):
        preprocessedWatermarkImg = self.preprocess_image(self.watermarkImg)

        # scale
        resizedPreprocessedWatermarkImg = self.scale_down_7x7(preprocessedWatermarkImg)

        # gray to binary
        binaryResizedPreprocessedWatermarkImg = self.gray_to_binary(resizedPreprocessedWatermarkImg) # 0 and 255
        binaryWatermarkImg = binaryResizedPreprocessedWatermarkImg // 255 # 0 and 1

        # update
        self.watermarkImg = binaryWatermarkImg
        self.watermarkImgHeight, self.watermarkImgWidth = self.watermarkImg.shape # (y, x)

    def detect_keypoints(self): 
        img = self.carrierImg # grayscale image

        sift = cv2.SIFT_create(contrastThreshold=0.2) 
        kp = sift.detect(img, None)

        # Sort by response (strength) in descending order
        keypoints_sorted = sorted(kp, key=lambda kp: kp.response, reverse=True)

        return keypoints_sorted
    
    @staticmethod
    def does_watermark_collide(watermarkPosition, wICCHeight, wICCWidth, x, y) -> bool:
        # remove colliding (if watermarkImg will collide)
        for xOffset in range (-wICCHeight, wICCHeight + 1): # exclusive, so +1 to include
            for yOffset in range (-wICCWidth, wICCWidth +1): # exclusive, so +1 to include
                if watermarkPosition[y+yOffset, x+xOffset]: # if 1, true, pixel already watermarked
                    return True
        return False

    def preprocess_keypoints(self, siftKeypoints):
        carrierImgHeight = self.carrierImgHeight
        carrierImgWidth = self.carrierImgWidth

        roundedCoordSiftKeypoints = []
        watermarkPosition = np.zeros((carrierImgHeight,carrierImgWidth), dtype=self.carrierImg.dtype)  # 0 matrix, same shape and type as pCarrierImg
        wICCHeight, wICCWidth = (self.watermarkImgHeight // 2, self.watermarkImgWidth // 2); # watermarkImgCentreCoord
        watermarkHere = np.ones((self.watermarkImgHeight,self.watermarkImgWidth), dtype=self.watermarkImg.dtype)

        # if pWatermarkImg is larger than pCarrierImg
        if wICCWidth > carrierImgWidth-wICCWidth-1 or wICCHeight > carrierImgHeight-wICCHeight-1: # x exceeds or y exceeds
            return roundedCoordSiftKeypoints # an empty array

        # to identify cropping: add corner points (top left corner, bottom right corner)
        
        x = wICCWidth # leftmost
        y = wICCHeight # topmost
        cornerPoint = (x,y)
        watermarkPosition[y-wICCHeight:y+wICCHeight+1, x-wICCWidth:x+wICCWidth+1] = cv2.add(watermarkPosition[y-wICCHeight:y+wICCHeight+1, x-wICCWidth:x+wICCWidth+1], watermarkHere)  # simple add
        roundedCoordSiftKeypoints.append(cornerPoint)
        #print(f"corner coord boundaries, sift point (x={x}, y={y}) added")

        x = carrierImgWidth-wICCWidth-1 # rightmost
        y = carrierImgHeight-wICCHeight-1 # bottommost
        cornerPoint = (x,y)
        watermarkPosition[y-wICCHeight:y+wICCHeight+1, x-wICCWidth:x+wICCWidth+1] = cv2.add(watermarkPosition[y-wICCHeight:y+wICCHeight+1, x-wICCWidth:x+wICCWidth+1], watermarkHere)  # simple add
        roundedCoordSiftKeypoints.append(cornerPoint)
        #print(f"corner coord boundaries, sift point (x={x}, y={y}) added")

        for siftKeypoint in siftKeypoints:
            #print(f"siftKeypoint (y={siftKeypoint.pt[0]}, x={siftKeypoint.pt[1]})")

            # problem: siftpoints shift a bit .6, .4 so round isnt good enough (addressed in compare_siftpoint_watermark)
            roundOffNo = 1
            roundedCoordSiftKeypoint = (roundOffNo*round(siftKeypoint.pt[0]/roundOffNo),roundOffNo*round(siftKeypoint.pt[1]/roundOffNo)) # round to nearest roundOffNo

            x = roundedCoordSiftKeypoint[0]
            y = roundedCoordSiftKeypoint[1]

            # move keypoint if adding watermarkImg will exceed boundary
            if x < wICCHeight or x > carrierImgWidth-wICCWidth-1:
                if x < wICCWidth or x > carrierImgWidth-wICCWidth-1:
                    #print("pWatermarkImg is larger than pCarrierImg")
                    continue
                #print(f"x coord exceeded, sift point (x={x}, y={y}) removed")
                if x < wICCWidth:
                    x = wICCWidth # move keypoint to possible location
                elif x > carrierImgWidth-wICCWidth-1:
                    x = carrierImgWidth-wICCWidth-1 # move keypoint to possible location
                
            if y < wICCHeight or y > carrierImgHeight-wICCHeight-1:
                #print(f"y coord exceeded, sift point (x={x}, y={y}) removed")
                if y < wICCHeight or y > carrierImgWidth-wICCHeight-1:
                    #print("pWatermarkImg is larger than pCarrierImg")
                    continue
                #print(f"x coord exceeded, sift point (x={x}, y={y}) removed")
                if y < wICCHeight:
                    y = wICCHeight # move keypoint to possible location
                elif y > carrierImgHeight-wICCHeight-1:
                    y = carrierImgHeight-wICCHeight-1 # move keypoint to possible location

            # remove if collide
            watermarkCollide = self.does_watermark_collide(watermarkPosition, wICCHeight, wICCWidth, x, y)
            if watermarkCollide:
                #print(f"watermark collision, sift point (x={x}, y={y}) removed")
                continue
            
            # add if within boundary
            watermarkPosition[y-wICCWidth:y+wICCWidth+1, x-wICCHeight:x+wICCHeight+1] = cv2.add(watermarkPosition[y-wICCWidth:y+wICCWidth+1, x-wICCHeight:x+wICCHeight+1], watermarkHere)  # simple add

            #print(f"within coord boundaries, sift point (x={x}, y={y}) added")
            roundedCoordSiftKeypoints.append(roundedCoordSiftKeypoint)
            
        #print(f"Number of roundedCoordSiftKeypoints are {len(roundedCoordSiftKeypoints)}")
        #print(roundedCoordSiftKeypoints)
        return roundedCoordSiftKeypoints
    
    def compare_siftpoint_watermark(self, roundedCoordSiftKeypoint, wICCHeight, wICCWidth, checkNeighbours):
        carrierImg = self.carrierImg
        watermarkImg = self.watermarkImg
        carrierImgWidth = self.carrierImgWidth
        carrierImgHeight = self.carrierImgHeight

        roundOffNo = 1

        x = roundedCoordSiftKeypoint[0]
        y = roundedCoordSiftKeypoint[1]
        #print(f"Searching for embedded pixel at coordinates (x={x}, y={y})")

        # Return false if overlay is out of bounds
        if x < wICCHeight or x > carrierImgWidth-wICCWidth-1:
            watermarkExists = False
            return watermarkExists

        if y < wICCHeight or y > carrierImgHeight-wICCHeight-1:
            watermarkExists = False
            return watermarkExists


        # Check for presence of pWatermarkImg LSB in embeddedImg LSB at roundedCoordSiftKeypoint

        for xOffset in range (-wICCWidth, wICCWidth + 1): # exclusive, so +1 to include
            for yOffset in range (-wICCHeight, wICCHeight +1): # exclusive, so +1 to include
        
                # Compare LSB
                if watermarkImg[wICCHeight+yOffset, wICCWidth+xOffset] & 0b00000001 == carrierImg[y+yOffset, x+xOffset] & 0b00000001:
                    # watermark exists
                    watermarkExists = True # still true
                elif watermarkImg[wICCHeight+yOffset, wICCWidth+xOffset] & 0b00000001 != carrierImg[y+yOffset, x+xOffset] & 0b00000001:
                    watermarkExists = False
                    #print(f"mismatch coord check at x = {x}, y = {y}")

                    if checkNeighbours:
                        checkNeighbours = False
                        neighbours = [(roundOffNo, 0), (roundOffNo, roundOffNo), (0, roundOffNo), (-roundOffNo, roundOffNo), (-roundOffNo, 0), (-roundOffNo, -roundOffNo), (0, -roundOffNo), (roundOffNo, -roundOffNo)]
                        for neighbour in neighbours:
                            neighbourSiftKeypoint = (roundedCoordSiftKeypoint[0]+neighbour[0], roundedCoordSiftKeypoint[1]+neighbour[1])
                            watermarkExists = self.compare_siftpoint_watermark(neighbourSiftKeypoint, wICCHeight, wICCWidth, checkNeighbours)
                            #print(watermarkExists)
                            if watermarkExists:
                                watermarkExists = True 
                                return watermarkExists
                    return watermarkExists # false
                else:
                    print("not supposed to reach here")
        return watermarkExists # true



class WatermarkEncoding(WatermarkProcessor):
    def __init__(self, carrierImgPath, watermarkImgPath):
        super().__init__(carrierImgPath, watermarkImgPath)  # Calls WatermarkProcessor's __init__()
        # self.salary = salary # other variables here

    def embed_carrier_img(self, roundedCoordSiftKeypoints):
        carrierImg = self.carrierImg
        watermarkImg = self.watermarkImg
    
        wICCHeight, wICCWidth = (self.watermarkImgHeight // 2, self.watermarkImgWidth // 2); # watermarkImgCentreCoord
        
        # Encoding

        for roundedCoordSiftKeypoint in roundedCoordSiftKeypoints:
            # Alter embeddedImg LSB to pWatermarkImg at roundedCoordSiftKeypoint

            x = roundedCoordSiftKeypoint[0]
            y = roundedCoordSiftKeypoint[1]
            #print(f"Embedded pixel at coordinates (y={y}, x={x})")

            for xOffset in range (-wICCWidth, wICCWidth + 1): # exclusive, so +1 to include
                for yOffset in range (-wICCHeight, wICCHeight +1): # exclusive, so +1 to include            
                    if watermarkImg[wICCHeight+yOffset, wICCWidth+xOffset] == 0:
                        # if watermarkImg pixel is 0, clear LSB for carrierImg
                        carrierImg[y+yOffset, x+xOffset] = carrierImg[y+yOffset, x+xOffset] & 0b11111110 # clear bit
                    elif watermarkImg[wICCHeight+yOffset, wICCWidth+xOffset] == 1:
                        # if watermarkImg pixel is 1, set LSB for carrierImg
                        carrierImg[y+yOffset, x+xOffset] = carrierImg[y+yOffset, x+xOffset] | 0b00000001 # set bit
                    else:
                        print("not supposed to reach here")

        # update
        self.carrierImg = carrierImg 


    def watermark_encoding(self):
        self.preprocess_carrier_image()
        self.preprocess_watermark_image()

        siftKeypoints = self.detect_keypoints()
        roundedCoordSiftKeypoints = self.preprocess_keypoints(siftKeypoints)

        self.embed_carrier_img(roundedCoordSiftKeypoints)


class AuthenticityVerifier(WatermarkProcessor):
    def __init__(self, carrierImgPath, watermarkImgPath):
        super().__init__(carrierImgPath, watermarkImgPath)  # Calls Person's __init__()
        self.isAuthenticMessage = None #string

    def verify_authenticity(self, roundedCoordSiftKeypoints):
        wICCHeight, wICCWidth = (self.watermarkImgHeight // 2, self.watermarkImgWidth // 2); # watermarkImgCentreCoord
        watermarkExists = True

        # decoding
        for roundedCoordSiftKeypoint in roundedCoordSiftKeypoints:
            checkNeighbours = True
            matchingSiftpointWatermark = self.compare_siftpoint_watermark(roundedCoordSiftKeypoint, wICCHeight, wICCWidth, checkNeighbours)
            if matchingSiftpointWatermark == False:
                watermarkExists = False
                return watermarkExists
        return watermarkExists

    def authenticity_verifier(self):
        self.preprocess_carrier_image()
        self.preprocess_watermark_image()
    
        siftKeypoints = self.detect_keypoints()
        roundedCoordSiftKeypoints = self.preprocess_keypoints(siftKeypoints)

        isAuthentic = self.verify_authenticity(roundedCoordSiftKeypoints)
        if isAuthentic:
            self.isAuthenticMessage = "Yes"
        else:
            self.isAuthenticMessage = "No"


class TamperingDetector(WatermarkProcessor):
    def __init__(self, carrierImgPath, watermarkImgPath):
        super().__init__(carrierImgPath, watermarkImgPath)  # Calls Person's __init__()
        self.hasTamperingMessage = None #string
        self.identifiedTamperingImage = None

    def distinguish_authenticity(self, roundedCoordSiftKeypoints):
        wICCHeight, wICCWidth = (self.watermarkImgHeight // 2, self.watermarkImgWidth // 2); # watermarkImgCentreCoord
        watermarkExists = True
        missingKeyPointsCoords = []

        # decoding
        for roundedCoordSiftKeypoint in roundedCoordSiftKeypoints:
            checkNeighbours = True
            matchingSiftpointWatermark = self.compare_siftpoint_watermark(roundedCoordSiftKeypoint, wICCHeight, wICCWidth, checkNeighbours)
            if matchingSiftpointWatermark == False:
                missingKeyPointsCoords.append(roundedCoordSiftKeypoint)
                watermarkExists = False
            
        return watermarkExists, missingKeyPointsCoords

    @staticmethod
    def get_identified_tampering_img(identifiedTamperingImage, identifiedTamperingCoords):
        # colour image
        if identifiedTamperingImage.ndim < 3:
            #print(identifiedTamperingImage.ndim)

            identifiedTamperingImageColour = cv2.cvtColor(identifiedTamperingImage, cv2.COLOR_GRAY2BGR)
        else:
            identifiedTamperingImageColour = identifiedTamperingImage

        markerSize = identifiedTamperingImage.shape[1] // 10

        # Loop through each coordinate pair
        for identifiedTamperingCoord in identifiedTamperingCoords: # (x,y)
            cv2.drawMarker(identifiedTamperingImageColour, (identifiedTamperingCoord[0], identifiedTamperingCoord[1]), color = (0, 0, 255), markerType=cv2.MARKER_DIAMOND, 
            markerSize=markerSize, thickness=1, line_type=cv2.LINE_AA)

        return identifiedTamperingImageColour

    def tampering_detector(self):
        self.preprocess_carrier_image()
        self.preprocess_watermark_image()
        
        siftKeypoints = self.detect_keypoints()
        roundedCoordSiftKeypoints = self.preprocess_keypoints(siftKeypoints)

        isAuthentic, identifiedTamperingCoords = self.distinguish_authenticity(roundedCoordSiftKeypoints)
        identifiedTamperingImage = self.get_identified_tampering_img(self.carrierImg, identifiedTamperingCoords)

        if isAuthentic:
            self.hasTamperingMessage = "No"
        else:
            self.hasTamperingMessage = "Yes"
        self.identifiedTamperingImage = identifiedTamperingImage
        
        