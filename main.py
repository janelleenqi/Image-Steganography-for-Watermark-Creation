import tkinter # GUI
import customtkinter # GUI
from PIL import Image, ImageTk # to display image in GUi
import cv2
import os
from assignment import WatermarkEncoding, AuthenticityVerifier, TamperingDetector


# aesthetics
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class ImageFrame(customtkinter.CTkScrollableFrame): # for image displays
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure((0, 1), weight=1)

        # Add Widgets in Frame

        # Title
        self.resultsTitle = customtkinter.CTkLabel(self, text = "")
        self.resultsTitle.grid(row=3, column=0, columnspan=2, sticky="ew")

        # Input
        self.display1 = customtkinter.CTkLabel(self, image = None, text = "")
        self.display1.grid(row=4, column=0, columnspan=1, sticky="ew")

        self.displayDescription1 = customtkinter.CTkLabel(self, text = "")
        self.displayDescription1.grid(row=5, column=0, columnspan=1, sticky="ew")

        self.display2 = customtkinter.CTkLabel(self, image = None, text = "")
        self.display2.grid(row=4, column=1, columnspan=1, sticky="ew")

        self.displayDescription2 = customtkinter.CTkLabel(self, text = "")
        self.displayDescription2.grid(row=5, column=1, columnspan=1, sticky="ew")

        # Output
        self.display3 = customtkinter.CTkLabel(self, image = None, text = "")
        self.display3.grid(row=6, column=0, columnspan=2, sticky="ew")

        self.displayDescription3 = customtkinter.CTkLabel(self, text = "")
        self.displayDescription3.grid(row=7, column=0, columnspan=2, sticky="ew")

    @property # access like a variable
    def frame_width(self) -> int:
        self.update()
        return self.winfo_width()
    
    def display_result(self, selectedImage, selectedDesc):
        # Rescale to Fit
        if selectedImage == None:
            emptyPilImg = Image.new("RGBA", (1, 1), (255, 255, 255, 0))  # transparent
            emptyCtkImg = customtkinter.CTkImage(light_image=emptyPilImg, size=(1, 1))
            self.display3.configure(image = emptyCtkImg, text = "")
        else:
            newWidth = self.frame_width  
            newHeight = int(newWidth/selectedImage.width * selectedImage.height)  # calculates your new height

            self.displayImg3 = customtkinter.CTkImage(light_image = selectedImage, size=(newWidth, newHeight))
            self.display3.configure(image = self.displayImg3, text = "")

        self.displayDescription3.configure(text = selectedDesc)

    def display_results_title(self, selectedDesc):
        self.resultsTitle.configure(text = selectedDesc)


    def display_selected_images(self, selectedImage1, selectedImageDesc1, selectedImage2, selectedImageDesc2):

        # Rescale to Fit
        newWidth = self.frame_width // 2
        newHeight1 = int(newWidth/selectedImage1.width * selectedImage1.height)  # calculates your new height

        self.displayImg1 = customtkinter.CTkImage(light_image = selectedImage1, size=(newWidth, newHeight1))
        self.display1.configure(image = self.displayImg1, text = "")

        self.displayDescription1.configure(text = selectedImageDesc1)

        # Rescale to Fit
        newHeight2 = int(newWidth/selectedImage2.width * selectedImage2.height)  # calculates your new height

        self.displayImg2 = customtkinter.CTkImage(light_image = selectedImage2, size=(newWidth, newHeight2))
        self.display2.configure(image = self.displayImg2, text = "")

        self.displayDescription2.configure(text = selectedImageDesc2)

# App Frame
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("720x960")
        self.title("Image-to-Image Steganography")
        self.grid_rowconfigure((0,1,2), weight=1)  # configure grid system 
        self.grid_rowconfigure((3), weight=40) 
        self.grid_columnconfigure((0, 1), weight=1) # give columns 0 and 1 equal weight
        

        # UI Elements
        self.watermarkEmbedder = customtkinter.CTkButton(self, text = "Embed Watermark", command=lambda: self.embed_watermark())
        self.watermarkEmbedder.grid(row=0, columnspan=2, padx = 20, pady = 10)

        self.watermarkRecovery = customtkinter.CTkButton(self, text = "Recover Watermark", command=lambda: self.recover_watermark())
        self.watermarkRecovery.grid(row=1, columnspan=2, padx = 20, pady = 10)

        self.tamperingDetector = customtkinter.CTkButton(self, text = "Detect Tampering", command=lambda: self.detect_tampering())
        self.tamperingDetector.grid(row=2, columnspan=2, padx = 20, pady = 10)

        self.display = ImageFrame(master=self)
        self.display.grid(row=3, padx=20, pady=10, sticky="nsew", columnspan=2)

        # Button for closing
        self.exit_button = customtkinter.CTkButton(self, text="Exit", command=self.destroy)
        self.exit_button.grid(row=8, padx=20, pady=10, columnspan=2)

    
    # Functions
    def select_image(self, imageDesc):

        path = customtkinter.filedialog.askopenfilename()

        # ensure a file path was selected
        if len(path) > 0:

            selectedImage = Image.open(path)
            self.display.display_image(selectedImage, imageDesc)

 
    def select_images(self, carrierImageDesc):
        filetypes = ( ('image files', '*.png *.tif'), ('All files', '*.*') )

        carrierImgPath = customtkinter.filedialog.askopenfilename(title=str(f'Select {carrierImageDesc}'), initialdir=os.getcwd(),filetypes=filetypes)

        # ensure a file path was selected
        if len(carrierImgPath) > 0:
            selectedCarrierImage = Image.open(carrierImgPath)
        watermarkImgPath = customtkinter.filedialog.askopenfilename(title='Select Watermark Image', initialdir=os.getcwd(),filetypes=filetypes)

        # ensure a file path was selected
        if len(watermarkImgPath) > 0:
            selectedWatermarkImage = Image.open(watermarkImgPath)
        
        self.display.display_selected_images(selectedCarrierImage, carrierImageDesc, selectedWatermarkImage, "Selected Watermark Image")

        return carrierImgPath, watermarkImgPath
    
    def save_image(self, WatermarkProcessorObject, imageDesc):
        imgPath = customtkinter.filedialog.askdirectory(title='Select Directory to Save Image', initialdir=os.getcwd())
        WatermarkProcessorObject.save_img(imgPath, imageDesc)
    
    def show_results(self, resultsTitle, outputImg, outputImgDesc):
        self.display.display_results_title(resultsTitle)
        self.display.display_result(outputImg, outputImgDesc)


    def embed_watermark(self):
        carrierImgPath, watermarkImgPath = self.select_images("Image to Embed Watermark")

        watermarkEncodingObject = WatermarkEncoding(carrierImgPath, watermarkImgPath)
        watermarkEncodingObject.watermark_encoding()

        self.save_image(watermarkEncodingObject, "WatermarkEncodedImg")
        self.show_results("Embed Watermark Results", Image.fromarray(watermarkEncodingObject.carrierImg), "Embed Image")

    def recover_watermark(self):
        carrierImgPath, watermarkImgPath = self.select_images("Image to Check for Watermark")
        authenticityVerifierObject = AuthenticityVerifier(carrierImgPath, watermarkImgPath)
        authenticityVerifierObject.authenticity_verifier()

        self.show_results("Recover Watermark Results", None, str(f"Does the image contain the watermark? {authenticityVerifierObject.isAuthenticMessage}"))


    def detect_tampering(self):
        carrierImgPath, watermarkImgPath = self.select_images("Image that May be Tampered")
        tamperingDetectorObject = TamperingDetector(carrierImgPath, watermarkImgPath)
        tamperingDetectorObject.tampering_detector()

        self.show_results("Detect Tampering Results", Image.fromarray(tamperingDetectorObject.identifiedTamperingImage), str(f"Is tampering detected? {tamperingDetectorObject.hasTamperingMessage}"))


# Instantiate app
app = App()

# Run app
app.mainloop()
