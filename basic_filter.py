from PIL import Image, ImageFilter, ImageEnhance  # Importing the Image, ImageFilter, and ImageEnhance modules from the PIL library.
import matplotlib.pyplot as plt  # Importing the matplotlib.pyplot module for displaying and saving the image.

def apply_filters(image_path):  # Defining a function that applies multiple filters to an image.
    try:
        img = Image.open(image_path)  # Opens the image file specified by the given path.
        img_resized = img.resize((128, 128))  # Resizes the image to 128x128 pixels.

        # Apply Gaussian Blur filter
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))  # Applies a Gaussian blur filter with a radius of 2.
        plt.imshow(img_blurred)  # Displays the blurred image using matplotlib.
        plt.axis('off')  # Hides the axes in the displayed image.
        plt.savefig("blurred_image.png")  # Saves the processed image as 'blurred_image.png'.
        print("Processed image saved as 'blurred_image.png'.")  # Prints a message indicating that the image was successfully saved.

        # Apply Box Blur filter
        img_box_blur = img_resized.filter(ImageFilter.BoxBlur(radius=2))  # Applies a Box Blur filter with a radius of 2.
        plt.imshow(img_box_blur)  # Displays the Box Blurred image using matplotlib.
        plt.axis('off')  # Hides the axes in the displayed image.
        plt.savefig("box_blur_image.png")  # Saves the processed image as 'box_blur_image.png'.
        print("Processed image saved as 'box_blur_image.png'.")  # Prints a message indicating that the image was successfully saved.

        # Apply Contour filter
        img_contour = img_resized.filter(ImageFilter.CONTOUR)  # Applies a Contour filter to the image.
        plt.imshow(img_contour)  # Displays the Contour-filtered image using matplotlib.
        plt.axis('off')  # Hides the axes in the displayed image.
        plt.savefig("contour_image.png")  # Saves the processed image as 'contour_image.png'.
        print("Processed image saved as 'contour_image.png'.")  # Prints a message indicating that the image was successfully saved.

        # Apply Emboss filter
        img_emboss = img_resized.filter(ImageFilter.EMBOSS)  # Applies an Emboss filter to the image.
        plt.imshow(img_emboss)  # Displays the Emboss-filtered image using matplotlib.
        plt.axis('off')  # Hides the axes in the displayed image.
        plt.savefig("emboss_image.png")  # Saves the processed image as 'emboss_image.png'.
        print("Processed image saved as 'emboss_image.png'.")  # Prints a message indicating that the image was successfully saved.

        # Apply Abstract filter with sharper edges and brighter colors
        img_abstract = img_resized.filter(ImageFilter.EDGE_ENHANCE_MORE)  # Enhances the edges for a sharper effect.
        enhancer_color = ImageEnhance.Color(img_abstract)  # Initializes a color enhancer.
        img_abstract = enhancer_color.enhance(3.5)  # Exaggerates the colors by increasing saturation.
        enhancer_contrast = ImageEnhance.Contrast(img_abstract)  # Initializes a contrast enhancer.
        img_abstract = enhancer_contrast.enhance(3.0)  # Boosts the contrast for sharper and more vivid shapes.
        enhancer_brightness = ImageEnhance.Brightness(img_abstract)  # Initializes a brightness enhancer.
        img_abstract = enhancer_brightness.enhance(1.8)  # Increases the brightness for a more vibrant look.
        plt.imshow(img_abstract)  # Displays the Abstract-filtered image using matplotlib.
        plt.axis('off')  # Hides the axes in the displayed image.
        plt.savefig("abstract_image.png")  # Saves the processed image as 'abstract_image.png'.
        print("Processed image saved as 'abstract_image.png'.")  # Prints a message indicating that the image was successfully saved.

    except Exception as e:  # Handles any exceptions that may occur.
        print(f"Error processing image: {e}")  # Prints an error message if there is an exception.

if __name__ == "__main__":  # Checks if the script is being run directly (not imported as a module).
    image_path = "basic_cat.jpg"  # Specifies the path to the input image file.
    apply_filters(image_path)  # Calls the apply_filters function with the specified image path.
