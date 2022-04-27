# Document Scan

Using OpenCV with Pytesseract, this script opens a local image in your script folder and reads the text in the image and outputs it as text. This is designed to work specifically with documents inside of an image, as it searches the image for a document, then reads the text inside it.

The script does the following things:
1. Opens the image and pre-processes it to find the edges of the document in the photo.
2. Determines the coordinates of the points of the document by finding the larges 4 sided contour
3. Uses the contour to cut out and align the document from the image into a new image, replicating document scanning
4. Using Pytesseract, the text from the scanned document is read and converted into text for an output

Future Changes
- Apply more sophhisticated filtering to the scanned image to assist in text recognition.

How To Use
- A screenshot for a PDF has been included to test the script
- The script looks for a folder named "input.png" to be processed
- The images of the processing steps are displayed as the steps take place
- The script outputs text in the console for the resulting output
