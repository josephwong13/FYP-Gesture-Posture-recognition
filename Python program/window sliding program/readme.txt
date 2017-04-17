This program was developed for human recognition with the sliding window algorithm and Keras

This program has a MIT license,

Copyright <2017> <Wong Mun Hin>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


***Instruction***

To use the program, simply click the sliding_window_v1.exe, the program will guide you through setting.
For most of the question, input T/F for true or false. Read the following instruction for help.

1. >> Use webcam to fetch image (T/F)?
Select image fetching mode, if selected T, the 1.1 and 1.2 question will follow. Else, skip to 2.
1.1 >> Select camera to use
The first camera connected to the device is 0, then 1, then 2,3,4...
1.2 >> Press any key to take photo
Simply press a key to take the photo from the selected camera

2. >>Input file name
If not using the camera, the image file will be fetched from the input file. The name should contain the format
3. >>Use default setting (T/F)?
If F is selected, move on to 4

4.1 >>Input the list of frame
Define the frame size to use in the sliding window, use comma to separate. For example, input 0.1,0.2 would result in the 
algorithm first run with frame size = 10% image width, then with frame size = 2% image width

4.2 >>Input the moving step
The moving step is the pixels the frame moved everytime, larger moving step result in faster running speed, but less accuate

4.3 >>Input the detection delay
If delay > 0, the actual saved image are moved right and down for pixels equal to delay * moving step. For example, if
moving step is 2 and delay is 1, the saved image is moved 2 pixels right and down

4.4 >>Input the human ratio
The frame height to width ratio, frame height is defined by width * width ratio

4.5 >>Enable display mode?
The found human will be showed on window real time if enabled

4.6 >>Enable white filter?
The white filter, if enabled, will filter all frame with more than 50% white pixels

For more details, see my dissertation.

