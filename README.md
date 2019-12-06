# Image-Segmentaion
<figure>
<img src="./test/1.jpg" height="100" width="100"> <img src="./test/2.jpg" height="100" width="100"> <img src="./test/3.jpg" height="100" width="100">
<figcaption>Fig.2 input images</figcaption>


<img src="./mask/1.png" height="100" width="100"> <img src="./mask/1.png" height="100" width="100"> <img src="./mask/1.png" height="100" width="100">
<figcaption>Fig.2 Ground Truth</figcaption>


<img src="./pred/1.png" height="100" width="100"> <img src="./pred/2.png" height="100" width="100"> <img src="./pred/3.png" height="100" width="100">
<figcaption>Fig.3 Predicted Output</figcaption>
</figure>

## Train
Before training, model configuration  can be set in `config.py` file. After that open conda enviroemnt and change directory to project folder. Run `python train.py` to start training process. I trained this model for binary class segmentation using U-Net. Due to unavailability of GPU, it took 16 hours to train a model on CPU :blush:.

## Inference
* Open `stream_c.py` and locate yourself at `cv2.imread()` where you need to manually enter path of image that you want to input the model.

* Activate conda enviroment and run `stream_c.py`. I am not good at naming files, you can change if you want.


### Credits
Code and datset are taken from [@seth814](https://github.com/seth814). I modifed code files to work on CPU and interpret a input image rather stream of video, to perform semantic segmentation.
