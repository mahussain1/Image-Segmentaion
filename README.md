# Image-Segmentaion

<img src="./test/1.jpg" height="100" width="100"> <img src="./test/2.jpg" height="100" width="100"> <img src="./test/3.jpg" height="100" width="100">

  *input images*

<img src="./mask/1.png" height="100" width="100"> <img src="./mask/1.png" height="100" width="100"> <img src="./mask/1.png" height="100" width="100">

 *Ground Truth*

<img src="./pred/1.png" height="100" width="100"> <img src="./pred/2.png" height="100" width="100"> <img src="./pred/3.png" height="100" width="100">

 *Predicted Output*

##Train
For training, model configuration can be set in `config.py` file. I trained this model for binary class segmentation using U-Net. Due to unavailability of GPU, it took 16 hours to train a model on CPU.


## Model File


Note: Code and custom datset are taken from [@seth814](https://github.com/seth814). I changed code files to work on CPU and interpret a input image to perform semantic segmentation.
