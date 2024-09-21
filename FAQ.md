# Problem Summary

The blog with a summary of the issues can be found at [this link](https://blog.csdn.net/weixin_44791964/article/details/107517428).

## 1. Download Issues

### a. Code Download

**Q: Can you send me the code? Where can I download it?**  
**A: The GitHub link is in the video description. Just copy it and you’ll be able to download the code.**

**Q: Why is the downloaded code showing that the zip file is corrupted?**  
**A: Please download the code again from GitHub.**

**Q: Why is the code I downloaded different from what’s shown in the video or blog?**  
**A: I frequently update the code, so the latest version should always be considered as the accurate one.**

### b. Model Weight Download

**Q: Why is there no .pth or .h5 file under `model_data` in the downloaded code?**  
**A: I usually upload the weights to GitHub and Baidu Cloud. You can find them in the GitHub README.**

### c. Dataset Download

**Q: Where can I download the XXXX dataset?**  
**A: I usually place the download link in the README. If you can't find it, please contact me to add it. You can also raise an issue on GitHub.**

## 2. Environment Setup Issues

### a. Current Libraries and Versions

**Pytorch code corresponds to version 1.2, see the blog post** [here](https://blog.csdn.net/weixin_44791964/article/details/106037141).  
**Keras code corresponds to TensorFlow version 1.13.2, Keras version is 2.1.5, see the blog post** [here](https://blog.csdn.net/weixin_44791964/article/details/104702142).  
**TensorFlow 2 code corresponds to version 2.2.0, no need to install Keras separately, see the blog post** [here](https://blog.csdn.net/weixin_44791964/article/details/109161493).

**Q: Can I use other versions of TensorFlow or PyTorch with your code?**  
**A: It's best to stick to the versions I recommend. I haven't tested other versions, but they should generally work with minor code adjustments.**

### b. 30 Series GPU Setup

Due to framework updates, the usual setup tutorials won't work for 30 series GPUs.  
I’ve tested the following configurations with 30 series GPUs:

- **Pytorch code**: Pytorch version 1.7.0, CUDA 11.0, cuDNN 8.0.5  
- **Keras code**: Can’t configure CUDA 11 on Windows 10. It works on Ubuntu, with TensorFlow version 1.15.4, and Keras version 2.1.5 or 2.3.1 (some functions may need adjustments).  
- **TensorFlow 2 code**: TensorFlow version 2.4.0, CUDA 11.0, cuDNN 8.0.5

### c. GPU Utilization and Environment Setup

**Q: Why am I not using the GPU despite having installed TensorFlow-GPU?**  
**A: Check that TensorFlow-GPU is installed by using `pip list` to verify the version. Then use the task manager or NVIDIA commands to see if the GPU is being used.**

![Task Manager Example](https://img-blog.csdnimg.cn/20201013234241524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)

**Q: Why is my GPU not being utilized for training?**  
**A: Check the GPU usage using NVIDIA's command line tools or by monitoring GPU memory usage in the task manager.**

### d. Import Errors (no module)

**Q: Why do I get "No module named utils.utils" or similar errors?**  
**A: These modules are in the repository root directory. The error occurs when the root directory is incorrectly set. Make sure the relative and root paths are correct.**

**Q: Why does "No module named matplotlib/PIL/cv2" keep appearing?**  
**A: These libraries aren’t installed. Use the command `pip install matplotlib`.**

### e. CUDA Installation Issues

**CUDA installation generally requires Visual Studio. Version 2017 works fine.**

### f. Ubuntu System

**All code works on Ubuntu. I've tested it on both Windows and Ubuntu systems.**

### g. VSCode Errors

**Q: Why does VSCode show a bunch of errors?**  
**A: It's a VSCode issue, but it doesn’t affect the code execution. If you don’t want to see the errors, switch to PyCharm.**

### h. Using CPU for Training and Prediction

- For Keras and TensorFlow 2 code, simply install the CPU version of TensorFlow.  
- For PyTorch code, change `cuda=True` to `cuda=False`.

### i. tqdm No `pos` Parameter Error

**Q: Why does the error `'tqdm' object has no attribute 'pos'` occur?**  
**A: Reinstall tqdm or try using a different version.**

### j. TypeError Related to Arrays

**Q: Why am I seeing the error `TypeError: __array__() takes 1 positional argument but 2 were given`?**  
**A: This can be resolved by installing Pillow version 8.2.0:**

```bash
pip install pillow==8.2.0

