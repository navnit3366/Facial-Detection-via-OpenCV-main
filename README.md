# Facial-Detection-via-OpenCV
A simple Python code that recognize the faces of humans, and rarely, faces of ghosts. More features coming soon as I continue to study the OpenCV documentation.

## About
This is a project by Lin as a way to challenge himself to create a facial detection software within 2 days. As he is typing this right now, he is currently under immense headache from the back all the way to the front. If you wish to add more features into this project, you are free to contribute.

### HOW TO
To use the code for your personal project, follow these easy steps (read everything first before doing so):
1. Crop and copy the images of the faces of people you want the computer to identify to the `resx` folder. Using different folders, separate the images to their corresponding folder names (use their real name, or not; depends if you want to make a fool of somebody by labelling them something else). The name of the folders will be used as labels. The name of the images doesn't matter.
2. Run `os-walk.py`. This will create the `trainer.yml` and `labels.pickle` files. These files contain the data necessary for person identification for `main.py`.
3. Run `main.py` and enjoy!
4. `coco-detect.py` is for object detection. It utilizes the files `frozen_inference_graph.pb` and `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`, so it doesn't require steps #1-#3. You can simply run this as is!

<!--
> I think this will be my last commit to this repo. I'm no longer touching data science until I learn the maths about it.

Dear future me,

Make sure you cringe at your past, you piece of shit. You were nothing but a fraud that just watched YouTube videos and copy-pasted codes from documentations and sites but still show off to you friends in Facebook. Hope you're no longer coding. Sayonara, fucker. 
-->
