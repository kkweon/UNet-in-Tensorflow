# U-Net Implementation in TensorFlow

<img src="assets/output.gif" width=1024 />

Re implementation of U-Net in Tensorflow
- to check how image segmentations can be used for detection problems

Original Paper
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Summary

Vehicle Detection using U-Net

Objective: detect vehicles
Find a function f such that y = f(X)
<table>
    <tr>
        <th>Input</th>
        <th>Shape</th>
        <th>Explanation</th>
        <th>Example</th>
    </tr>
    <tr>
        <td>X: 3-D Tensor</td>
        <td>(640, 960, 3)</td>
        <td>RGB image in an array</td>
        <td><img src="assets/example_input.jpg" width=320 /></td>
    </tr>
    <tr>
        <td>y: 3-D Tensor</td>
        <td>(640, 960, 1)</td>
        <td>Binarized image. Bacground is 0<br />vehicle is masked as 255</td>
        <td><img src="assets/example_output.jpg" width=320 /></td>
    </tr>
</table>

Loss function: maximize IOU
```
    (intersection of prediction & grount truth)
    -------------------------------------------
    (union of prediction & ground truth)
```

### Examples on Test Data: trained for 3 epochs
<img src="assets/result1.png" />
<img src="assets/result2.png" />
<img src="assets/result3.png" />
<img src="assets/result4.png" />
<img src="assets/result5.png" />
<img src="assets/result6.png" />
<img src="assets/result7.png" />


## Get Started

### Download dataset

```bash
make download
```


### Resize image and generate mask images

```bash
make generate
```

### Train Test Split

Make sure masks and bounding boxes

```bash
jupyter notebook "Visualization & Train Test Split.ipynb"
```
### Train

```bash
python train.py
```
